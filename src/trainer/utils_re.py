from typing import List, Tuple, Dict
import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab
import tokenizers as tk
import torch.nn.functional as F
import torch.profiler
from time import time

from utils import pred_token_within_range, subsequent_mask
from vocab import (
    HTML_TOKENS,
    TASK_TOKENS,
    RESERVED_TOKENS,
    BBOX_TOKENS,
    OTSL_TOKENS
)


VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
VALID_OTSL_TOKEN = ["<eos>"] + OTSL_TOKENS
INVALID_CELL_TOKEN = (
    ["<sos>", "<pad>", "<empty>", "<sep>"] + TASK_TOKENS + RESERVED_TOKENS
)
VALID_BBOX_TOKEN = [
    "<eos>"
] + BBOX_TOKENS  # image size will be addressed after instantiation

# MIX_HTML_TOKENS와 VALID_MIX_TOKENS 생성
# VALID_HTML_TOKEN : 49
# INVALID_CELL_TOKEN : 18
# VALID_BBOX_TOKEN : 881




class Batch_MIX:
    """Wrap up a batch of training samples with different training targets.
    The input is not torch tensor
    Shape of the image (src): B, S, E
    Shape of the text (tgt): B, N, S, E (M includes 1 table detection, 1 structure, 1 cell, and multiple bbox)
    Reshape text to (B * N, S, E) and inflate the image to match the shape of the text
    """

    def __init__(
        self,
        device: torch.device,
        target: str,
        vocab_html: Vocab,  # ✅
        vocab_bbox: Vocab,  # ✅
        obj: List,  # ✅
        use_mix_loss = False,  # ✅
        otsl_mode: bool = True,  # ✅
        only_structure: bool = False,
    ) -> None:

        self.device = device
        self.image = obj[0].to(device)
        self.name = obj[1]["filename"]
        self.target = target
        self.vocab_html = vocab_html  # ✅
        self.vocab_bbox = vocab_bbox  # ✅
        self.image_size = self.image.shape[-1]

        self.use_mix_loss = use_mix_loss
        self.otsl_mode = otsl_mode
        self.only_structure = only_structure

        # grammer = 'OTSL' if otsl_mode else 'HTML'
        # print(f'✅ [ GRAMMER ] : {grammer} (utils.py)')
        # print(self.vocab_html)
        # vocab_dict = self.vocab_html.get_vocab()  
        # print(f"총 어휘 수: {len(vocab_dict)}")
        # for token, idx in vocab_dict.items():
        #     print(f"{idx:4d}  : {token}")

        
        if "table" in target:
            raise NotImplementedError

        # valid token 설정
        if otsl_mode:
            self.valid_html_token = [vocab_html.token_to_id(i) for i in VALID_OTSL_TOKEN]
        else:
            self.valid_html_token = [vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN]


        (
            self.html_src,
            self.html_tgt,
            self.html_casual_mask,
            self.html_padding_mask,
        ) = self._prepare_transformer_input(obj[1]["html"])  # ✅

        (
            self.bbox_src,
            self.bbox_tgt,
            self.bbox_casual_mask,
            self.bbox_padding_mask,
        ) = self._prepare_transformer_input(obj[1]["bbox"])  # ✅

        vocab_html_dict = self.vocab_html.get_vocab()
        vocab_bbox_dict = self.vocab_bbox.get_vocab()


        if use_mix_loss:
            (
                self.old_to_new_mapping_html,
                self.old_to_new_mapping_bbox,
                self.html_target_indices,
                self.bbox_target_indices,
            ) = create_mix_tokens(vocab_html_dict, vocab_bbox_dict)

            # ★ 상수 텐서 캐싱: 고정 target 값들은 매 forward마다 재생성하지 않고 캐싱
            self.html_target_tensor = torch.tensor(self.html_target_indices, device=self.device)
            self.bbox_target_tensor = torch.tensor(self.bbox_target_indices, device=self.device)
            self.constant_two = torch.tensor(2, device=self.device)

            if not hasattr(self, '_html_mapping_table'):
                vocab_size_html = max(self.old_to_new_mapping_html.keys()) + 1
                mapping_table = torch.zeros((vocab_size_html, 2), dtype=torch.long, device=self.device)
                is_expanded = torch.zeros(vocab_size_html, dtype=torch.bool, device=self.device)
                for token_id, mapping in self.old_to_new_mapping_html.items():
                    if isinstance(mapping, dict):
                        mapping_table[token_id, 0] = mapping["left"]
                        mapping_table[token_id, 1] = mapping["right"]
                        is_expanded[token_id] = True
                    else:
                        mapping_table[token_id, 0] = mapping
                        # 두 번째 슬롯은 사용하지 않으므로 0으로 남김
                self._html_mapping_table = mapping_table
                self._html_is_expanded = is_expanded


    def _prepare_transformer_input(
        self, seq: List[tk.Encoding]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tmp = [i.ids for i in seq]
        tmp = torch.tensor(tmp, dtype=torch.int32)
        src = tmp[:, :-1].to(self.device)
        tgt = tmp[:, 1:].type(torch.LongTensor).to(self.device)
        casual_mask = subsequent_mask(src.shape[-1]).to(self.device)
        tmp = [i.attention_mask[:-1] for i in seq]  # padding mask
        tmp = torch.tensor(tmp, dtype=torch.bool)
        padding_mask = (~tmp).to(self.device)

        return src, tgt, casual_mask, padding_mask


    def _inference_one_task(
        self, model, memory, src, casual_mask, padding_mask, use_ddp
    ):
        if use_ddp:
            out = model.module.decode(memory, src, casual_mask, padding_mask)
            out = model.module.generator(out)
        else:
            out = model.decode(memory, src, casual_mask, padding_mask)
            out = model.generator(out)

        return out


    def inference(
        self,
        model_html: nn.Module,  # ✅
        model_bbox: nn.Module,  # ✅
        criterion_html: nn.Module,
        criterion_bbox: nn.Module,
        criterion_mix: nn.Module,
        loss_weights: dict = None,
        use_ddp: bool = True,
    ) -> Tuple[Dict, Dict]:
        pred = dict()
        loss = dict(table=0, html=0, cell=0, bbox=0)

        if use_ddp:
            memory_html = model_html.module.encode(self.image)  # ✅
            if not self.only_structure:
                memory_bbox = model_bbox.module.encode(self.image)  # ✅
        else:
            memory_html = model_html.encode(self.image)
            if not self.only_structure:
                memory_bbox = model_bbox.encode(self.image)

        # inference + suppress invalid logits + compute loss

        # inference html
        st = time()
        out_html = self._inference_one_task(
            model_html,
            memory_html,
            self.html_src,
            self.html_casual_mask,
            self.html_padding_mask,
            use_ddp,
        )

        pred_html_logits = pred_token_within_range(
            out_html, white_list=self.valid_html_token
        ).permute(0, 2, 1)
        pred["html"] = pred_html_logits
        loss["html"] = criterion_html(pred_html_logits, self.html_tgt)
        time_inf_html = time() - st

        # inference bbox
        st = time()
        if not self.only_structure:
            out_bbox = self._inference_one_task(
                model_bbox,
                memory_bbox,
                self.bbox_src,
                self.bbox_casual_mask,
                self.bbox_padding_mask,
                use_ddp,
            )

            pred_bbox_logits = out_bbox.permute(0, 2, 1)
            pred["bbox"] = pred_bbox_logits
            loss["bbox"] = criterion_bbox(pred_bbox_logits, self.bbox_tgt)
        else:
            loss['bbox'] = torch.tensor(0, device=self.device, dtype=torch.int64)
        time_inf_bbox = time() - st


        # Mix Token OTSL 버전도 만들어야 함
        st = time()
        if self.use_mix_loss:
            # HTML
            probs_html = F.softmax(out_html, dim=-1)  # shape: (B, S, num_html_tokens)
            probs_token_html = probs_html.argmax(dim=-1)  # shape: (B, S)

            # BBOX
            probs_bbox = F.softmax(out_bbox, dim=-1)  # shape: (B, S, num_bbox_tokens)
            probs_token_bbox = probs_bbox.argmax(dim=-1)  # shape: (B, S)

            st = time()
            # padding 처리: eos 이후는 pad 값(2)로 대체
            eos_mask = probs_token_html == 1  # (B, S)
            first_eos_idx = torch.where(
                eos_mask.any(dim=1),
                eos_mask.float().argmax(dim=1),
                torch.full((probs_token_html.size(0),), probs_token_html.size(1), device=self.device),
            )
            seq_range = torch.arange(probs_token_html.size(1), device=self.device).unsqueeze(0)
            pad_mask = seq_range > first_eos_idx.unsqueeze(1)
            probs_token_html = torch.where(pad_mask, self.constant_two, probs_token_html)

            eos_mask = probs_token_bbox == 1  # (B, S)
            first_eos_idx = torch.where(
                eos_mask.any(dim=1),
                eos_mask.float().argmax(dim=1),
                torch.full((probs_token_bbox.size(0),), probs_token_bbox.size(1), device=self.device),
            )
            seq_range = torch.arange(probs_token_bbox.size(1), device=self.device).unsqueeze(0)
            pad_mask = seq_range > first_eos_idx.unsqueeze(1)
            probs_token_bbox = torch.where(pad_mask, self.constant_two, probs_token_bbox)

            # [ Loss 2 : Combined Tokens ]
            combined_tokens_tgt = self.combine(self.html_tgt, self.bbox_tgt)
            combined_tokens_pred = self.combine(probs_token_html, probs_token_bbox)
            combined_tokens_tgt, combined_tokens_pred = self.pad_to_same_length(
                combined_tokens_tgt, combined_tokens_pred, pad_value=2
            )
            new_loss_2_combine = criterion_mix(
                combined_tokens_tgt.float(), combined_tokens_pred.float())
            loss['mix_combine'] = new_loss_2_combine

            del combined_tokens_tgt, combined_tokens_pred, seq_range, pad_mask, probs_token_bbox,\
                probs_html, probs_token_html, probs_bbox, eos_mask, first_eos_idx
        else:
            # loss['mix_combine'] = 0
            loss['mix_combine'] = torch.tensor(0, device=self.device, dtype=torch.int64)
        inf_combine_result = time() - st
            
        
        time_dict = {
            "time_inf_html": time_inf_html,
            "time_inf_bbox": time_inf_bbox,
            "time_inf_combine": inf_combine_result}

        total = torch.tensor(0.0).to(self.device)
        if self.only_structure:
            loss["total"] = loss['html']
        else:
            for k, v in loss_weights.items():
                total += loss[k] * v
            loss["total"] = total

        return loss, pred, time_dict
    


    def combine(self, token_html, token_bbox):
        ##############################################
        # 1. HTML 토큰 확장: 벡터화하여 GPU에서 처리
        ##############################################
        # token_html: (B, S)
        mapping_table = self._html_mapping_table   # shape: (vocab_size, 2)
        is_expanded = self._html_is_expanded           # Boolean 텐서, 동일 device

        # GPU 연산: 각 토큰에 대해 left와 right 토큰을 가져옴.
        left_tokens = mapping_table[token_html, 0]     # (B, S)
        expanded_flags = is_expanded[token_html]         # (B, S), Bool tensor
        right_tokens = mapping_table[token_html, 1]      # (B, S)

        B, S = token_html.shape
        # 각 토큰의 확장 여부를 정수형으로 변환 (0 또는 1)
        expanded_int = expanded_flags.to(torch.int64)
        # 누적합을 통해 각 토큰 이전까지 확장된 개수를 계산 (현재 토큰 미포함)
        extra = torch.cat(
            [torch.zeros((B, 1), device=token_html.device, dtype=torch.int64),
            torch.cumsum(expanded_int, dim=1)[:, :-1]],
            dim=1,
        )
        # 각 토큰의 최종 위치 = 원래 인덱스 + 누적 확장 개수
        positions = torch.arange(S, device=token_html.device).unsqueeze(0).expand(B, S) + extra
        # 최종 시퀀스 길이는 원래 S + 배치별 확장 토큰 개수
        final_lengths = S + torch.sum(expanded_int, dim=1)
        overall_max_length = int(final_lengths.max().item())

        # 전체 배치에 대해 pad_value=0으로 초기화한 텐서 생성
        new_token_html = torch.full(
            (B, overall_max_length), 0, dtype=token_html.dtype, device=token_html.device
        )
        batch_indices = torch.arange(B, device=token_html.device).unsqueeze(1).expand(B, S)
        new_token_html[batch_indices, positions] = left_tokens
        right_positions = positions + 1
        new_token_html[batch_indices[expanded_flags], right_positions[expanded_flags]] = right_tokens[expanded_flags]

        ##############################################
        # 2. BBOX 토큰 매핑 (기존 벡터화 코드)
        ##############################################
        if not hasattr(self, '_bbox_mapping_table'):
            vocab_size_bbox = max(self.old_to_new_mapping_bbox.keys()) + 1
            mapping_tensor_bbox = torch.zeros(vocab_size_bbox, dtype=torch.long, device=token_bbox.device)
            for token_id, mapping in self.old_to_new_mapping_bbox.items():
                mapping_tensor_bbox[token_id] = mapping
            self._bbox_mapping_table = mapping_tensor_bbox
        mapping_tensor_bbox = self._bbox_mapping_table
        new_token_bbox = mapping_tensor_bbox[token_bbox]  # (B, S)

        ##############################################
        # 3. Combined Tokens 생성: HTML 토큰에 BBOX 토큰 삽입
        ##############################################
        # 캐싱된 target tensor 재사용 (매번 새로 생성하지 않음)
        html_target_tensor = self.html_target_tensor
        bbox_target_tensor = self.bbox_target_tensor

        trigger_mask = torch.isin(new_token_html, html_target_tensor)  # (B, L)

        combined_tokens_list = []
        for b in range(B):
            L_b = int(final_lengths[b].item())
            html_seq = new_token_html[b, :L_b]
            trigger_mask_b = trigger_mask[b, :L_b]
            bbox_seq = new_token_bbox[b]
            mask_bbox = torch.isin(bbox_seq, bbox_target_tensor)
            filtered_bbox = bbox_seq[mask_bbox]

            trigger_indices = torch.nonzero(trigger_mask_b, as_tuple=False).squeeze(1)
            num_triggers = trigger_indices.numel()
            total_inserts = min(4 * num_triggers, filtered_bbox.numel())

            combined = []
            bbox_ptr = 0
            for i in range(L_b):
                combined.append(html_seq[i])
                if trigger_mask_b[i]:
                    if bbox_ptr < total_inserts:
                        end = min(bbox_ptr + 4, total_inserts)
                        combined.extend(filtered_bbox[bbox_ptr:end])
                        bbox_ptr = end
            combined_tokens_list.append(torch.stack(combined))
        combined_tokens = torch.nn.utils.rnn.pad_sequence(
            combined_tokens_list, batch_first=True, padding_value=2
        )
        return combined_tokens

    def pad_to_same_length(self, t1: Tensor, t2: Tensor, pad_value: int = 0) -> Tuple[Tensor, Tensor]:
        # 두 텐서의 두 번째 차원(시퀀스 길이)의 최대 길이를 계산
        max_len = max(t1.shape[1], t2.shape[1])
        if t1.shape[1] < max_len:
            pad_size = max_len - t1.shape[1]
            # (left_pad, right_pad) 형식으로 pad; 여기서는 오른쪽에 패딩
            t1 = torch.nn.functional.pad(t1, (0, pad_size), value=pad_value)
        if t2.shape[1] < max_len:
            pad_size = max_len - t2.shape[1]
            t2 = torch.nn.functional.pad(t2, (0, pad_size), value=pad_value)
        return t1, t2





def configure_optimizer_weight_decay(
    model: nn.Module, weight_decay: float
) -> List[Dict]:
    weight_decay_blacklist = (nn.LayerNorm, nn.BatchNorm2d, nn.Embedding)

    if hasattr(model, "no_weight_decay"):
        skip_list = model.no_weight_decay()
    decay = set()
    no_decay = set()
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if pn.endswith("bias"):
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, weight_decay_blacklist):
                no_decay.add(fpn)
            elif pn in skip_list:
                no_decay.add(fpn)

    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay = param_dict.keys() - no_decay

    optim_groups = [
        {
            "params": [param_dict[pn] for pn in sorted(list(decay))],
            "weight_decay": weight_decay,
        },
        {
            "params": [param_dict[pn] for pn in sorted(list(no_decay))],
            "weight_decay": 0.0,
        },
    ]

    return optim_groups


def print_model_architecture(model: nn.Module, indent: int = 0, show_dimensions: bool = True):
    for name, module in model.named_children():
        print("  " * indent + f"{name}: {module.__class__.__name__}")
        
        # Try to get input/output dimensions if available
        if show_dimensions:
            try:
                if hasattr(module, "in_features") and hasattr(module, "out_features"):
                    print("  " * (indent + 1) + f"- dims: {module.in_features} → {module.out_features}")
                elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
                    print("  " * (indent + 1) + f"- dims: {module.in_channels} → {module.out_channels}")
                elif hasattr(module, "num_heads"):
                    print("  " * (indent + 1) + f"- heads: {module.num_heads}")
                elif hasattr(module, "embedding_dim"):
                    print("  " * (indent + 1) + f"- embedding_dim: {module.embedding_dim}")
                elif hasattr(module, "weight") and hasattr(module.weight, "shape"):
                    print("  " * (indent + 1) + f"- weight shape: {list(module.weight.shape)}")
            except Exception as e:
                # If there's an error retrieving dimensions, just continue
                pass
                
        # Print parameter information
        for param_name, param in module.named_parameters(recurse=False):
            print("  " * (indent + 1) + f"- param: {param_name}, shape={list(param.shape)}, requires_grad={param.requires_grad}")
            
        # Recursively print child modules
        print_model_architecture(module, indent + 1, show_dimensions)

def turn_off_beit_grad(model: nn.Module):
    "Freeze BEiT pretrained weights."
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.pos_embed.parameters():
        param.requires_grad = False


def turn_off_beit_grad_weights_freezing(model: nn.Module):
    """
    1) 백본(backbone), 위치 임베딩(pos_embed)은 전부 동결.
    2) 엔코더(encoder)는 마지막 레이어만 학습, 나머지 동결.
    3) 디코더(decoder)는 기존 코드처럼 마지막 레이어만 학습, 나머지 동결.
    """
    # 1) backbone / pos_embed 모두 동결
    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.pos_embed.parameters():
        param.requires_grad = False

    # 2) 엔코더: 마지막 레이어만 학습
    #    (예: model.encoder.encoder.layers가 실제 레이어 스택이라고 가정)
    encoder_layers = model.encoder.encoder.layers
    for i, layer in enumerate(encoder_layers):
        requires_grad = (i == len(encoder_layers) - 1)  # 마지막 레이어만 True
        for param in layer.parameters():
            #param.requires_grad = requires_grad
            param.requires_grad = False

    # 3) 디코더: 기존 코드처럼 마지막 디코더 레이어만 학습
    decoder_layers = model.decoder.decoder.layers
    for i, layer in enumerate(decoder_layers):
        # i == 3 인 레이어만 True, 나머지는 False
        for param in layer.parameters():
            # param.requires_grad = (i == 3)
            param.requires_grad = True



def turn_on_beit_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
        
def create_mix_tokens(vocab_html: dict, vocab_bbox: dict):
    """
    HTML 토큰은 그대로 유지하고, BBOX 토큰만 Offset을 적용하여
    두 vocab을 하나로 합칠 때 필요한 매핑 정보를 생성합니다.

    Args:
        vocab_html (dict[str, int]): 원본 HTML token → index 매핑
        vocab_bbox (dict[str, int]): 원본 BBOX  token → index 매핑

    Returns:
        old_to_new_mapping_html (dict[int, int]):
            원래 HTML token index → (변경 없는) 새 index
        old_to_new_mapping_bbox (dict[int, int]):
            원래 BBOX token index → offset 적용된 새 index
        html_target_indices (list[int]):
            mix 삽입 트리거로 사용할 HTML token index 리스트
        bbox_target_indices (list[int]):
            mix 삽입 시 사용할 BBOX token index 리스트
    """
    # 1) HTML 토큰은 그대로 1:1 매핑
    old_to_new_mapping_html = { idx: idx for idx in vocab_html.values() }

    # 2) BBOX 토큰은 HTML vocab 뒤에 shift
    offset = max(vocab_html.values()) + 1
    old_to_new_mapping_bbox = { idx: idx + offset for idx in vocab_bbox.values() }

    # 3) HTML에서 mix 삽입 트리거로 쓸 토큰들
    html_targets = []
    for t in ("<td>[]</td>", ">[]</td>"):
        if t not in vocab_html:
            raise ValueError(f"HTML vocab에 '{t}' 토큰이 없습니다.")
        html_targets.append(vocab_html[t])

    # 4) BBOX에서 mix 삽입 시 사용할 토큰들
    bbox_targets = [
        orig_idx + offset
        for token, orig_idx in vocab_bbox.items()
        if "bbox" in token.lower()
    ]

    return old_to_new_mapping_html, old_to_new_mapping_bbox, html_targets, bbox_targets




class Batch:
    """Wrap up a batch of training samples with different training targets.
    The input is not torch tensor
    Shape of the image (src): B, S, E
    Shape of the text (tgt): B, N, S, E (M includes 1 table detection, 1 structure, 1 cell, and multiple bbox)
    Reshape text to (B * N, S, E) and inflate the image to match the shape of the text

    Args:
    ----
        device: gpu id
    """

    def __init__(
        self,
        device: torch.device,
        target: str,
        vocab: Vocab,
        obj: List,
    ) -> None:
        self.device = device
        self.image = obj[0].to(device)
        self.name = obj[1]["filename"]
        self.target = target
        self.vocab = vocab
        self.image_size = self.image.shape[-1]

        if "table" in target:
            raise NotImplementedError

        if "html" in target:

            self.valid_html_token = [vocab.token_to_id(i) for i in VALID_HTML_TOKEN]
            (
                self.html_src,
                self.html_tgt,
                self.html_casual_mask,
                self.html_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["html"])

        if "cell" in target:
            self.invalid_cell_token = [vocab.token_to_id(i) for i in INVALID_CELL_TOKEN]
            (
                self.cell_src,
                self.cell_tgt,
                self.cell_casual_mask,
                self.cell_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["cell"])

        if "bbox" in target:
            (
                self.bbox_src,
                self.bbox_tgt,
                self.bbox_casual_mask,
                self.bbox_padding_mask,
            ) = self._prepare_transformer_input(obj[1]["bbox"])

        
    def _prepare_transformer_input(
        self, seq: List[tk.Encoding]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tmp = [i.ids for i in seq]
        tmp = torch.tensor(tmp, dtype=torch.int32)
        src = tmp[:, :-1].to(self.device)
        tgt = tmp[:, 1:].type(torch.LongTensor).to(self.device)
        casual_mask = subsequent_mask(src.shape[-1]).to(self.device)
        tmp = [i.attention_mask[:-1] for i in seq]  # padding mask
        tmp = torch.tensor(tmp, dtype=torch.bool)
        padding_mask = (~tmp).to(self.device)

        return src, tgt, casual_mask, padding_mask

    def _inference_one_task(
        self, model, memory, src, casual_mask, padding_mask, use_ddp
    ):
        if use_ddp:
            out = model.module.decode(memory, src, casual_mask, padding_mask)
            out = model.module.generator(out)
        else:
            out = model.decode(memory, src, casual_mask, padding_mask)
            out = model.generator(out)

        return out

    def inference(
        self,
        model: nn.Module,
        criterion: nn.Module,
        criterion_bbox: nn.Module = None,
        loss_weights: dict = None,
        use_ddp: bool = True,
    ) -> Tuple[Dict, Dict]:
        pred = dict()
        loss = dict(table=0, html=0, cell=0, bbox=0)

        if use_ddp:
            memory = model.module.encode(self.image)
        else:
            memory = model.encode(self.image)

        # inference + suppress invalid logits + compute loss
        if "html" in self.target:
            out_html = self._inference_one_task(
                model,
                memory,
                self.html_src,
                self.html_casual_mask,
                self.html_padding_mask,
                use_ddp,
            )

            pred["html"] = pred_token_within_range(
                out_html, white_list=self.valid_html_token
            ).permute(0, 2, 1)
            loss["html"] = criterion(pred["html"], self.html_tgt)
            # print(f'pred html : {pred["html"]}')

        if "cell" in self.target:
            out_cell = self._inference_one_task(
                model,
                memory,
                self.cell_src,
                self.cell_casual_mask,
                self.cell_padding_mask,
                use_ddp,
            )

            pred["cell"] = pred_token_within_range(
                out_cell, black_list=self.invalid_cell_token
            ).permute(0, 2, 1)
            loss["cell"] = criterion(pred["cell"], self.cell_tgt)

        if "bbox" in self.target:
            assert criterion_bbox is not None

            out_bbox = self._inference_one_task(
                model,
                memory,
                self.bbox_src,
                self.bbox_casual_mask,
                self.bbox_padding_mask,
                use_ddp,
            )
            pred["bbox"] = out_bbox.permute(0, 2, 1)
            loss["bbox"] = criterion_bbox(pred["bbox"], self.bbox_tgt)

        total = 0.0
        for k, v in loss_weights.items():
            total += loss[k] * v
        loss["total"] = total

        return loss, pred
