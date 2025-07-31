from typing import List, Tuple, Dict
import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab
import tokenizers as tk
import torch.nn.functional as F


from utils import pred_token_within_range, subsequent_mask
from vocab import (
    HTML_TOKENS,
    TASK_TOKENS,
    RESERVED_TOKENS,
    BBOX_TOKENS,
)


VALID_HTML_TOKEN = ["<eos>"] + HTML_TOKENS
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

    Args:
    ----
        device: gpu id
    """

    def __init__(
        self,
        device: torch.device,
        target: str,
        vocab_html: Vocab, # ✅
        vocab_bbox: Vocab, # ✅
        obj: List,# ✅
        ) -> None:
        
        self.device = device
        self.image = obj[0].to(device)
        self.name = obj[1]["filename"]
        self.target = target
        self.vocab_html = vocab_html # ✅
        self.vocab_bbox = vocab_bbox # ✅
        self.image_size = self.image.shape[-1]

        if "table" in target:
            raise NotImplementedError

        # ✅
        self.valid_html_token = [vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN]

        (
            self.html_src,
            self.html_tgt,
            self.html_casual_mask,
            self.html_padding_mask,
        ) = self._prepare_transformer_input(obj[1]["html"]) # ✅

        # ✅
        (
            self.bbox_src,
            self.bbox_tgt,
            self.bbox_casual_mask,
            self.bbox_padding_mask,
        ) = self._prepare_transformer_input(obj[1]["bbox"]) # ✅

        vocab_html = self.vocab_html.get_vocab()
        vocab_bbox = self.vocab_bbox.get_vocab()
        self.bbox_target_indices = [idx for token, idx in vocab_bbox.items() if "bbox" in token]
        self.html_target_indices = [
        vocab_html.get(">[]</td>", None),
        vocab_html.get("<td>[]</td>", None)
        ]

        self.old_to_new_mapping_html, self.old_to_new_mapping_bbox, self.html_target_indices, self.bbox_target_indices\
            = create_mix_tokens(vocab_html, vocab_bbox)

        self.get_mix_vocab_size()

        self.embedding_dim_html = None
        self.embedding_dim_bbox = None
        

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
        model_html: nn.Module, # ✅
        model_bbox: nn.Module, # ✅
        criterion_html: nn.Module,
        criterion_bbox: nn.Module,
        criterion_mix: nn.Module,
        loss_weights: dict = None,
        use_ddp: bool = True,
    ) -> Tuple[Dict, Dict]:
        pred = dict()
        loss = dict(table=0, html=0, cell=0, bbox=0)

        self.set_embedding_dim(model_html, model_bbox)
        
        if use_ddp:
            memory_html = model_html.module.encode(self.image) # ✅
            memory_bbox = model_bbox.module.encode(self.image) # ✅
        else:
            memory_html = model_html.encode(self.image)
            memory_bbox = model_bbox.encode(self.image)

        # inference + suppress invalid logits + compute loss
        out_html = self._inference_one_task(
                model_html,
                memory_html,
                self.html_src,
                self.html_casual_mask,
                self.html_padding_mask,
                use_ddp,
            )

        pred_html_logits = pred_token_within_range(
            out_html, white_list=self.valid_html_token).permute(0, 2, 1)
        pred["html"] = pred_html_logits
        loss["html"] = criterion_html(pred_html_logits, self.html_tgt)


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


        # Softmax 확률 분포 그대로 사용
        probs_html = F.softmax(out_html, dim=-1)    # (B, S_html, V_html)
        probs_bbox = F.softmax(out_bbox, dim=-1)      # (B, S_bbox, V_bbox)

        # <eos> 이후를 pad 토큰 확률로 마스킹하는 처리는 필요 시 추가 (여기서는 생략)

        # Differentiable combine: soft predictions 결합
        combined_soft_pred = self.combine_soft(probs_html, probs_bbox)  
        # combined_soft_pred의 shape: (B, L, X), 여기서 X == V_html (예: 1397)

        # --- 추가: combined_soft_pred의 마지막 차원을 모델 임베딩 차원으로 맞춤 ---
        # (만약 self.vocab_html.get_vocab_size() != self.embedding_dim_html)
        if combined_soft_pred.size(-1) != self.embedding_dim_html:
            # 초기화되지 않았다면 projection layer 생성
            if not hasattr(self, "proj_to_mix"):
                # self.vocab_html.get_vocab_size()가 보통 soft_html의 마지막 차원 (예: 1397)입니다.
                self.proj_to_mix = nn.Linear(combined_soft_pred.size(-1), self.embedding_dim_html).to(self.device)
            combined_soft_pred = self.proj_to_mix(combined_soft_pred)
        # -----------------------------------------------------------------------

        # GT 결합: 기존 combine 함수로 discrete token id 시퀀스 생성
        combined_tokens_tgt = self.combine(self.html_tgt, self.bbox_tgt)  # (B, L_t)
        # GT 결합 토큰들을 모델의 임베딩 공간으로 변환 (여기서는 html 임베딩 사용)
        combined_tokens_tgt_embed = self.embedding_html(combined_tokens_tgt)  # (B, L_t, d_model)

        # 두 결합 표현의 시퀀스 길이가 다르면 pad_to_same_length로 맞춤
        combined_tokens_tgt_embed, combined_soft_pred = self.pad_to_same_length(
            combined_tokens_tgt_embed, combined_soft_pred, pad_value=0.0
        )

        # mix loss 계산: mix_classifier로 continuous 표현을 merged vocabulary logits로 변환
        if not hasattr(self, "mix_classifier"):
            self.mix_vocab_size = self.get_mix_vocab_size()
            self.mix_classifier = nn.Linear(self.embedding_dim_html, self.mix_vocab_size).to(self.device)
        mix_logits = self.mix_classifier(combined_soft_pred)  # (B, L, mix_vocab_size)
        mix_logits = mix_logits.permute(0, 2, 1)  # (B, mix_vocab_size, L)

        # criterion_mix는 nn.CrossEntropyLoss, target: (B, L) (정수 token id)
        new_loss_2_combine = criterion_mix(mix_logits, combined_tokens_tgt.long())
        loss['mix'] = new_loss_2_combine

        total = torch.tensor(0.0).to(probs_bbox.device)
        for k, v in loss_weights.items():
            total += loss[k] * v
        loss["total"] = total

        return loss, pred

    def get_mix_vocab_size(self) -> int:
        """
        Computes the merged vocabulary size based on the old_to_new_mapping for html and bbox.
        """
        max_val = 0
        for mapping in self.old_to_new_mapping_html.values():
            if isinstance(mapping, dict):
                max_val = max(max_val, mapping["left"], mapping["right"])
            else:
                max_val = max(max_val, mapping)
        for mapping in self.old_to_new_mapping_bbox.values():
            max_val = max(max_val, mapping)
        return max_val + 1


    def set_embedding_dim(self, model_html: nn.Module, model_bbox: nn.Module):
        """
        Sets the embedding dimensions based on the provided models.
        Since EncoderDecoder does not have a config attribute,
        we extract the hidden dimension (d_model) from the token_embed.
        그리고 생성된 임베딩 레이어를 self.device로 이동시킵니다.
        """
        # DDP 래핑 여부 확인
        if hasattr(model_html, "module"):
            model_html = model_html.module
        if hasattr(model_bbox, "module"):
            model_bbox = model_bbox.module

        # model_html.token_embed.embedding.weight.shape: (vocab_size, d_model)
        self.embedding_dim_html = model_html.token_embed.embedding.weight.shape[1]
        self.embedding_dim_bbox = model_bbox.token_embed.embedding.weight.shape[1]

        # 임베딩 레이어 생성 후, self.device로 이동
        self.embedding_html = nn.Embedding(self.vocab_html.get_vocab_size(), self.embedding_dim_html).to(self.device)
        self.embedding_bbox = nn.Embedding(self.vocab_bbox.get_vocab_size(), self.embedding_dim_bbox).to(self.device)



    def combine_soft(self, soft_html: torch.Tensor, soft_bbox: torch.Tensor) -> torch.Tensor:
        """
        soft_html: (B, S_html, V_html) — html decoder의 softmax 결과
        soft_bbox: (B, S_bbox, V_bbox) — bbox decoder의 softmax 결과

        각 확률 분포를 임베딩 공간의 기대값(soft embedding)으로 변환한 후,
        html target 토큰(예: self.html_target_indices)에 해당하는 위치에 대해 bbox 임베딩을 가중합합니다.
        
        만약 두 텐서의 시퀀스 길이(S_html, S_bbox)가 다르면, 짧은 쪽을 오른쪽에 0으로 패딩하여 길이를 맞춥니다.
        """
        B = soft_html.shape[0]
        
        # 시퀀스 길이 맞추기: (B, S, V)
        if soft_html.shape[1] < soft_bbox.shape[1]:
            pad_size = soft_bbox.shape[1] - soft_html.shape[1]
            # pad (last dimension: no pad; second-last dimension: pad at end)
            soft_html = torch.nn.functional.pad(soft_html, (0, 0, 0, pad_size), value=0.0)
        elif soft_bbox.shape[1] < soft_html.shape[1]:
            pad_size = soft_html.shape[1] - soft_bbox.shape[1]
            soft_bbox = torch.nn.functional.pad(soft_bbox, (0, 0, 0, pad_size), value=0.0)
        
        # 기대 임베딩 계산
        # self.embedding_html.weight: (V_html, d_html)
        # self.embedding_bbox.weight: (V_bbox, d_bbox)
        soft_embed_html = torch.matmul(soft_html, self.embedding_html.weight)   # (B, S, d_html)
        soft_embed_bbox = torch.matmul(soft_bbox, self.embedding_bbox.weight)   # (B, S, d_bbox)
        
        # 만약 두 임베딩의 차원이 다르면 bbox 임베딩을 html 임베딩 차원으로 선형 변환
        if soft_embed_html.size(-1) != soft_embed_bbox.size(-1):
            if not hasattr(self, 'bbox_to_html_proj'):
                self.bbox_to_html_proj = nn.Linear(soft_embed_bbox.size(-1), soft_embed_html.size(-1)).to(soft_embed_bbox.device)
            soft_embed_bbox = self.bbox_to_html_proj(soft_embed_bbox)  # (B, S, d_html)
        
        # html target 토큰에 해당하는 확률 mass를 구함.
        # target_mask: (V_html,) - html target 토큰에 해당하면 1, 아니면 0.
        target_mask = torch.zeros(soft_html.size(-1), device=soft_html.device)
        for idx in self.html_target_indices:
            if idx is not None:
                target_mask[idx] = 1.0
        # 각 위치에서 html soft 예측에서 target 토큰 확률의 합 → (B, S, 1)
        target_indicator = torch.matmul(soft_html, target_mask.unsqueeze(-1))
        
        # 단순 결합: html 임베딩에 target_indicator가 클수록 bbox 임베딩의 영향을 더 부여
        combined = soft_embed_html + target_indicator * soft_embed_bbox
        # 결과: (B, S, d_html)
        return combined



    def combine(self, token_html, token_bbox):
        ##############################
        # HTML 토큰 처리 (최적화 적용)
        ##############################
        batch_size, seq_length = self.html_tgt.shape

        # 사전: HTML mapping 테이블과 확장 여부 플래그를 GPU 텐서로 미리 생성
        if not hasattr(self, '_html_mapping_table'):
            vocab_size_html = max(self.old_to_new_mapping_html.keys()) + 1
            mapping_table = torch.zeros((vocab_size_html, 2), dtype=torch.long, device=token_html.device)
            is_expanded = torch.zeros(vocab_size_html, dtype=torch.bool, device=token_html.device)
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

        mapping_table = self._html_mapping_table
        is_expanded = self._html_is_expanded

        # token_html: (B, S) → left 토큰, 확장 플래그, right 토큰 (모두 (B, S))
        left_tokens = mapping_table[token_html, 0]     # 기본 변환 값
        expanded_flags = is_expanded[token_html]         # True면 2개 토큰으로 확장
        right_tokens = mapping_table[token_html, 1]      # 확장 시 두 번째 토큰

        # 각 배치별로 확장된 시퀀스 생성 (배치 크기가 작다고 가정)
        new_token_html_list = []
        for b in range(batch_size):
            tokens_b = []
            # GPU 텐서는 .item()을 호출하면 CPU로 옮겨지므로, 가능하면 인덱싱 후 한 번에 처리
            left_b = left_tokens[b]
            right_b = right_tokens[b]
            flags_b = expanded_flags[b]
            for i in range(seq_length):
                if flags_b[i].item():  # 확장이 필요한 경우
                    tokens_b.append(left_b[i])
                    tokens_b.append(right_b[i])
                else:
                    tokens_b.append(left_b[i])
            # tokens_b는 리스트 형태의 1D 텐서; pad_sequence 후 결합 예정
            new_token_html_list.append(torch.stack(tokens_b))
        # 모든 배치의 길이가 다를 수 있으므로 pad 처리 (pad value: 0)
        new_token_html = torch.nn.utils.rnn.pad_sequence(new_token_html_list, batch_first=True, padding_value=0)

        ##############################
        # BBOX 토큰 처리 (완전 벡터화)
        ##############################
        # 사전: BBOX mapping 테이블을 GPU 텐서로 생성
        if not hasattr(self, '_bbox_mapping_table'):
            vocab_size_bbox = max(self.old_to_new_mapping_bbox.keys()) + 1
            mapping_tensor_bbox = torch.zeros(vocab_size_bbox, dtype=torch.long, device=token_bbox.device)
            for token_id, mapping in self.old_to_new_mapping_bbox.items():
                mapping_tensor_bbox[token_id] = mapping
            self._bbox_mapping_table = mapping_tensor_bbox
        mapping_tensor_bbox = self._bbox_mapping_table
        # token_bbox: (B, S) → vectorized mapping
        new_token_bbox = mapping_tensor_bbox[token_bbox]  # shape: (B, seq_length)

        ##############################
        # Combined Tokens 생성
        ##############################
        # target set을 GPU 텐서로 변환 (멤버십 테스트에 활용)
        html_target_tensor = torch.tensor(list(self.html_target_indices), device=new_token_html.device)
        bbox_target_tensor = torch.tensor(list(self.bbox_target_indices), device=new_token_bbox.device)

        combined_tokens_list = []
        for b in range(batch_size):
            # HTML 결합 대상 토큰: GPU 연산으로 membership test (하지만, 최종 결합은 파이썬 루프로 진행)
            html_tokens_b = new_token_html[b]  # shape: (L,)
            bbox_tokens_b = new_token_bbox[b]  # shape: (S,)
            # torch.isin를 사용해 bbox 필터링 (vectorized)
            mask_bbox = torch.isin(bbox_tokens_b, bbox_target_tensor)
            filtered_bbox = bbox_tokens_b[mask_bbox]
            bbox_ptr = 0
            combined_batch = []
            for token in html_tokens_b:
                combined_batch.append(token)
                # html_target_set은 파이썬 set이므로, membership 체크는 .item() 후 진행
                if token.item() in self.html_target_indices:
                    # 최대 4개의 bbox 토큰 삽입
                    num_to_insert = 4
                    if bbox_ptr < filtered_bbox.numel():
                        end = min(bbox_ptr + num_to_insert, filtered_bbox.numel())
                        combined_batch.extend(filtered_bbox[bbox_ptr:end])
                        bbox_ptr = end
            # 리스트를 텐서로 변환
            combined_tokens_list.append(torch.stack(combined_batch))
        # 배치별 길이가 다를 수 있으므로 pad 처리 (padding_value: 2)
        combined_tokens = torch.nn.utils.rnn.pad_sequence(combined_tokens_list, batch_first=True, padding_value=2)

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



def turn_off_beit_grad(model: nn.Module):
    """
    Freezes all BEiT-related weights except the last decoder layer.
    """
    # print("\n🔍 Before freezing:")

    # Freeze encoder, backbone, and positional embedding
    for param in model.encoder.parameters():
        param.requires_grad = False

    for param in model.backbone.parameters():
        param.requires_grad = False

    for param in model.pos_embed.parameters():
        param.requires_grad = False

    # Freeze decoder layers 0, 1, 2 and keep the last layer (3) trainable
    decoder_layers = model.decoder.decoder.layers
    for i, layer in enumerate(decoder_layers):
        for param in layer.parameters():
            param.requires_grad = (i == 3)  # Only last decoder layer trainable



def turn_on_beit_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True


def create_mix_tokens(vocab_html, vocab_bbox):
    # merged_vocab을 구성하기 위한 새로운 vocab_html 딕셔너리와 매핑 딕셔너리 생성
    new_vocab_html = {}
    old_to_new_mapping_html = {}

    # 오른쪽 토큰("]</td>")의 키를 일관되게 사용하기 위한 변수 (조건에 맞게 정확히 표기)
    right_token = "]</td>"

    # vocab_html의 항목들을 인덱스 순으로 순회합니다.
    # (원래 vocab_html은 토큰→인덱스 딕셔너리입니다.)
    for token, idx in sorted(vocab_html.items(), key=lambda x: x[1]):
        if token == "<td>[]</td>":
            # 조건 2: 원래의 인덱스(idx)를 그대로 사용하여 "<td>["에 할당
            new_vocab_html["<td>["] = idx
            # 조건 4: 새 토큰 "]</td>"를 vocab_html의 마지막 인덱스 + 1로 추가 (아직 추가되지 않았다면)
            if right_token not in new_vocab_html:
                new_index = max(vocab_html.values()) + 1
                new_vocab_html[right_token] = new_index
            # 매핑: 원래 "<td>[]</td>"의 인덱스 idx가 분리되어, left는 idx, right는 새로 추가된 인덱스로 기록
            old_to_new_mapping_html[idx] = {"left": idx, "right": new_vocab_html[right_token]}
        elif token == ">[]</td>":
            # 조건 3: 원래의 인덱스(idx)를 그대로 사용하여 ">["에 할당
            new_vocab_html[">["] = idx
            # 조건 4: 새 토큰 "]</td>"가 없으면 추가 (이미 추가되어 있다면 그대로 사용)
            if right_token not in new_vocab_html:
                new_index = max(vocab_html.values()) + 1
                new_vocab_html[right_token] = new_index
            old_to_new_mapping_html[idx] = {"left": idx, "right": new_vocab_html[right_token]}
        else:
            # 그 외 토큰은 그대로 복사
            new_vocab_html[token] = idx
            old_to_new_mapping_html[idx] = idx  # 변환이 없으면 원래 인덱스로 기록

    # ---------------

    # vocab_bbox 처리 (조건 5)
    # vocab_bbox의 모든 토큰 인덱스에 new_vocab_html의 마지막 인덱스 + 1 만큼의 오프셋을 더합니다.
    offset = max(new_vocab_html.values()) + 1
    new_vocab_bbox = { token: (idx + offset) for token, idx in vocab_bbox.items() }

    # 두 vocab을 병합합니다.
    merged_vocab = {**new_vocab_html, **new_vocab_bbox}

    # 추가 개발사항: 기존 bbox vocab의 인덱스와 새로운 인덱스 매핑 딕셔너리 생성
    old_to_new_mapping_bbox = { idx: (idx + offset) for token, idx in vocab_bbox.items() }

    # ---------------

    # 타겟 인덱스 설정
    # HTML 타겟: 새롭게 생성된 ">["와 "<td>["의 인덱스
    html_target_indices = [
        new_vocab_html.get(">["),      # ">["가 원래 ">[]</td>"의 인덱스를 그대로 가짐
        new_vocab_html.get("<td>[")      # "<td>["가 원래 "<td>[]</td>"의 인덱스를 그대로 가짐
    ]

    # BBOX 타겟: merged된 bbox vocab(new_vocab_bbox)에서 "bbox"라는 문자열이 포함된 토큰들의 인덱스 모두
    bbox_target_indices = [
        new_vocab_bbox[token] for token in new_vocab_bbox if "bbox" in token.lower()
    ]

    # ---------------

    # 객체 변수에 저장
    merged_vocab = merged_vocab
    html_target_indices = html_target_indices
    bbox_target_indices = bbox_target_indices
    old_to_new_mapping_html = old_to_new_mapping_html
    old_to_new_mapping_bbox = old_to_new_mapping_bbox

    return old_to_new_mapping_html, old_to_new_mapping_bbox, html_target_indices, bbox_target_indices





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
