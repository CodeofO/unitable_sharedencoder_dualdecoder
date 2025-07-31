from typing import List, Tuple, Dict
import torch
from torch import Tensor, nn
from torchtext.vocab import Vocab
import tokenizers as tk
from time import time
# import editdistance

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

class Batch:
    """Placeholder Batch class for compatibility with legacy imports."""
    pass




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
    ) -> None:

        self.device = device
        self.image = obj[0].to(device)
        self.name = obj[1]["filename"]
        self.target = target
        self.vocab_html = vocab_html  # ✅
        self.vocab_bbox = vocab_bbox  # ✅
        self.image_size = self.image.shape[-1]

        self.use_mix_loss = use_mix_loss
        
        if "table" in target:
            raise NotImplementedError

        # valid token 설정
        self.pad_idx = self.vocab_html.token_to_id("<pad>")  
        self.eos_idx = self.vocab_html.token_to_id("<eos>")
        
        if otsl_mode:
            # print(f'🔥 vocab_html : {vocab_html.get_vocab()}')
            # print(f'🔥 vocab_bbox : {vocab_bbox.get_vocab()}')
            self.valid_html_token = [vocab_html.token_to_id(i) for i in VALID_OTSL_TOKEN]
            self.f_c_idx = self.vocab_html.token_to_id("F_C")
            self.f_c_idx_merge = None

        else:
            self.valid_html_token = [vocab_html.token_to_id(i) for i in VALID_HTML_TOKEN]
            self.f_c_idx = self.vocab_html.token_to_id("<td>[]</td>")
            self.f_c_idx_merge = self.vocab_html.token_to_id(">[]</td>")

        self.constant_two = torch.tensor(2, device=self.device)

        # --- bbox valid ids 계산 ---
        # vocab_bbox.get_vocab() 은 {token_str: idx, ...} 형태의 dict 반환
        bbox_vocab_dict = vocab_bbox.get_vocab()
        # "bbox" 가 이름에 들어간 토큰들만 골라서 ID 리스트로
        bbox_ids = [idx for tok, idx in bbox_vocab_dict.items() if "bbox" in tok]
        # 필요하다면 정렬
        bbox_ids = sorted(bbox_ids)
        # GPU 텐서로 저장
        self.bbox_valid_ids = torch.tensor(bbox_ids, device=self.device, dtype=torch.long)

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

    def inference_shared(
        self,
        model: nn.Module,  # SharedEncoder_DualDecoder
        criterion_html: nn.Module,
        criterion_bbox: nn.Module,
        criterion_mix: nn.Module,
        loss_weights: dict = None,
        use_ddp: bool = True,
    ) -> Tuple[Dict, Dict]:
        """
        SharedEncoder_DualDecoder용 inference 메서드
        한 번의 forward pass로 HTML과 BBOX 결과를 동시에 얻음
        """
        pred = dict()
        loss = dict(table=0, html=0, cell=0, bbox=0)
        
        st = time()
        if use_ddp:
            # SharedEncoder_DualDecoder forward pass
            outputs = model.module(
                self.image,
                self.html_src,
                self.bbox_src,
                self.html_casual_mask,
                self.bbox_casual_mask,
                self.html_padding_mask,
                self.bbox_padding_mask,
            )
        else:
            outputs = model(
                self.image,
                self.html_src,
                self.bbox_src,
                self.html_casual_mask,
                self.bbox_casual_mask,
                self.html_padding_mask,
                self.bbox_padding_mask,
            )
        
        
        # HTML 결과 처리
        out_html = outputs["html"]
        pred_html_logits = pred_token_within_range(
            out_html, white_list=self.valid_html_token
        ).permute(0, 2, 1)
        pred["html"] = pred_html_logits
        # print(f'🔥 pred_html_logits : {pred_html_logits.shape}')
        # print(f'🔥 out_html : {out_html}')
        # print(f'🔥 self.html_tgt : {self.html_tgt}')
        # print(f'🔥 self.valid_html_token : {self.valid_html_token}')
        loss["html"] = criterion_html(pred_html_logits, self.html_tgt)
        
        # BBOX 결과 처리
        out_bbox = outputs["bbox"]
        pred_bbox_logits = out_bbox.permute(0, 2, 1)
        pred["bbox"] = pred_bbox_logits
        loss["bbox"] = criterion_bbox(pred_bbox_logits, self.bbox_tgt)
        time_inf = time() - st
        
        # Mix loss 계산
        st = time()
        self.check_b = 3
        loss['mix_combine'] = self.cal_mix_loss(out_html, pred_bbox_logits, criterion_mix)
        time_inf_combine = time() - st
        
        time_dict = {
            "time_inf": time_inf,
            "time_inf_combine": time_inf_combine
        }
        
        # Total loss 계산
        total = torch.tensor(0.0).to(self.device)
        for k, v in loss_weights.items():
            total += loss[k] * v
        loss["total"] = total
        
        return loss, pred, time_dict



    def cal_mix_loss(self, out_html: torch.Tensor, pred_bbox_logits: torch.Tensor, criterion_mix):
        if not self.use_mix_loss:
            return torch.tensor(0, device=self.device, dtype=torch.int64)
        
        else:
            B, T = out_html.size(0), out_html.size(1)
            offset  = len(self.vocab_html.get_vocab())
            pad_idx = self.pad_idx

            # 1) PRED HTML argmax & EOS→2
            html_pred = out_html.argmax(dim=-1)
            eos_mask  = html_pred.eq(self.eos_idx)
            cum_eos   = eos_mask.cumsum(dim=1)
            mask_from = cum_eos >= 1
            mask_a    = torch.zeros_like(mask_from)
            mask_a[:,1:] = mask_from[:,:-1]
            html_pred = html_pred.masked_fill(mask_a, 2)

            # 2) GT HTML에도 동일한 EOS→2
            html_gt   = self.html_tgt.clone()
            eos_gt    = html_gt.eq(self.eos_idx)
            cum_gt    = eos_gt.cumsum(dim=1)
            mask_gt_f = cum_gt >= 1
            mask_gt_a = torch.zeros_like(mask_gt_f)
            mask_gt_a[:,1:] = mask_gt_f[:,:-1]
            html_gt   = html_gt.masked_fill(mask_gt_a, 2)

            # 3) BBOX argmax & valid mask
            pred_bbox_ids = pred_bbox_logits.argmax(dim=1)
            mask_valid    = torch.isin(pred_bbox_ids, self.bbox_valid_ids)

            # 4) 샘플별 bbox 그룹화
            groups_pred = []
            groups_gt   = []
            for b in range(B):
                v  = pred_bbox_ids[b][mask_valid[b]]
                n  = v.numel() // 4
                groups_pred.append(v[:n*4].view(n,4) if n>0 else torch.empty((0,4),device=v.device))

                mg = torch.isin(self.bbox_tgt[b], self.bbox_valid_ids)
                vg = self.bbox_tgt[b][mg]
                ng = vg.numel() // 4
                groups_gt.append(vg[:ng*4].view(ng,4) if ng>0 else torch.empty((0,4),device=vg.device))

            # 5) GT 기준 new_len 결정
            max_extra = max(g.size(0)*4 for g in groups_gt)
            new_len   = T + max_extra

            # 6) pred_mix, gt_mix 초기화
            pred_mix = torch.full((B,new_len), pad_idx, device=html_pred.device, dtype=html_pred.dtype)
            gt_mix   = torch.full((B,new_len), pad_idx, device=html_gt.device,   dtype=html_gt.dtype)

            # 7) HTML 삽입 위치 계산 (pred / gt 각각)
            if self.f_c_idx_merge is not None:
                is_fc_pred = (html_pred == self.f_c_idx) | (html_pred == self.f_c_idx_merge)
            else:
                is_fc_pred = (html_pred == self.f_c_idx)

            pref_pred  = is_fc_pred.int().cumsum(dim=1)
            shift_pred = torch.cat([
                torch.zeros(B,1,device=html_pred.device,dtype=pref_pred.dtype),
                pref_pred[:,:-1]
            ], dim=1) * 4
            idxs       = torch.arange(T, device=html_pred.device).unsqueeze(0).expand(B,-1)
            new_pos_html = (idxs + shift_pred).clamp(0, new_len-1)
            
            if self.f_c_idx_merge is not None:
                is_fc_gt   = (html_gt == self.f_c_idx) | (html_gt == self.f_c_idx_merge)
            else:
                is_fc_gt   = (html_gt == self.f_c_idx)

            pref_gt    = is_fc_gt.cumsum(dim=1)
            shift_gt   = torch.cat([
                torch.zeros(B,1,device=html_gt.device,dtype=pref_gt.dtype),
                pref_gt[:,:-1]
            ], dim=1) * 4
            new_pos_gt = (idxs + shift_gt).clamp(0, new_len-1)

            # 8) HTML scatter
            pred_mix.scatter_(1, new_pos_html.long(), html_pred)
            gt_mix  .scatter_(1, new_pos_gt.long(),    html_gt)

            # 9) BBOX 삽입
            offs_back = torch.arange(1,5,device=html_pred.device).view(1,4)
            offs_front = torch.arange(1,5,device=html_pred.device).view(1,4)
            back = True
            if back:
                offs = offs_back
            else:
                offs = offs_front
            for b in range(B):
                # PRED
                pos_indices_p = is_fc_pred[b].nonzero(as_tuple=True)[0][:groups_pred[b].size(0)]
                pos_p = new_pos_html[b, pos_indices_p].unsqueeze(-1)
                locs_p = (pos_p + offs).view(-1).clamp(0, new_len-1)
                vals_p = (groups_pred[b] + offset).view(-1)
                if vals_p.numel():
                    pred_mix[b].scatter_(0, locs_p.long(), vals_p.long())

                # GT
                pos_indices_g = is_fc_gt[b].nonzero(as_tuple=True)[0][:groups_gt[b].size(0)]
                pos_g = new_pos_gt[b, pos_indices_g].unsqueeze(-1)
                locs_g = (pos_g + offs).view(-1)
                vals_g = (groups_gt[b] + offset).view(-1)
                if vals_g.numel():
                    gt_mix[b].scatter_(0, locs_g.long(), vals_g.long())

            # 11) Masked L1 Loss
            loss_mat = criterion_mix(pred_mix.float(), gt_mix.float())
            mask     = (gt_mix != pad_idx).float()
            return (loss_mat * mask).sum() / mask.sum() * 0.05


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
            param.requires_grad = False

    # 3) 디코더: 기존 코드처럼 마지막 디코더 레이어만 학습
    decoder_layers = model.decoder.decoder.layers
    for i, layer in enumerate(decoder_layers):
        # i == 3 인 레이어만 True, 나머지는 False
        for param in layer.parameters():
            param.requires_grad = (i == 3)
            # param.requires_grad = True



def turn_on_beit_grad(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True




