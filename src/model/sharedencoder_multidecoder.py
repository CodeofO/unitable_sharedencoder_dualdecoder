import torch
from torch import Tensor, nn
import torch.nn.functional as F
from functools import partial
from typing import Dict

from model.components import (
    Encoder,
    Decoder,
    PositionEmbedding,
    TokenEmbedding,
)


# ---------------------------------------------------------------------
# 1) SharedEncoder_DualDecoder
# ---------------------------------------------------------------------
class SharedEncoder_DualDecoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder_html: nn.Module,
        decoder_bbox: nn.Module,
        vocab_size_html: int,
        vocab_size_bbox: int,
        d_model: int,
        padding_idx: int,
        max_seq_len:int,
        max_seq_len_html: int,
        max_seq_len_bbox: int,
        dropout: float,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ):
        super().__init__()

        # --- 1. Shared Modules ---
        self.backbone = backbone
        self.encoder = encoder
        self.pos_embed   = PositionEmbedding(max_seq_len, d_model, dropout)

        # --- 2. HTML Modules ---
        self.pos_embed_html  = PositionEmbedding(max_seq_len_html, d_model, dropout)
        self.decoder_html    = decoder_html
        self.token_embed_html = TokenEmbedding(vocab_size_html, d_model, padding_idx)
        self.generator_html   = nn.Linear(d_model, vocab_size_html)

        # --- 3. BBOX Modules ---
        self.pos_embed_bbox  = PositionEmbedding(max_seq_len_bbox, d_model, dropout)
        self.decoder_bbox    = decoder_bbox
        self.token_embed_bbox = TokenEmbedding(vocab_size_bbox, d_model, padding_idx)
        self.generator_bbox   = nn.Linear(d_model, vocab_size_bbox)

        # --- Norm & Init ---
        self.norm = norm_layer(d_model)
        self.trunc_normal = partial(nn.init.trunc_normal_, std=init_std,
                                    a=-init_std, b=init_std)
        self.apply(self._init_weights)

    # weight init 동일 …
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (PositionEmbedding, TokenEmbedding)):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed_img", "pos_embed_html", "pos_embed_bbox",
            "token_embed_html", "token_embed_bbox",
        }
        
    # --- Encoder / Decoder ---
    def encode(self, src: Tensor) -> Tensor:
        src_feat = self.backbone(src)
        src_feat = self.pos_embed(src_feat)
        memory   = self.encoder(src_feat)
        return self.norm(memory)

    def decode_html(self, memory, tgt, tgt_mask, tgt_pad_mask):
        tgt_feat = self.pos_embed_html(self.token_embed_html(tgt))
        return self.decoder_html(tgt_feat, memory, tgt_mask, tgt_pad_mask)

    def decode_bbox(self, memory, tgt, tgt_mask, tgt_pad_mask):
        tgt_feat = self.pos_embed_bbox(self.token_embed_bbox(tgt))
        return self.decoder_bbox(tgt_feat, memory, tgt_mask, tgt_pad_mask)

    def forward(
        self, src, tgt_html, tgt_bbox,
        tgt_mask_html, tgt_mask_bbox,
        tgt_pad_html, tgt_pad_bbox,
    ) -> Dict[str, Tensor]:
        memory   = self.encode(src)
        html_out = self.generator_html(
            self.decode_html(memory, tgt_html, tgt_mask_html, tgt_pad_html))
        bbox_out = self.generator_bbox(
            self.decode_bbox(memory, tgt_bbox, tgt_mask_bbox, tgt_pad_bbox))
        return {"html": html_out, "bbox": bbox_out}


# ---------------------------------------------------------------------
# 2) HierarchicalSharedEncoder
# ---------------------------------------------------------------------
class HierarchicalSharedEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder_html: nn.Module,
        decoder_bbox: nn.Module,
        decoder_center: nn.Module,
        vocab_size_html: int,
        vocab_size_bbox: int,
        d_model: int,
        padding_idx: int,
        max_seq_len:int,
        max_seq_len_html: int,
        max_seq_len_bbox: int,
        dropout: float,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ):
        super().__init__()

        # --- Shared ---
        self.backbone = backbone
        self.encoder  = encoder
        self.pos_embed     = PositionEmbedding(max_seq_len, d_model, dropout)

        # --- Center ---
        vocab_size_center = vocab_size_bbox
        max_seq_len_center = max_seq_len_bbox // 2

        self.pos_embed_center  = PositionEmbedding(max_seq_len_center, d_model, dropout)
        self.decoder_center    = decoder_center
        self.token_embed_center = TokenEmbedding(vocab_size_center, d_model, padding_idx)
        self.generator_center   = nn.Linear(d_model, vocab_size_center)
        self.norm_center_feat   = norm_layer(d_model)

        # --- HTML ---
        self.pos_embed_html    = PositionEmbedding(max_seq_len_html, d_model, dropout)
        self.decoder_html      = decoder_html
        self.token_embed_html  = TokenEmbedding(vocab_size_html, d_model, padding_idx)
        self.generator_html    = nn.Linear(d_model, vocab_size_html)

        # --- BBOX ---
        self.pos_embed_bbox    = PositionEmbedding(max_seq_len_bbox, d_model, dropout)
        self.decoder_bbox      = decoder_bbox
        self.token_embed_bbox  = TokenEmbedding(vocab_size_bbox, d_model, padding_idx)
        self.generator_bbox    = nn.Linear(d_model, vocab_size_bbox)

        # --- Norm & Init ---
        self.norm = norm_layer(d_model)
        self.trunc_normal = partial(nn.init.trunc_normal_, std=init_std,
                                    a=-init_std, b=init_std)
        self.apply(self._init_weights)

    # weight init 동일 …
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight); nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (PositionEmbedding, TokenEmbedding)):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "pos_embed_img", "pos_embed_center",
            "pos_embed_html", "pos_embed_bbox",
            "token_embed_center", "token_embed_html", "token_embed_bbox",
        }

    # --- Encoder / Decoder ---
    def encode(self, src: Tensor) -> Tensor:
        feat   = self.backbone(src)
        feat   = self.pos_embed(feat)
        memory = self.encoder(feat)
        return self.norm(memory)

    def decode_center(self, memory, tgt, tgt_mask, tgt_pad):
        tgt_feat = self.pos_embed_center(self.token_embed_center(tgt))
        return self.decoder_center(tgt_feat, memory, tgt_mask, tgt_pad)

    def decode_html(self, memory, tgt, tgt_mask, tgt_pad):
        tgt_feat = self.pos_embed_html(self.token_embed_html(tgt))
        return self.decoder_html(tgt_feat, memory, tgt_mask, tgt_pad)

    def decode_bbox(self, memory, tgt, tgt_mask, tgt_pad):
        tgt_feat = self.pos_embed_bbox(self.token_embed_bbox(tgt))
        return self.decoder_bbox(tgt_feat, memory, tgt_mask, tgt_pad)

    def forward(
        self, src, tgt_html, tgt_bbox, tgt_center,
        tgt_mask_html, tgt_mask_bbox, tgt_mask_center,
        tgt_pad_html, tgt_pad_bbox, tgt_pad_center
    ) -> Dict[str, Tensor]:

        memory = self.encode(src)

        # print(f'🔥🔥 tgt_center : {tgt_center.shape} | tgt_pad_bbox : {tgt_pad_bbox.shape}')
        center_feat = self.decode_center(memory, tgt_center,
                                            tgt_mask_center, tgt_pad_center)
        center_log  = self.generator_center(center_feat)

        center_norm = self.norm_center_feat(center_feat)        
        center_resized = F.interpolate(
            center_norm.transpose(1, 2),                        # (B, D, Lc)
            size=memory.size(1),                                # Lm
            mode="linear", align_corners=False
        ).transpose(1, 2)                                       # (B, Lm, D)

        enhanced_memory = memory + center_resized

        html_log = self.generator_html(
            self.decode_html(enhanced_memory, tgt_html,
                             tgt_mask_html, tgt_pad_html))
        bbox_log = self.generator_bbox(
            self.decode_bbox(enhanced_memory, tgt_bbox,
                             tgt_mask_bbox, tgt_pad_bbox))

        return {"html": html_log, "bbox": bbox_log, "cp": center_log}
