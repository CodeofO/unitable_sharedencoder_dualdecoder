import torch
from torch import Tensor, nn
from functools import partial
from typing import Dict

# Assuming these components are available from the original project structure
from model.components import (
    Encoder,
    Decoder,
    PositionEmbedding,
    TokenEmbedding,
)


class SharedEncoder_DualDecoder(nn.Module):
    """Shared encoder dual decoder architecture that takes in a tabular image and generates both HTML and BBOX outputs.
    The shared encoder processes the image once, then two independent decoders generate different task-specific outputs.

    Args:
    ----
        backbone: tabular image processor
        encoder: shared transformer encoder
        decoder_html: transformer decoder for HTML task
        decoder_bbox: transformer decoder for BBOX task
        vocab_size_html: size of the HTML vocabulary
        vocab_size_bbox: size of the BBOX vocabulary
        d_model: feature size
        padding_idx: index of <pad> in the vocabulary
        max_seq_len: max sequence length of generated text
        dropout: dropout rate
        norm_layer: layernorm
        init_std: std in weights initialization
    """

    def __init__(
        self,
        backbone: nn.Module,
        encoder: nn.Module,
        decoder_html: nn.Module,  # Decoder for HTML
        decoder_bbox: nn.Module,  # Decoder for BBOX
        vocab_size_html: int,
        vocab_size_bbox: int,
        d_model: int,
        padding_idx: int,
        max_seq_len: int,
        dropout: float,
        norm_layer: nn.Module,
        init_std: float = 0.02,
    ):
        super().__init__()

        # --- 1. Shared Modules ---
        # These components are used by both tasks.
        self.backbone = backbone
        self.encoder = encoder
        self.pos_embed = PositionEmbedding(
            max_seq_len=max_seq_len, d_model=d_model, dropout=dropout
        )
        self.norm = norm_layer(d_model)

        # --- 2. HTML-Specific Modules ---
        # Independent components dedicated solely to the HTML task.
        self.decoder_html = decoder_html
        self.token_embed_html = TokenEmbedding(
            vocab_size=vocab_size_html, d_model=d_model, padding_idx=padding_idx
        )
        self.generator_html = nn.Linear(d_model, vocab_size_html)

        # --- 3. BBOX-Specific Modules ---
        # Independent components dedicated solely to the BBOX task.
        self.decoder_bbox = decoder_bbox
        self.token_embed_bbox = TokenEmbedding(
            vocab_size=vocab_size_bbox, d_model=d_model, padding_idx=padding_idx
        )
        self.generator_bbox = nn.Linear(d_model, vocab_size_bbox)

        # --- Weight Initialization ---
        self.trunc_normal = partial(
            nn.init.trunc_normal_, std=init_std, a=-init_std, b=init_std
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """Initializes weights for the model's modules."""
        if isinstance(m, nn.Linear):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            self.trunc_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        # Handle both shared and separate embedding layers
        elif isinstance(m, (PositionEmbedding, TokenEmbedding)):
            self.trunc_normal(m.embedding.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        """Specifies which parameters should not have weight decay."""
        # Now includes both token embeddings
        return {"pos_embed", "token_embed_html", "token_embed_bbox"}

    def encode(self, src: Tensor) -> Tensor:
        src_feature = self.backbone(src)
        src_feature = self.pos_embed(src_feature)
        memory = self.encoder(src_feature)
        memory = self.norm(memory)
        return memory

    def decode_html(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        tgt_feature = self.pos_embed(self.token_embed_html(tgt))
        tgt = self.decoder_html(tgt_feature, memory, tgt_mask, tgt_padding_mask)
        return tgt

    def decode_bbox(
        self, memory: Tensor, tgt: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor
    ) -> Tensor:
        tgt_feature = self.pos_embed(self.token_embed_bbox(tgt))
        tgt = self.decoder_bbox(tgt_feature, memory, tgt_mask, tgt_padding_mask)
        return tgt

    def forward(
        self,
        src: Tensor,
        tgt_html: Tensor,
        tgt_bbox: Tensor,
        tgt_mask_html: Tensor,
        tgt_mask_bbox: Tensor,
        tgt_padding_mask_html: Tensor,
        tgt_padding_mask_bbox: Tensor,
    ) -> Dict[str, Tensor]:
        memory = self.encode(src)
        
        tgt_html = self.decode_html(memory, tgt_html, tgt_mask_html, tgt_padding_mask_html)
        tgt_html = self.generator_html(tgt_html)
        
        tgt_bbox = self.decode_bbox(memory, tgt_bbox, tgt_mask_bbox, tgt_padding_mask_bbox)
        tgt_bbox = self.generator_bbox(tgt_bbox)

        return {"html": tgt_html, "bbox": tgt_bbox}
