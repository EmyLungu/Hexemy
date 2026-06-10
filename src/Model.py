import math

import torch
import torch.nn as nn


class HexModel(nn.Module):
    def __init__(
        self,
        hex_vocab_size: int,
        c_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model

        self.hex_embedding: nn.Embedding = nn.Embedding(
            hex_vocab_size, d_model
        )
        self.c_embeddin: nn.Embedding = nn.Embedding(c_vocab_size, d_model)
        self.pos_encoder: PositionalEncoder = PositionalEncoder(d_model)

        self.transformer: nn.Transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True,
        )

        self.fc_out = nn.Linear(d_model, c_vocab_size)

    def forward(self, src, tgt, src_pad_idx, tgt_pad_idx) -> torch.Tensor:
        src_key_padding_mask = self.make_src_padding_mask(src, src_pad_idx)
        tgt_key_padding_mask = self.make_src_padding_mask(tgt, tgt_pad_idx)

        tgt_len: int = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt_len, device=src.device, dtype=torch.bool
        )

        src_emb = self.pos_encoder(
            self.hex_embedding(src) * math.sqrt(self.d_model)
        )
        tgt_emb = self.pos_encoder(
            self.c_embedding(tgt) * math.sqrt(self.d_model)
        )

        out = self.transformer(
            src_emb,
            tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.fc_out(out)

    def make_src_padding_mask(self, src, pad_idx):
        """
        [Batch_Size, Src_Len] -> Tell attention to ignore [PAD] positions
        """
        return src == pad_idx


class PositionalEncoder(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10_000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe.unsqueeze(0)[:, : x.size(1), :]
