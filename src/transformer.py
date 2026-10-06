"""A small Transformer encoder for the price window, written from scratch in PyTorch (no nn.MultiheadAttention,
no nn.TransformerEncoder): scaled dot-product attention, multi-head projections, pre-LayerNorm residual blocks and a
sinusoidal positional encoding. The 144-hour window is cut into patches of 6 hours (as in PatchTST), so the
sequence the attention sees is 24 tokens long.

It plugs into the same SequenceModel base as the LSTM, CNN and MLP, so it is held to the same no-leakage window."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models import SequenceModel


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """softmax(q k^T / sqrt(d)) v for tensors shaped (batch, heads, tokens, head_dim); also returns the weights."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    weights = torch.softmax(scores, dim=-1)
    return weights @ v, weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)  # each (b, heads, n, dk)
        ctx, weights = scaled_dot_product_attention(q, k, v)
        y = self.out(ctx.transpose(1, 2).reshape(b, n, d))
        return (y, weights) if return_weights else y


class EncoderBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.mlp(self.ln2(x)))


def sinusoidal_positions(n_tokens: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(n_tokens).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(n_tokens, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class PatchTransformerNet(nn.Module):
    def __init__(self, lookback: int = 144, patch: int = 6, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        if lookback % patch:
            raise ValueError("lookback must be a multiple of the patch size")
        self.patch = patch
        self.embed = nn.Linear(patch, d_model)
        self.register_buffer("pos", sinusoidal_positions(lookback // patch, d_model))
        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, lookback, 1)
        b, n, _ = x.shape
        tokens = self.embed(x.reshape(b, n // self.patch, self.patch)) + self.pos
        for block in self.blocks:
            tokens = block(tokens)
        return self.head(self.ln(tokens)[:, -1, :]).squeeze(-1)  # the most recent patch summarises the window


class TransformerModel(SequenceModel):
    def _build_net(self) -> nn.Module:
        return PatchTransformerNet(self.lookback, patch=6, d_model=2 * self.hidden_size, n_heads=4, n_layers=2)
