from math import sqrt

import torch as tc
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(s, dim, eps=1e-6):
        super().__init__()
        s.eps = eps
        s.w = nn.Parameter(tc.ones(dim))

    def forward(s, x: tc.Tensor):
        # x: (..., dim) -> (..., dim)
        ms = x.pow(2).mean(-1, keepdim=True)
        return s.w * x * tc.rsqrt(ms + s.eps)


def RoPE(x: tc.Tensor, freq_cis):
    # x: (..., even) -> (..., even)
    x2 = x.float().reshape(*x.shape[:-1], -1, 2)
    z = tc.view_as_complex(x2) * freq_cis
    return tc.view_as_real(z).flatten(-2).type_as(x)


class GPTConf:
    n_embed = 2
    n_head = 12
    n_kv_head = 3
    head_dim = 6


class GroupedQueryAttention(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.c = c
        E, D = c.n_embed, c.head_dim
        assert D % 2 == 0 and c.n_head % c.n_kv_head == 0
        s.q = nn.Linear(E, c.n_head * D, bias=False)
        s.k = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.v = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.o = nn.Linear(c.n_head * D, E, bias=False)

    def forward(s, x: tc.Tensor, freq_cis):
        # x: (B, T, n_embed) -> (B, T, n_embed)
        B, T, E = x.shape
        D = s.c.head_dim
        R = s.c.n_head // s.c.n_kv_head
        q: tc.Tensor = s.q(x).view(B, T, -1, D)
        k: tc.Tensor = s.k(x).view(B, T, -1, D)
        v: tc.Tensor = s.v(x).view(B, T, -1, D)
        q, k = [RoPE(a, freq_cis) for a in [q, k]]
        k, v = [a.repeat_interleave(R, dim=2) for a in [k, v]]
        q, k, v = [a.transpose(1, 2) for a in [q, k, v]]

        score = (q @ k.transpose(2, 3)) / sqrt(D)
        mask = tc.tril(tc.ones(T, T)).to(x.device)
        score = score.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(score, dim=-1)
        out = (attn @ v).transpose(1, 2)
        return s.o(out.reshape(B, T, -1))
