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


class RoPE(nn.Module):
    def __init__(s, max_T, head_dim, base=1e4):
        super().__init__()
        D = head_dim
        exp = tc.arange(0, D, 2).float() / D
        w = 1 / (base**exp)
        t = tc.arange(max_T).float()
        wt = tc.outer(t, w)
        s.cis = tc.polar(tc.ones_like(wt), wt)

    def forward(s, x: tc.Tensor):
        # x: (B, T, n_head, head_dim) -> (B, T, n_head, head_dim)
        # cis: (max_T, head_dim//2)
        B, T, H, D = x.shape
        z = tc.view_as_complex(x.float().view(B, T, H, D // 2, 2))
        z *= s.cis[:T].view(1, T, 1, D // 2)
        return tc.view_as_real(z).view(*x.shape).type_as(x)


class SwiGLU(nn.Module):
    def __init__(s, dim, hid):
        super().__init__()
        s.gate = nn.Linear(dim, hid, bias=False)
        s.up = nn.Linear(dim, hid, bias=False)
        s.out = nn.Linear(hid, dim, bias=False)

    def forward(s, x):
        # x: (..., dim) -> (..., dim)
        return s.out(F.silu(s.gate(x)) * s.up(x))


class GPTConf:
    # GroupedQueryAttention
    emb_dim = 2
    n_head = 12
    n_kv_head = 3
    head_dim = 6
    # SwiGLU
    hid_dim = 7


class GroupedQueryAttention(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.c = c
        E, D = c.emb_dim, c.head_dim
        assert D % 2 == 0 and c.n_head % c.n_kv_head == 0
        s.q = nn.Linear(E, c.n_head * D, bias=False)
        s.k = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.v = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.o = nn.Linear(c.n_head * D, E, bias=False)

    def forward(s, x: tc.Tensor, rope: RoPE):
        # x: (B, T, emb_dim) -> (B, T, emb_dim)
        B, T, E = x.shape
        D = s.c.head_dim
        R = s.c.n_head // s.c.n_kv_head
        q: tc.Tensor = s.q(x).view(B, T, -1, D)
        k: tc.Tensor = s.k(x).view(B, T, -1, D)
        v: tc.Tensor = s.v(x).view(B, T, -1, D)
        q, k = map(rope, [q, k])
        k, v = [a.repeat_interleave(R, dim=2) for a in [k, v]]
        q, k, v = [a.transpose(1, 2) for a in [q, k, v]]

        score = (q @ k.transpose(2, 3)) / sqrt(D)
        mask = tc.tril(tc.ones(T, T))
        score = score.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(score, dim=-1)
        out = (attn @ v).transpose(1, 2)
        return s.o(out.reshape(B, T, -1))


class TransBlock(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.norm1 = RMSNorm(c.emb_dim)
        s.attn = GroupedQueryAttention(c)
        s.norm2 = RMSNorm(c.emb_dim)
        s.ff = SwiGLU(c.emb_dim, c.hid_dim)

    def forward(s, x, rope):
        # x: (B, T, emb_dim) -> (B, T, emb_dim)
        x += s.attn(s.norm1(x), rope)
        x += s.ff(s.norm2(x))
        return x
