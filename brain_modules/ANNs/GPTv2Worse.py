import math

import torch as tc
import torch.nn as nn
import torch.nn.functional as F

import brain_modules.refs.nanochat as nano
from brain_modules.ANNs.GPT import GPTConf


def rms_norm(x: tc.Tensor):
    return F.rms_norm(x, (x.size(-1),))


class RoPE(nn.Module):
    def __init__(s, max_T, head_dim, base=1e4):
        super().__init__()
        D = head_dim
        exp = tc.arange(0, D, 2).float() / D
        w = 1 / (base**exp)
        t = tc.arange(max_T).float()
        wt = tc.outer(t, w)
        s.cos = wt.cos().bfloat16()[None, :, None, :]
        s.sin = wt.sin().bfloat16()[None, :, None, :]

    def forward(s, x: tc.Tensor, t1=0):
        # x: (B, T, n_head, head_dim) -> (B, T, n_head, head_dim)
        T, d = x.shape[1], x.shape[3] // 2
        cos, sin = s.cos[:, t1 : t1 + T], s.sin[:, t1 : t1 + T]
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return tc.cat([y1, y2], 3)


class GroupedQueryAttention(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.c = c
        E, D = c.emb_dim, c.head_dim
        s.c_q = nn.Linear(E, c.n_head * D, bias=False)
        s.c_k = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.c_v = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.c_proj = nn.Linear(c.n_head * D, E, bias=False)

    def forward(s, x: tc.Tensor, rope):
        # x: (B, T, emb_dim) -> (B, T, emb_dim)
        B, T, E = x.shape
        D = s.c.head_dim
        q: tc.Tensor = s.c_q(x).view(B, T, -1, D)
        k: tc.Tensor = s.c_k(x).view(B, T, -1, D)
        v: tc.Tensor = s.c_v(x).view(B, T, -1, D)
        q, k = map(rms_norm, map(rope, [q, k]))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return s.c_proj(y.contiguous().view(B, T, -1))


class FeedForward(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        E = c.emb_dim
        s.c_fc = nn.Linear(E, 4 * E, bias=False)
        s.c_proj = nn.Linear(4 * E, E, bias=False)

    def forward(s, x):
        return s.c_proj(F.relu(s.c_fc(x)).square())


class TransLayer(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.attn = GroupedQueryAttention(c)
        s.mlp = FeedForward(c)

    def forward(s, x, rope):
        x = x + s.attn(rms_norm(x), rope)
        x = x + s.mlp(rms_norm(x))
        return x


class GPTv2(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.c = c
        vocab_size2 = math.ceil(c.vocab_size / 64) * 64
        wte = nn.Embedding(vocab_size2, c.emb_dim)
        h = nn.ModuleList([TransLayer(c) for _ in range(c.n_layer)])
        s.transformer = nn.ModuleDict({"wte": wte, "h": h})
        s.lm_head = nn.Linear(c.emb_dim, vocab_size2, bias=False)
        s.rope = RoPE(c.max_T, c.head_dim)

        s.resid_lambdas = nn.Parameter(tc.ones(c.n_layer))
        s.x0_lambdas = nn.Parameter(tc.zeros(c.n_layer))

    def forward(s, idx: tc.Tensor):
        x = rms_norm(s.transformer.wte(idx))
        x0 = x
        for i, layer in enumerate(s.transformer.h):
            x = s.resid_lambdas[i] * x + s.x0_lambdas[i] * x0
            x = layer(x, s.rope)
        x = rms_norm(x)
        soft_cap = 15
        logits = s.lm_head(x).float()[..., : s.c.vocab_size]
        logits = soft_cap * tc.tanh(logits / soft_cap)
        return logits


def test_correctness():
    c1 = nano.GPTConfig()
    c2 = GPTConf()
    c1.sequence_len = c2.max_T
    c1.vocab_size = c2.vocab_size = 100
    c1.n_layer = c2.n_layer
    c1.n_head = c2.n_head
    c1.n_kv_head = c2.n_kv_head
    c1.n_embd = c2.emb_dim

    idx = tc.randint(0, c1.vocab_size, (2, c2.max_T))

    g1 = nano.GPT(c1)
    g2 = GPTv2(c2)
    g2.load_state_dict(g1.state_dict())
    y1 = g1(idx)
    y2 = g2(idx)
    print(y1.shape, tc.all(y1 == y2))
