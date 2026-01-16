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
        z *= s.cis[None, :T, None, :]
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
    emb_dim = 128
    n_head = 4
    n_kv_head = 2
    head_dim = 32
    # SwiGLU
    hid_dim = 128
    # RoPE
    max_T = 64
    # Transformer
    n_layer = 3
    # TextHead
    vocab_size: int


class GroupedQueryAttention(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.c = c
        E, D = c.emb_dim, c.head_dim
        assert D % 2 == 0 and c.n_head % c.n_kv_head == 0
        s.q = nn.Linear(E, c.n_head * D, bias=False)
        s.k = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.v = nn.Linear(E, c.n_kv_head * D, bias=False)
        s.out = nn.Linear(c.n_head * D, E, bias=False)

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
        return s.out((attn @ v).transpose(1, 2).reshape(B, T, -1))


class TransLayer(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.norm1 = RMSNorm(c.emb_dim)
        s.attn = GroupedQueryAttention(c)
        s.norm2 = RMSNorm(c.emb_dim)
        s.ff = SwiGLU(c.emb_dim, c.hid_dim)

        s.attn.out.is_residual = True
        s.ff.out.is_residual = True

    def forward(s, x, rope):
        # x: (B, T, emb_dim) -> (B, T, emb_dim)
        x = x + s.attn(s.norm1(x), rope)
        x = x + s.ff(s.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.layers = nn.ModuleList([TransLayer(c) for _ in range(c.n_layer)])
        s.norm = RMSNorm(c.emb_dim)
        s.rope = RoPE(c.max_T, c.head_dim)
        for m in s.modules():
            init_weights(m, c.n_layer)

    def forward(s, x: tc.Tensor):
        # x: (B, T, emb_dim) -> (B, T, emb_dim)
        for l in s.layers:
            x = l(x, s.rope)
        return s.norm(x)


class TextEmb(nn.Module):
    def __init__(s, c: GPTConf):
        super().__init__()
        s.emb = nn.Embedding(c.vocab_size, c.emb_dim)
        s.out = nn.Linear(c.emb_dim, c.vocab_size, bias=False)
        s.emb.weight = s.out.weight
        s.apply(init_weights)


def init_weights(m: nn.Module, n_layer=1, std=0.02):
    if isinstance(m, nn.Linear):
        if hasattr(m, "is_residual"):
            std *= 1 / sqrt(2 * n_layer)
        nn.init.normal_(m.weight, mean=0.0, std=std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=std)


def test_GPT(GPTCls):
    from brain_modules.utils import CharTokenizer, cross_ent

    with open("test.md") as f:
        text = f.read().replace("\n", ". ")
    tok = CharTokenizer(text)
    data = tok.encode(text)

    c = GPTConf()
    c.vocab_size = tok.vocab_size
    emb = TextEmb(c)
    gpt: GPT = GPTCls(c)
    params = [*emb.parameters(), *gpt.parameters()]
    opt = tc.optim.AdamW(params, lr=1e-3, weight_decay=0.1)

    for e in range(1000):
        starts = tc.randint(len(data) - c.max_T, size=(32,))
        d = tc.stack([data[i : i + c.max_T] for i in starts])
        x, y = d[:, :-1], d[:, 1:]
        if isinstance(gpt, GPT):
            logits = emb.out(gpt(emb.emb(x)))
        else:
            logits = gpt(x)
        yp = tc.argmax(logits, dim=-1)
        loss = cross_ent(logits, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if e % 10 == 0:
            print(
                f"{e}\t loss: {loss.item():.4f}\t y: {tok.decode(y[0])}\t yp: {tok.decode(yp[0])}"
            )
