import math

import minimind as mm
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from minimalist_RL.utils import set_seed


class RMSNorm(nn.Module):
    def __init__(s, dim, eps=1e-5):
        super().__init__()
        s.eps = eps
        s.weight = nn.Parameter(tc.ones(dim))

    def _norm(s, x: tc.Tensor):
        return x * tc.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps)

    def forward(s, x: tc.Tensor):
        """
        input: (..., dim)
        output: (..., dim)
        """
        return s.weight * s._norm(x.float()).type_as(x)


class RotaryPositionEmbed(nn.Module):
    cos: tc.Tensor
    sin: tc.Tensor

    end = int(32 * 1024)
    base = 1e6
    unsqueeze_dim = 1

    # scaling
    scaling = None
    orig_max_pos_embed = 2048
    factor = 16
    beta_fast = 32
    beta_slow = 1
    attn_factor = 1

    def __init__(s, dim: int):
        super().__init__()
        exp = tc.arange(0, dim, 2)[: (dim // 2)].float() / dim
        freq = 1 / (s.base**exp)
        attn_factor = 1
        if s.scaling is not None:
            orig_max = s.orig_max_pos_embed
            attn_factor = s.attn_factor
            if s.end / orig_max > 1:

                def inv_dim(b):
                    x = dim * math.log(orig_max / (b * 2 * math.pi))
                    return x / (2 * math.log(s.base))

                low = max(math.floor(inv_dim(s.beta_fast)), 0)
                high = min(math.ceil(inv_dim(s.beta_slow)), dim // 2 - 1)
                x = tc.arange(dim // 2, device=freq.device).float() - low
                ramp = tc.clamp(x / max(high - low, 1e-3), 0, 1)
                freq *= 1 - ramp + ramp / s.factor
        t = tc.arange(s.end, device=freq.device)
        freq = tc.outer(t, freq).float()
        cos = tc.cat([tc.cos(freq), tc.cos(freq)], dim=-1) * attn_factor
        sin = tc.cat([tc.sin(freq), tc.sin(freq)], dim=-1) * attn_factor
        D = s.unsqueeze_dim
        s.register_buffer("cos", cos.unsqueeze(D), persistent=False)
        s.register_buffer("sin", sin.unsqueeze(D), persistent=False)

    def forward(s, q: tc.Tensor, start_pos=0):
        """
        input: (bsz, seq_len, n_head, dim)
        output: (bsz, seq_len, n_head, dim)
        """
        a, b = start_pos, start_pos + q.shape[1]
        i = q.shape[-1] // 2
        q2 = tc.cat((-q[..., i:], q[..., :i]), dim=-1)
        return q * s.cos[a:b] + q2 * s.sin[a:b]


class LLMConf:
    hidden_size = 512
    n_attn_head = 8
    n_kv_head = 2
    dropout = 0

    @property
    def dim(s):
        return s.hidden_size // s.n_attn_head


class Attention(nn.Module):
    def __init__(s, c: LLMConf):
        super().__init__()
        s.c = c
        s.dim = c.hidden_size // c.n_attn_head
        s.q = nn.Linear(c.hidden_size, c.n_attn_head * s.dim, bias=False)
        s.k = nn.Linear(c.hidden_size, c.n_kv_head * s.dim, bias=False)
        s.v = nn.Linear(c.hidden_size, c.n_kv_head * s.dim, bias=False)
        s.o = nn.Linear(c.n_attn_head * s.dim, c.hidden_size, bias=False)
        s.attn_drop = nn.Dropout(c.dropout)
        s.resid_drop = nn.Dropout(c.dropout)

    def forward(s, x: tc.Tensor, rope, past_kv=None, use_cache=False, attn_mask=None):
        """
        input: (bsz, seq_len, hidden_size)
        output: (bsz, seq_len, hidden_size)
        """
        bsz, seq_len, hidden_size = x.shape
        q, k, v = [f(x).view(bsz, seq_len, -1, s.dim) for f in [s.q, s.k, s.v]]
        q, k = rope(q), rope(k)
        if past_kv is not None:
            k = tc.cat([past_kv[0], k], dim=1)
            v = tc.cat([past_kv[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        R = s.c.n_attn_head // s.c.n_kv_head

        def repeat(x: tc.Tensor):
            if R == 1:
                return x
            B, S, H, D = x.shape
            x = x[:, :, :, None, :].expand(B, S, H, R, D)
            return x.reshape(B, S, H * R, D)

        q, k, v = [y.transpose(1, 2) for y in [q, repeat(k), repeat(v)]]
        if (
            seq_len > 1
            and past_kv is None
            and (attn_mask is None or tc.all(attn_mask == 1))
        ):
            drop = s.c.dropout if s.training else 0
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=drop, is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(s.dim)
            y = tc.full((seq_len, seq_len), float("-inf"), device=scores.device)
            scores[:, :, :, -seq_len:] += tc.triu(y, diagonal=1)
            if attn_mask is not None:
                scores += (1.0 - attn_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            out = s.attn_drop(scores) @ v
        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = s.resid_drop(s.o(out))
        return out, past_kv


def test_RMSNorm():
    bsz, dim = 3, 4
    x = tc.rand(bsz, dim)
    y1 = mm.RMSNorm(dim)(x)
    y2 = RMSNorm(dim)(x)
    print(tc.all(y1 == y2))
    print(y2.shape)


def test_RoPE_Attn():
    c = LLMConf()
    bsz, seq_len, n_head, dim = 3, 4, 5, c.dim
    x = tc.rand(bsz, seq_len, n_head, dim)
    cos, sin = mm.precompute_freqs_cis(dim)
    cos, sin = cos[:seq_len], sin[:seq_len]
    y1 = mm.apply_rotary_pos_emb(x, x, cos, sin)[0]
    rope = RotaryPositionEmbed(dim)
    y2 = rope(x)
    print(tc.all(y1 == y2))
    print(y2.shape)

    set_seed()
    m1 = mm.Attention(mm.MiniMindConfig())
    set_seed()
    m2 = Attention(c)

    x = tc.rand(bsz, seq_len, c.hidden_size)
    y1, _ = m1(x, (cos, sin))
    y2, _ = m2(x, rope)
    print(tc.all(y1 == y2))
    print(y2.shape)


test_RMSNorm()
test_RoPE_Attn()
