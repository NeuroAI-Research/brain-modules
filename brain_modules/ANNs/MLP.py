import torch.nn as nn

from brain_modules.ANNs.GPT import RMSNorm, SwiGLU, init_weights


class SwiGLUMLPLayer(nn.Module):
    def __init__(s, emb_dim, hid_dim):
        super().__init__()
        s.norm2 = RMSNorm(emb_dim)
        s.ff = SwiGLU(emb_dim, hid_dim)
        s.ff.out.is_residual = True

    def forward(s, x):
        x = x + s.ff(s.norm2(x))
        return x


class SwiGLUMLP(nn.Module):
    def __init__(s, in_dim, out_dim, n_layer=3, emb_dim=64, hid_dim=64):
        super().__init__()
        s.emb = nn.Linear(in_dim, emb_dim, bias=False)
        L = SwiGLUMLPLayer
        s.layers = nn.ModuleList([L(emb_dim, hid_dim) for _ in range(n_layer)])
        s.norm = RMSNorm(emb_dim)
        s.out = nn.Linear(emb_dim, out_dim, bias=False)
        for m in s.modules():
            init_weights(m, n_layer)

    def forward(s, x):
        x = s.emb(x)
        for l in s.layers:
            x = l(x)
        return s.out(s.norm(x))
