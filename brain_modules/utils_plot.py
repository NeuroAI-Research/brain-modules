import math
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from brain_modules.utils import to_np

D_TYPE = Dict[str, np.ndarray]


def plot_general(plots: D_TYPE, id, C=2, bins=100):
    C = min(C, len(plots))
    R = int(np.ceil(len(plots) / C))
    plt.figure(figsize=(4 * C, 3 * R))
    i = 0
    for k, v in plots.items():
        v = to_np(v)
        i += 1
        plt.subplot(R, C, i)
        title = k
        if "_hist" in k:
            v = v.flatten()
            if len(v):
                if np.issubdtype(v.dtype, np.number):
                    mean, std = v.mean(), v.std() * 3
                    try:
                        plt.hist(v, bins=bins, range=[mean - std, mean + std])
                    except Exception:
                        pass
                    title = f"{k}\n N={len(v):.1e}, {v.min():.5g} ~ {v.max():.5g}"
                else:
                    plt.hist(v)
                    title = f"{k}\n N={len(v):.1e}"
        elif isinstance(v, dict):
            plt.scatter(v["x"], v["y"], s=3)
            corr = np.corrcoef(v["x"], v["y"])[0, 1]
            title = f"{k}\n corr={corr:.3g}"
        else:
            x = np.arange(len(v))
            if np.any(np.isnan(v)):
                plt.scatter(x, v, s=3)
            else:
                plt.plot(x, v)
        plt.title(title)
    plt.tight_layout()
    plt.savefig(id)
    plt.close()


def plot_pos_embedding(pos, emb, pos_idx=0, C=4):
    """
    :param pos: (B, env_dim)
    :param emb: (B, emb_dim)
    """
    pos, emb = to_np([pos, emb])
    x = pos[:, pos_idx]
    N = min(emb.shape[-1], 16)
    R = math.ceil(N / C)
    plt.figure(figsize=(3 * C, 3 * R))
    y_lim = [emb.min(), emb.max()]
    for i in range(N):
        plt.subplot(R, C, i + 1)
        plt.scatter(x, emb[:, i], s=5)
        plt.ylim(*y_lim)
        plt.title(f"place_cell_{i}")
    plt.suptitle(f"place_cell (embed_dim) activation VS position_{pos_idx}")
    plt.tight_layout()
    plt.savefig("data/plot_pos_embedding")
    plt.close()
