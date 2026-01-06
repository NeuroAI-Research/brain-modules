import math

import matplotlib.pyplot as plt

from brain_modules.utils import to_np


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
