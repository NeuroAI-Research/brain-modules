import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn.functional as F

from brain_modules.ANNs.GRU import GRU
from brain_modules.utils import field_2d, make_path_2d, tensor, to_np


class BrainGPS:
    n_head_dir_cell = 64
    kappa = 3
    n_grid_cell = 32
    sigma = 0.2

    def __init__(s):
        # head direction cells
        s.pref_dir = np.linspace(0, 2 * np.pi, s.n_head_dir_cell, endpoint=False)
        # grid cells
        a = np.pi / 3 * np.arange(3)
        s.ks = 2 * np.pi * np.array([np.cos(a), np.sin(a)]).T
        s.lams = 0.1 * np.arange(1, s.n_grid_cell + 1)

    def head_dir_pop(s, angle):
        return np.exp(s.kappa * np.cos(angle - s.pref_dir))

    def grid_single(s, r, lam=0.5):
        return sum(np.cos(np.dot(k / lam, r)) for k in s.ks)

    def grid_pop(s, r):
        return np.array([s.grid_single(r, lam) for lam in s.lams])

    def place_single(s, r, r_pref):
        dist = ((r - r_pref) ** 2).sum(axis=-1)
        return np.exp(-dist / (2 * s.sigma**2))

    def plot(s):
        plt.subplot(2, 2, 1)
        plt.scatter(s.pref_dir, s.head_dir_pop(np.pi / 3))
        plt.title(f"{s.n_head_dir_cell} head direction cells\n angle = pi/3")
        plt.ylabel("firing rate")
        plt.xlabel("each cell's preferred direction")

        plt.subplot(2, 2, 2)
        plt.imshow(field_2d(s.grid_single).T, origin="lower")
        plt.title("1 grid cell")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.colorbar(label="firing rate")

        plt.subplot(2, 2, 3)
        r_pref = np.array([[0.6, 0.3], [-0.6, -0.3]])
        fn = lambda r: s.place_single(r, r_pref).sum()
        plt.imshow(field_2d(fn).T, origin="lower")
        plt.title("2 place cells")
        plt.ylabel("y")
        plt.xlabel("x")
        plt.colorbar(label="firing rate")

        plt.tight_layout()
        plt.savefig("data/BrainGPS")
        plt.close()


def learn_path_integration_2d():
    inp, out = tensor(make_path_2d())
    net = GRU(in_dim=2, out_dim=3)
    opt = tc.optim.Adam(net.parameters(), lr=1e-3)
    for e in range(1000):
        pred = net(inp)
        loss = F.mse_loss(pred, out)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 20 == 0:
            print(f"{e}, loss: {loss.item():.6f}")
            for i in range(9):
                plt.subplot(3, 3, i + 1)
                pred = to_np(pred)
                plt.scatter(out[i, :, 0], out[i, :, 1], s=5, label="target")
                plt.scatter(pred[i, :, 0], pred[i, :, 1], s=5, label="predict")
                plt.legend()
            plt.tight_layout()
            plt.savefig("data/path_integration")
            plt.close()
