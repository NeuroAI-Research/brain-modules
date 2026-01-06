import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from brain_modules.utils import mlp, to_np


class HippocampalPlaceCells(nn.Module):
    centers: tc.Tensor

    def __init__(s, num_place_cells=64, hidden=[128, 128], env_dim=1, sigma=0.1):
        super().__init__()
        s.env_dim, s.sigma = env_dim, sigma
        s.register_buffer("centers", tc.rand(num_place_cells, env_dim))
        s.encoder = mlp([env_dim, *hidden, num_place_cells])

    def gaussian_targets(s, pos: tc.Tensor):
        """
        :param pos: (B, env_dim)
        returns: (B, num_place_cells)
        """
        diff = pos.unsqueeze(1) - s.centers.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        return tc.exp(-dist_sq / (2 * s.sigma**2))

    def plot_pos_encoding(s, embedding: tc.Tensor):
        """
        :param embedding: (B, num_place_cells)
        """
        x = to_np(s.centers[:, 0])
        for i, y in enumerate(to_np(embedding[:3])):
            plt.scatter(x, y, s=5, label=f"sample {i}")
        plt.legend()
        plt.show()


def main():
    net = HippocampalPlaceCells()
    opt = tc.optim.Adam(net.parameters(), lr=1e-3)
    for e in range(2000):
        pos = tc.rand(512, net.env_dim)
        target = net.gaussian_targets(pos)
        pred = net.encoder(pos)
        loss = F.mse_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 200 == 0:
            print(f"{e+1}, loss: {loss.item():.4f}")
            net.plot_pos_encoding(pred)


if __name__ == "__main__":
    main()
