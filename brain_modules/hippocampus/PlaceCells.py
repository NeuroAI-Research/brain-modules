import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from brain_modules.envs.simple import RoomEnv
from brain_modules.utils import mlp
from brain_modules.utils_plot import plot_pos_embedding


class HippocampalPlaceCells(nn.Module):
    centers: tc.Tensor

    def __init__(
        s,
        obs_dim,
        act_dim,
        hidden=[128, 128],
        embed_dim=16,
    ):
        """
        :param embed_dim: Number of place cells!
        """
        super().__init__()
        s.obs_dim = obs_dim
        s.register_buffer("centers", tc.rand(embed_dim, obs_dim))

        s.encoder = mlp([obs_dim, *hidden, embed_dim])
        s.decoder = mlp([embed_dim, *hidden, obs_dim])
        s.predictor = mlp([embed_dim + act_dim, *hidden, embed_dim])

    def gaussian_embed(s, pos: tc.Tensor, sigma=0.1):
        """
        :param pos: (B, obs_dim)
        returns: (B, embed_dim)
        """
        diff = pos.unsqueeze(1) - s.centers.unsqueeze(0)
        dist_sq = (diff**2).sum(dim=2)
        return tc.exp(-dist_sq / (2 * sigma**2))


# ==== learning ====


def supervised_learning():
    net = HippocampalPlaceCells(obs_dim=1, act_dim=1)
    opt = tc.optim.Adam(net.parameters(), lr=1e-3)
    for e in range(10000):
        pos = tc.rand(512, net.obs_dim)
        emb = net.encoder(pos)
        loss = F.mse_loss(emb, net.gaussian_embed(pos))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 200 == 0:
            print(f"{e+1}, loss: {loss.item():.6f}")
            plot_pos_embedding(pos, emb)


def unsupervised_learning():
    env = RoomEnv()
    obs = env.reset()[0]
    net = HippocampalPlaceCells(obs_dim=obs.shape[-1], act_dim=env.act_dim)
    print(net)
    opt = tc.optim.Adam(net.parameters(), lr=1e-3)
    for e in range(10000):
        act = env.rand_act()
        obs = env.obs
        obs2 = env.step(act)[0]
        # world model
        emb: tc.Tensor = net.encoder(obs)
        emb2_pred = net.predictor(tc.cat([emb, act], dim=1))

        loss_recon = F.mse_loss(net.decoder(emb), obs)
        loss_pred = F.mse_loss(net.decoder(emb2_pred), obs2)
        loss_sparse = 1e-2 * emb.abs().mean()

        loss = loss_recon + loss_pred + loss_sparse

        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 200 == 0:
            print(f"{e+1}, loss: {loss.item():.6f}")
            plot_pos_embedding(env.pos, net.encoder(env.obs))
