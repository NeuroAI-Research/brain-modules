import matplotlib.pyplot as plt
import torch as tc
import torch.nn as nn
import torch.nn.functional as F

from brain_modules.utils import to_np


class GRU(nn.Module):
    def __init__(
        s,
        in_dim,
        out_dim,
        hid_dim=128,
    ):
        super().__init__()
        s.rnn = nn.GRU(in_dim, hid_dim, batch_first=True)
        s.fc = nn.Linear(hid_dim, out_dim)

    def forward(s, x):
        """
        :param x: (batch_size, seq_len, in_dim)
        :return: (batch_size, seq_len, out_dim)
        """
        return s.fc(s.rnn(x)[0])


def wave_to_xy(wave, seq_len):
    xs, ys = [], []
    for t in range(len(wave) - seq_len - 1):
        xs.append(wave[t : t + seq_len])
        ys.append(wave[t + 1 : t + seq_len + 1])
    x = tc.stack(xs).unsqueeze(-1)
    y = tc.stack(ys).unsqueeze(-1)
    return x, y


def make_wave_data(N=1000):
    wave = tc.zeros(N)
    for n in range(3, 8):
        wave += tc.sin(tc.linspace(0, 2**n, N))
    return wave


def supervised_learning():
    gru = GRU(in_dim=1, out_dim=1)
    opt = tc.optim.Adam(gru.parameters(), lr=1e-3)
    wave = make_wave_data(N=1000)
    x, y = wave_to_xy(wave, seq_len=200)
    for e in range(100):
        y_pred = gru(x)
        loss = F.mse_loss(y_pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if e % 10 == 0:
            print(f"{e}, loss: {loss.item():.6f}")
            plt.plot(y[0, :, 0], label="y")
            plt.plot(to_np(y_pred[0, :, 0]), label="y_pred")
            plt.legend()
            plt.savefig("data/test")
            plt.close()
