import matplotlib.pyplot as plt
import numpy as np
import torch as tc
import torch.nn as nn

DEVICE = tc.device("cuda" if tc.cuda.is_available() else "cpu")


def mlp(sizes, Act=nn.ReLU, out=[]):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), Act()]
    return nn.Sequential(*layers[:-1], *out)


def to_np(x):
    if isinstance(x, tc.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: to_np(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_np(v) for v in x]
    return x


def tensor(x):
    if isinstance(x, np.ndarray):
        return tc.from_numpy(x.copy()).float().to(DEVICE)
    if isinstance(x, dict):
        return {k: tensor(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [tensor(v) for v in x]
    return x


def shape(x):
    if isinstance(x, (np.ndarray, tc.Tensor)):
        return x.shape
    if isinstance(x, dict):
        return {k: shape(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [shape(v) for v in x]
    return x


def field_2d(fn_r, a=-1, b=1, N=100):
    xs = ys = np.linspace(a, b, N)
    res = np.zeros((N, N))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            res[i, j] = fn_r(np.array([x, y]))
    return res


def make_path_2d(num=9, T=50, dt=0.2, plot=False):
    def make_one():
        v = np.random.uniform(0, 1, T)
        omega = np.random.uniform(-1, 1, T)
        theta = np.cumsum(omega * dt)
        x = np.cumsum(v * np.cos(theta) * dt)
        y = np.cumsum(v * np.sin(theta) * dt)
        return np.array([v, omega]).T, np.array([x, y, theta]).T

    res = [make_one() for _ in range(num)]
    inp = np.array([r[0] for r in res])
    out = np.array([r[1] for r in res])
    if plot:
        for i in range(num):
            plt.scatter(out[i, :, 0], out[i, :, 1], s=5)
        plt.show()
    return inp, out
