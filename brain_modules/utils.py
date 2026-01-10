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
