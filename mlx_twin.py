"""MLX (open source, MIT) copy of the twin's network and rollout: used for training and local evaluation on the GPU.

Layer for layer the same as twin.py, in MLX's channels-last layout. Weights convert both ways, so the shipped file is
still the PyTorch checkpoint that serve.py loads. Never imported by the server.
"""
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch

import twin

H, W = twin.NLAT, twin.NLON


class Step(nn.Module):
    """twin.Step in MLX. Weights stay float32; convolutions run in `compute` (bfloat16 to train fast)."""

    def __init__(self, width: int, blocks: int, compute=mx.float32):
        super().__init__()
        self.compute = compute
        self.stem = nn.Conv2d(twin.N_IN, width, 3, padding=1)
        self.blocks = [nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)]
        self.head = nn.Conv2d(width, twin.NC, 1)
        self.head.weight = mx.zeros_like(self.head.weight); self.head.bias = mx.zeros_like(self.head.bias)

    def _conv(self, c, x, pad):
        y = mx.conv2d(x.astype(self.compute), c.weight.astype(self.compute), padding=pad)
        return y.astype(mx.float32) + c.bias

    def __call__(self, x):
        h = self._conv(self.stem, x, 1)
        for c in self.blocks:
            h = h + self._conv(c, nn.gelu(h), 1)
        return self._conv(self.head, nn.gelu(h), 0)


class Nets(nn.Module):
    """The trainable part: one or more Steps, averaged."""

    def __init__(self, n: int, width: int, blocks: int, compute=mx.float32):
        super().__init__()
        self.nets = [Step(width, blocks, compute) for _ in range(n)]

    def __call__(self, x):
        return sum(n(x) for n in self.nets) / len(self.nets)


class Fields:
    """The fixed fields of a PyTorch twin (climatology, scales, static maps, rain cutoffs), channels-last, on the GPU."""

    def __init__(self, t: twin.Twin):
        g = lambda name: getattr(t, name).detach().cpu().numpy()
        self.clim = mx.array(g("clim").transpose(0, 2, 3, 1))              # [366,H,W,6]
        self.scale, self.mean, self.std = (mx.array(g(n).reshape(twin.NC)) for n in ("scale", "mean", "std"))
        self.static = mx.array(g("static").transpose(1, 2, 0))             # [H,W,4]
        self.rain_cut = g("rain_cut")


def anomaly(F: Fields, x, slots):
    return (x - F.clim[slots]) / F.scale


def physical(F: Fields, z, slots):
    return F.clim[slots] + z * F.scale


def roll(nets, F: Fields, z, angle, slots):
    """Same as twin.Twin.roll. z [B,3,H,W,6] scaled anomalies, oldest first; angle [B]; slots [B,steps]."""
    B = z.shape[0]
    st = mx.broadcast_to(F.static, (B, H, W, 4))
    out = []
    for k in range(slots.shape[1]):
        c = F.clim[slots[:, k]]
        a = angle + 2 * math.pi * (k + 1) / 365
        season = mx.broadcast_to(mx.stack([mx.sin(a), mx.cos(a)], -1)[:, None, None, :], (B, H, W, 2))
        x = mx.concatenate([z[:, 0], z[:, 1], z[:, 2], (c - F.mean) / F.std, st, season], -1)
        new = z[:, -1] + nets(x)
        rain = mx.maximum(new[..., 1:2], -c[..., 1:2] / F.scale[1])        # rain >= 0
        new = mx.concatenate([new[..., :1], rain, new[..., 2:]], -1)
        out.append(new)
        z = mx.stack([z[:, 1], z[:, 2], new], 1)
    return mx.stack(out, 1)


def forecaster(nets, F: Fields, batch: int = 128):
    """A numpy forecaster like twin.Twin.forecast: init [B,3,6,32,36], sincos [B,2] -> [B,steps,6,32,36] physical."""
    def f(init, sincos, steps, pert=None):
        init = np.asarray(init, "f4")
        pert = np.zeros_like(init[:, -1]) if pert is None else np.asarray(pert, "f4")
        x = np.concatenate([init[:, :-1], (init[:, -1] + pert)[:, None]], 1)
        outs = []
        for b in range(0, len(x), batch):
            sc = sincos[b:b + batch]
            s_in = mx.array([[twin.clim_slot(s, c, k) for k in (-2, -1, 0)] for s, c in sc])
            s_out = mx.array([[twin.clim_slot(s, c, k) for k in range(1, steps + 1)] for s, c in sc])
            ang = mx.array([math.atan2(s, c) for s, c in sc], dtype=mx.float32)
            z = roll(nets, F, anomaly(F, mx.array(x[b:b + batch].transpose(0, 1, 3, 4, 2)), s_in), ang, s_out)
            outs.append(np.array(physical(F, z, s_out)).transpose(0, 1, 4, 2, 3))
        return twin.emit(np.concatenate(outs), F.rain_cut[:steps])
    return f


# ---- conversion to and from the PyTorch checkpoint (only array copies, no maths) ----
def _pairs(t: twin.Twin, nets: Nets):
    for tn, mn in zip(t.nets, nets.nets):
        yield from [(tn.stem, mn.stem), *zip(tn.blocks, mn.blocks), (tn.head, mn.head)]


def from_torch(t: twin.Twin, compute=mx.float32) -> Nets:
    nets = Nets(t.config["n_nets"], t.config["width"], t.config["blocks"], compute)
    for tc, mc in _pairs(t, nets):
        mc.weight = mx.array(tc.weight.detach().cpu().numpy().transpose(0, 2, 3, 1))
        mc.bias = mx.array(tc.bias.detach().cpu().numpy())
    return nets


def to_torch(nets: Nets, t: twin.Twin) -> twin.Twin:
    """Copies the MLX weights into the PyTorch twin t (whose fields are already set) and returns it."""
    with torch.no_grad():
        for tc, mc in _pairs(t, nets):
            tc.weight.copy_(torch.from_numpy(np.array(mc.weight).transpose(0, 3, 1, 2).copy()))
            tc.bias.copy_(torch.from_numpy(np.array(mc.bias)))
    return t


def load_forecaster(path):
    """The forecaster for a saved checkpoint, computed in MLX float32."""
    t = twin.load(path)
    return forecaster(from_torch(t), Fields(t))
