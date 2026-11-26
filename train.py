"""Train the monsoon twin from india_climate_train.nc and write weights/twin.pt.

  python train.py --years 2005-2017 --val-years 2018-2019 --out /tmp/dev.pt     # development run
  python train.py --years 2005-2021 --rain-cut-from /tmp/dev.pt                   # the shipped weights

Trains on the Apple GPU with MLX (open source, MIT); the weights are written as the PyTorch checkpoint serve.py loads.
"""
import argparse
import datetime as dt
import math
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import torch
import xarray as xr
from mlx.utils import tree_map

import mlx_twin
import twin

ROOT = Path(__file__).parent
DATA = ROOT.parent / "yug-monsoon-kit-main" / "release" / "india_climate_train.nc"


def load(path: Path = DATA):
    """-> X [N,6,32,36] float32 physical units, days [N] datetime.date, static [2,32,36] (land_sea_mask, orography_m)."""
    with xr.open_dataset(path) as ds:
        X = ds["data"].values.astype("f4")
        days = np.array([d.date() for d in ds["time"].to_index()])
        static = ds["static"].values.astype("f4")
    return X, days, static


def slot(d: dt.date) -> int:
    """Calendar slot 0..365 with 29 Feb at 59, the indexing of the 366-day climatology."""
    leap = d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)
    return d.timetuple().tm_yday - 1 + (d.month > 2 and not leap)


def climatology(X: np.ndarray, days, smooth: int = 15) -> np.ndarray:
    """[N,6,32,36] -> [366,6,32,36]: mean per calendar slot, then a circular running mean over `smooth` days."""
    idx = np.array([slot(d) for d in days])
    tot = np.zeros((366,) + X.shape[1:]); cnt = np.bincount(idx, minlength=366)
    np.add.at(tot, idx, X)
    mean = tot / np.maximum(cnt, 1)[:, None, None, None]
    h = smooth // 2
    return np.mean([np.roll(mean, -o, axis=0) for o in range(-h, h + 1)], axis=0).astype("f4")


# ---- training --------------------------------------------------------------------------------------------
LEADS = 8                                                  # longest lead trained on (the score uses 1, 4 and 8)
LEAD_W = np.array([1.0 if k in (1, 4, 8) else 0.3 for k in range(1, LEADS + 1)], "f4")
LAT_W = np.cos(np.deg2rad(np.arange(6.5, 38.0, 1.0))); LAT_W = (LAT_W / LAT_W.mean()).astype("f4")


def years(spec: str) -> set:
    a, _, b = spec.partition("-")
    return set(range(int(a), int(b or a) + 1))


def starts(days, yrs: set) -> np.ndarray:
    """Last input days whose 2 earlier days and LEADS later days all fall in yrs."""
    return np.array([i for i in range(2, len(days) - LEADS)
                     if days[i - 2].year in yrs and days[i + LEADS].year in yrs])


def sincos(d: dt.date):
    a = 2 * math.pi * d.timetuple().tm_yday / 365
    return math.sin(a), math.cos(a)


class Batches:
    """Everything a training step needs, precomputed per start and kept on the GPU (the data too, 172 MB):
    climatology slots, season angle, reference errors."""

    def __init__(self, X, days, idx, clim):
        self.idx = idx
        sc = [sincos(days[i]) for i in idx]
        self.slot_in = np.array([[twin.clim_slot(s, c, k) for k in (-2, -1, 0)] for s, c in sc])
        self.slot_out = np.array([[twin.clim_slot(s, c, k) for k in range(1, LEADS + 1)] for s, c in sc])
        self.angle = np.array([math.atan2(s, c) for s, c in sc], "f4")
        # reference errors per start, lead and channel (lat-weighted MSE): persistence and climatology
        mse = lambda a, b: ((a - b) ** 2 * LAT_W[:, None]).mean(axis=(-2, -1))
        pers = np.stack([mse(X[idx], X[idx + k]) for k in range(1, LEADS + 1)], 1)
        clm = np.stack([mse(clim[self.slot_out[:, k - 1]], X[idx + k]) for k in range(1, LEADS + 1)], 1)
        self.ref_pool = mx.array(np.minimum(pers.mean(0), clm.mean(0)).astype("f4"))     # [LEADS,6]
        self.X = mx.array(X.transpose(0, 2, 3, 1))                                          # channels-last [N,H,W,6]
        self.i, self.s_in, self.s_out = mx.array(idx), mx.array(self.slot_in), mx.array(self.slot_out)
        self.ang, self.ref = mx.array(self.angle), mx.array(np.sqrt(np.minimum(pers, clm)).astype("f4"))

    def get(self, b, unroll):
        b = mx.array(b)
        x = self.X[self.i[b][:, None] + mx.arange(-2, unroll + 1)]                          # [B, 3 + unroll, H, W, 6]
        return x[:, :3], x[:, 3:], self.s_in[b], self.s_out[b, :unroll], self.ang[b], self.ref[b, :unroll]


def loss_fn(nets, F, batch, unroll, kind, ref_pool):
    """kind "mse": squared error scaled by the pooled reference error (readings A and C).
    kind "score": half that, half each sample's RMSE over its own better reference (reading B)."""
    x_in, x_out, s_in, s_out, angle, ref_rmse = batch
    zp = mlx_twin.roll(nets, F, mlx_twin.anomaly(F, x_in, s_in), angle, s_out)
    err = (zp - mlx_twin.anomaly(F, x_out, s_out)) * F.scale
    err2 = (err ** 2 * mx.array(LAT_W)[:, None, None]).mean(axis=(2, 3))                    # [B,U,6]
    w = mx.array(LEAD_W[:unroll])
    pooled = ((err2 / ref_pool[:unroll]).mean(-1) * w).sum(1).mean() / w.sum()
    if kind == "mse":
        return pooled
    own = ((mx.sqrt(err2 + 1e-6) / ref_rmse).mean(-1) * w).sum(1).mean() / w.sum()
    return 0.5 * pooled + 0.5 * own


def fit(nets, F, train_b: Batches, val_b, schedule, kind, batch, lr, seed, log=print, on_phase=None):
    """Trains nets in place (bfloat16 convolutions, float32 weights) and returns the EMA weights as a new Nets."""
    rng = np.random.default_rng(seed)
    opt = optim.AdamW(learning_rate=lr, weight_decay=1e-4)
    opt.init(nets.trainable_parameters())
    ema = tree_map(lambda p: mx.array(p), nets.parameters())
    ema_nets = mlx_twin.Nets(len(nets.nets), nets.nets[0].stem.weight.shape[0], len(nets.nets[0].blocks),
                             nets.nets[0].compute)
    state = [nets.state, opt.state]
    t0 = time.time()
    for phase, (unroll, iters) in enumerate(schedule):
        peak = lr if phase == 0 else lr / 3
        grad = nn.value_and_grad(nets, lambda m, *bt: loss_fn(m, F, bt, unroll, kind, train_b.ref_pool))

        def train_step(*bt):                               # one optimiser step: gradients, clipping, update
            loss, g = grad(nets, *bt)
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(nets, g)
            return loss
        train_step = mx.compile(train_step, inputs=state, outputs=state)
        for it in range(iters):                                                    # warm up, then cosine down
            opt.learning_rate = peak * min(1.0, (it + 1) / 200) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * it / iters)))
            loss = train_step(*train_b.get(rng.choice(len(train_b.idx), batch), unroll))
            ema = tree_map(lambda e, p: 0.999 * e + 0.001 * p, ema, nets.parameters())
            mx.eval(loss, ema, state)
            if (it + 1) % 500 == 0 or it + 1 == iters:
                msg = f"unroll {unroll} it {it + 1}/{iters} loss {loss.item():.4f}"
                if val_b is not None:
                    ema_nets.update(ema)
                    vb = np.arange(0, len(val_b.idx), 7)
                    msg += f" | val {loss_fn(ema_nets, F, val_b.get(vb, unroll), unroll, kind, val_b.ref_pool).item():.4f}"
                log(f"{msg} | {time.time() - t0:.0f}s", flush=True)
        if on_phase is not None:
            ema_nets.update(ema); on_phase(phase, unroll, ema_nets)
    ema_nets.update(ema)
    return ema_nets


# ---- heavy-rain cutoffs ----------------------------------------------------------------------------------
def _csi(pred, true, axis=None):
    h = (pred & true).sum(axis); m = (~pred & true).sum(axis); f = (pred & ~true).sum(axis)
    with np.errstate(all="ignore"):
        return np.where(h + m + f > 0, h / (h + m + f), np.nan)


def _csi_skill(c, ref):
    with np.errstate(all="ignore"):
        return np.where(ref < 1, (c - ref) / (1 - ref), np.nan)


def best_cutoff(raw, truth, persist, clim, grid) -> float:
    """For one lead: the cutoff c (rain >= c is emitted as heavy) that maximises the worse of per-window and pooled
    heavy-rain CSI skill against the better reference. inf (no lift) unless some cutoff does strictly better.
    All arrays [windows,32,36] in mm/day."""
    import warnings
    t = truth > twin.HEAVY_RAIN
    ref_w = np.fmax(_csi(persist > twin.HEAVY_RAIN, t, (1, 2)), _csi(clim > twin.HEAVY_RAIN, t, (1, 2)))
    ref_p = np.fmax(_csi(persist > twin.HEAVY_RAIN, t), _csi(clim > twin.HEAVY_RAIN, t))

    def objective(c):
        pred = (raw >= c) | (raw > twin.HEAVY_RAIN)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            per_window = np.nanmean(_csi_skill(_csi(pred, t, (1, 2)), ref_w))
        return min(per_window, float(_csi_skill(_csi(pred, t), ref_p)))

    best_c, best = np.inf, objective(np.inf)
    for c in grid:
        v = objective(c)
        if v > best + 1e-9:
            best_c, best = float(c), v
    return best_c


def tune_rain_cut(f, X, days, idx, clim, log=print) -> list:
    """Per-step heavy-rain cutoffs from windows idx (never the years the score is reported on). f: a forecaster
    with the lift off."""
    init = np.stack([X[i - 2:i + 1] for i in idx]); sc = np.array([sincos(days[i]) for i in idx], "f4")
    raw = f(init, sc, LEADS + 2)[:, :, twin.TP]
    cuts = []
    for k in range(LEADS + 2):
        clm = clim[[slot(days[i + k + 1]) for i in idx], twin.TP]
        cuts.append(best_cutoff(raw[:, k], X[idx + k + 1, twin.TP], X[idx, twin.TP], clm, np.arange(4.0, 10.0, 0.25)))
    cuts += [cuts[-1]] * (twin.MAX_STEPS - len(cuts))
    log("heavy-rain cutoffs by day: " + " ".join(f"{c:.2f}" for c in cuts), flush=True)
    return cuts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--out", type=Path, default=twin.WEIGHTS)
    ap.add_argument("--years", default="2005-2017", help="training years, first-last")
    ap.add_argument("--val-years", default="", help="years to report validation loss on, first-last")
    ap.add_argument("--schedule", default="1:5000,4:3000", help="unroll:iterations for each phase")
    ap.add_argument("--loss", choices=("mse", "score"), default="score")
    ap.add_argument("--nets", type=int, default=1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rain-cut-from", type=Path, help="copy the heavy-rain cutoffs from these weights")
    ap.add_argument("--tune-rain", type=Path, help="no training: tune the heavy-rain cutoffs of these weights on --val-years")
    a = ap.parse_args(argv)

    mx.random.seed(a.seed)
    X, days, static = load(a.data)
    if a.tune_rain:
        t = twin.load(a.tune_rain)
        t.rain_cut.fill_(math.inf)
        F = mlx_twin.Fields(t)
        # reference climatology: the model's own, built from its training years only
        clim = np.array(F.clim).transpose(0, 3, 1, 2)
        cuts = tune_rain_cut(mlx_twin.forecaster(mlx_twin.from_torch(t), F), X, days, starts(days, years(a.val_years)), clim)
        t.rain_cut.copy_(torch.tensor(cuts))
        twin.save(t, a.out); print("wrote", a.out)
        return
    tr = years(a.years)
    keep = np.array([d.year in tr for d in days])
    clim = climatology(X[keep], days[keep])
    anom = X[keep] - clim[[slot(d) for d in days[keep]]]
    t = twin.Twin(a.nets, a.width, a.blocks)
    t.set_fields(clim, anom.std(axis=(0, 2, 3)), static, X[keep].mean(axis=(0, 2, 3)), X[keep].std(axis=(0, 2, 3)))
    del anom
    if a.rain_cut_from:
        t.rain_cut.copy_(twin.load(a.rain_cut_from).rain_cut)
    F = mlx_twin.Fields(t)
    train_b = Batches(X, days, starts(days, tr), clim)
    val_b = Batches(X, days, starts(days, years(a.val_years)), clim) if a.val_years else None
    schedule = [tuple(int(v) for v in p.split(":")) for p in a.schedule.split(",")]
    print(f"{len(train_b.idx)} training starts, schedule {schedule}, loss {a.loss}, MLX {mx.default_device()}", flush=True)

    def save_phase(phase, unroll, ema_nets):               # one checkpoint per phase, to compare unroll lengths
        if len(schedule) > 1 and phase < len(schedule) - 1:
            twin.save(mlx_twin.to_torch(ema_nets, t), a.out.with_suffix(f".unroll{unroll}.pt"))
    nets = mlx_twin.Nets(a.nets, a.width, a.blocks, mx.bfloat16)
    ema_nets = fit(nets, F, train_b, val_b, schedule, a.loss, a.batch, a.lr, a.seed, on_phase=save_phase)
    twin.save(mlx_twin.to_torch(ema_nets, t), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
