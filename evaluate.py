"""Local stand-in for the Stage 1 grader. Scores a forecaster with the kit's own scoring functions.

  python evaluate.py --years 2018 2019                  # the twin in weights/twin.pt
  python evaluate.py --years 2018 2019 --baseline       # damped persistence, for comparison

A forecaster is f(init [B,3,6,32,36], doy_sincos [B,2], steps, perturbation [B,6,32,36] | None) -> [B,steps,6,32,36],
physical units throughout. The grader's per-window formula is not published, so forecast skill is reported three ways:
  A  per window, leads 1/4/8 pooled into one RMSE      B  per window, each lead alone, then averaged
  C  all windows pooled
"""
import warnings

import numpy as np
from monsoon_r1 import contract as C, scoring as S

import mlx_twin
import train
import twin

LEADS = np.array(C.LEAD_DAYS)
STEPS = C.SCORE_STEPS


def starts(days, years, horizon: int = STEPS) -> np.ndarray:
    """Indices of last input days in `years` with 2 earlier days and `horizon` later days in the data."""
    years = set(years)
    return np.array([i for i in range(2, len(days) - horizon) if days[i].year in years])


def inputs(X, days, idx):
    init = np.stack([X[i - 2:i + 1] for i in idx])
    sincos = np.array([C.doy_sincos(days[i].timetuple().tm_yday) for i in idx], "f4")
    return init, sincos


def run(f, X, days, idx, steps: int = STEPS, shock=None, batch: int = 128) -> np.ndarray:
    """Forecasts for every start in idx. `shock(last_state [6,32,36]) -> delta` builds a perturbation per window."""
    out = []
    for b in range(0, len(idx), batch):
        init, sc = inputs(X, days, idx[b:b + batch])
        pert = None if shock is None else np.stack([shock(s) for s in init[:, -1]])
        out.append(f(init, sc, steps, pert))
    return np.concatenate(out)


# ---- baselines -------------------------------------------------------------------------------------------
def persistence(init, sincos, steps, pert=None):
    last = init[:, -1] + (0 if pert is None else pert)
    return np.repeat(last[:, None], steps, axis=1)


def damped_persistence(X, days, clim, fit_idx):
    """Climatology plus the last anomaly times a per-channel, per-lead regression coefficient fitted on fit_idx."""
    A = X - clim[[train.slot(d) for d in days]]
    a = []
    for lead in range(1, C.MAX_STEPS + 1):
        i = fit_idx[fit_idx + lead < len(X)]
        a.append((A[i + lead] * A[i]).sum(axis=(0, 2, 3)) / (A[i] ** 2).sum(axis=(0, 2, 3)))
    a = np.array(a, "f4")[:, :, None, None]

    def f(init, sincos, steps, pert=None):
        last = init[:, -1] + (0 if pert is None else pert)
        out = np.empty((len(last), steps) + last.shape[1:], "f4")
        for b, (s, c) in enumerate(sincos):
            anom = last[b] - clim[twin.clim_slot(s, c)]
            for k in range(steps):
                out[b, k] = clim[twin.clim_slot(s, c, k + 1)] + a[k] * anom
        out[:, :, 1] = np.maximum(out[:, :, 1], 0)
        return out
    return f


# ---- forecast skill (45 points) --------------------------------------------------------------------------
def _ss(p, t, pe, cl) -> float:
    return float(np.mean(S.skill_vs_best_ref(p, t, pe, cl)["ss"]))


def _csi_ss(p, t, pe, cl) -> float:
    return S.csi_skill(S.csi(p[..., 1, :, :], t[..., 1, :, :]), S.csi(pe[..., 1, :, :], t[..., 1, :, :]),
                       S.csi(cl[..., 1, :, :], t[..., 1, :, :]))


def forecast_scores(f, X, days, idx, clim, pred=None) -> dict:
    """Skill and CSI skill under readings A, B, C, the day-4 gate and anti-blur, for starts idx (clim = the grader's)."""
    pred = run(f, X, days, idx) if pred is None else pred
    P = pred[:, LEADS - 1]
    T = np.stack([X[idx + lead] for lead in LEADS], 1)
    PE = np.repeat(X[idx][:, None], len(LEADS), 1)
    CL = np.stack([clim[[train.slot(days[i + lead]) for i in idx]] for lead in LEADS], 1)
    W, nl = len(idx), len(LEADS)

    b_ss = np.array([[_ss(P[w, j], T[w, j], PE[w, j], CL[w, j]) for j in range(nl)] for w in range(W)])
    b_csi = np.array([[_csi_ss(P[w, j], T[w, j], PE[w, j], CL[w, j]) for j in range(nl)] for w in range(W)])
    a_ss = np.array([_ss(P[w], T[w], PE[w], CL[w]) for w in range(W)])
    a_csi = np.array([_csi_ss(P[w], T[w], PE[w], CL[w]) for w in range(W)])
    c_ss = np.array([_ss(P[:, j], T[:, j], PE[:, j], CL[:, j]) for j in range(nl)])
    c_csi = np.array([_csi_ss(P[:, j], T[:, j], PE[:, j], CL[:, j]) for j in range(nl)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b_csi_w = np.nanmean(b_csi, axis=1)                # all-NaN rows (no heavy rain anywhere) stay NaN and are skipped
    out = {
        "A": {"mean": float(a_ss.mean()), "csi": float(np.nanmean(a_csi)), **S.forecast_points(list(a_ss), list(a_csi))},
        "B": {"mean": float(b_ss.mean()), "by_lead": b_ss.mean(0).tolist(), "csi": float(np.nanmean(b_csi)),
              "csi_by_lead": np.nanmean(b_csi, 0).tolist(), **S.forecast_points(list(b_ss.mean(1)), list(b_csi_w))},
        "C": {"mean": float(c_ss.mean()), "by_lead": c_ss.tolist(), "csi": float(np.nanmean(c_csi)),
              "csi_by_lead": c_csi.tolist(), **S.forecast_points([float(c_ss.mean())], [float(np.nanmean(c_csi))])},
    }
    j4 = list(LEADS).index(4)
    out["day4_vs_persistence"] = float(np.mean([np.mean(S.skill_vs_best_ref(P[w, j4], T[w, j4], PE[w, j4], PE[w, j4])["ss"])
                                                for w in range(W)]))
    ratios = np.array([S.variance_ratio(pred[w, 7], X[idx[w] + 8]) for w in range(W)])
    out["blur_ratio_day8"] = np.nanmean(ratios, 0).tolist()
    out["blur_points"] = S.anti_blur_points(ratios)
    return out


# ---- dynamics (25 points) and the extra shock of R-6.8 -----------------------------------------------------
def stats_std() -> np.ndarray:
    """Per-channel std from the kit's stats.json: the scale the grader uses for the shock checks."""
    import json
    return np.array(json.loads((train.DATA.parent / "stats.json").read_text())["std"], "f4")


def _arabian_heat(last, std):
    d = np.zeros_like(last); d[0][C.box_mask(C.BOX_ARABIAN_SEA)] = 2.5   # +2.5 C over the Arabian Sea box
    return d


SHOCKS = {   # name: (box, delta builder); the first two are the grader's, the third is R-6.8
    "warm_ocean": (C.BOX_BAY_OF_BENGAL, lambda last, std: S.make_shock("warm_ocean", last, std)),
    "monsoon_stall": (C.BOX_ARABIAN_SEA, lambda last, std: S.make_shock("monsoon_stall", last, std)),
    "arabian_heat": (C.BOX_ARABIAN_SEA, _arabian_heat),
}


def dynamics_scores(f, X, days, idx, std) -> dict:
    """14-step validity, null-shock equality and the four causality checks per shock, as pass rates over windows."""
    out14 = run(f, X, days, idx, steps=C.MAX_STEPS)
    base = run(f, X, days, idx)
    null = run(f, X, days, idx, shock=lambda last: np.zeros_like(last))
    r = {"valid_14": float(np.mean([not S.validity(o, C.MAX_STEPS) for o in out14])),
         "null_ok": float(np.mean([S.null_consistent(a, b, std) for a, b in zip(base, null)])), "shocks": {}}
    checks = {}
    for name, (box, make) in SHOCKS.items():
        shocked = run(f, X, days, idx, shock=lambda last, make=make: make(last, std))
        checks[name] = [S.causality_checks(b, s, box, std) for b, s in zip(base, shocked)]
        z8 = np.abs((shocked - base)[:, 7] / std[None, :, None, None]).max(axis=(1, 2, 3))
        r["shocks"][name] = {k: float(np.mean([c[k] for c in checks[name]]))
                             for k in ("nonzero", "localized", "propagates", "stable", "near_frac_day1", "far_frac_max")}
        r["shocks"][name]["visible_day8"] = float(np.mean(z8 >= S.NONZERO_MIN))
    r["causality_points"] = float(np.mean([S.causality_points([w, m], bool(n)) for w, m, n in
                                           zip(checks["warm_ocean"], checks["monsoon_stall"],
                                               [S.null_consistent(a, b, std) for a, b in zip(base, null)])]))
    return r


def events(X, days, idx, clim, static, n: int = 15) -> dict:
    """Windows like the hidden test's five, picked from the truth of the 8 forecast days, at least 5 days apart."""
    land, sea = static[0] > 0.5, static[0] < 0.5
    nw = (C.LAT[:, None] >= 28) & (C.LON[None, :] <= 80)
    month = np.array([days[i].month for i in idx]); sl = np.array([train.slot(days[i]) for i in idx])
    fut = lambda i: X[i + 1:i + 9]
    anom_t = lambda i: (fut(i)[:, 0] - clim[[train.slot(days[i + k]) for k in range(1, 9)], 0])
    crit = {
        "heavy_rain": (np.isin(month, [6, 7, 8, 9]), lambda i: fut(i)[:, 1][:, land].mean()),
        "cyclone": (np.isin(month, [4, 5, 6, 10, 11, 12]), lambda i: -fut(i)[:, 4][:, sea].min()),
        "heatwave": (np.isin(month, [3, 4, 5, 6]), lambda i: anom_t(i)[:, land].mean(axis=1).max()),
        "winter_disturbance": (np.isin(month, [12, 1, 2, 3]), lambda i: fut(i)[:, 1][:, nw].mean()),
        "monsoon_onset": ((sl >= 145) & (sl <= 170), lambda i: fut(i)[:, 1][:, land].mean()),
    }
    out = {}
    for name, (ok, score) in crit.items():
        cand = sorted(idx[ok], key=score, reverse=True); pick = []
        for i in cand:
            if all(abs(i - j) >= 5 for j in pick):
                pick.append(i)
            if len(pick) == n:
                break
        out[name] = np.array(pick)
    return out


# ---- command line ----------------------------------------------------------------------------------------
def summary(name: str, r: dict) -> str:
    f = lambda v: " ".join(f"{x:+.3f}" for x in v)
    return (f"{name}\n"
            f"  A per window, leads pooled : skill {r['A']['mean']:+.4f}  csi {r['A']['csi']:+.3f}  points {r['A']['points']:.1f}/45\n"
            f"  B per window, per lead     : skill {r['B']['mean']:+.4f} [{f(r['B']['by_lead'])}]  csi {r['B']['csi']:+.3f} "
            f"[{f(r['B']['csi_by_lead'])}]  points {r['B']['points']:.1f}/45\n"
            f"  C all windows pooled       : skill {r['C']['mean']:+.4f} [{f(r['C']['by_lead'])}]  csi {r['C']['csi']:+.3f} "
            f"[{f(r['C']['csi_by_lead'])}]  points {r['C']['points']:.1f}/45\n"
            f"  day 4 vs persistence {r['day4_vs_persistence']:+.3f} | day-8 variance ratio [{f(r['blur_ratio_day8'])}] "
            f"blur points {r['blur_points']:.1f}/10")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", nargs="*", default=[str(twin.WEIGHTS)])
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019])
    ap.add_argument("--baseline", action="store_true", help="also score damped persistence fitted on 2005-2017")
    ap.add_argument("--dynamics", action="store_true", help="validity, null shock and the three shocks on every 12th window")
    ap.add_argument("--events", action="store_true", help="skill on windows like the five hidden ones")
    a = ap.parse_args(argv)
    X, days, static = train.load()
    clim = train.climatology(X, days)                       # the grader's climatology: all of 2005-2021
    idx = starts(days, a.years)
    print(f"{len(idx)} windows in {a.years}")
    if a.baseline:
        f = damped_persistence(X, days, clim, starts(days, range(2005, 2018)))
        print(summary("damped persistence", forecast_scores(f, X, days, idx, clim)))
    fs = {} if not a.baseline else {"damped persistence": damped_persistence(X, days, clim, starts(days, range(2005, 2018)))}
    fs.update({w: mlx_twin.load_forecaster(w) for w in a.weights})     # MLX on the GPU, float32
    for name, f in fs.items():
        if name != "damped persistence":
            print(summary(name, forecast_scores(f, X, days, idx, clim)))
        if a.dynamics:
            d = dynamics_scores(f, X, days, idx[::12], stats_std())
            print(f"  dynamics on {len(idx[::12])} windows: valid 14 steps {d['valid_14']:.2f}, null equal {d['null_ok']:.2f}, "
                  f"causality points {d['causality_points']:.1f}/15")
            for k, v in d["shocks"].items():
                print(f"    {k:14s} " + " ".join(f"{m} {x:.2f}" for m, x in v.items()))
        if a.events:
            for ev, ei in events(X, days, idx, clim, static).items():
                r = forecast_scores(f, X, days, ei, clim)
                print(f"    event {ev:19s} A {r['A']['mean']:+.3f}  B {r['B']['mean']:+.3f}  csi A {r['A']['csi']:+.3f} B {r['B']['csi']:+.3f}")


if __name__ == "__main__":
    main()
