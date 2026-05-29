"""
Generate the two headline figures for world-model-eval:
  1. fidelity_horizon.png  — free-running divergence vs step for three runs,
     each normalized to its own floor->ceiling: DIAMOND under its greedy policy,
     DIAMOND under random actions, and IRIS under random actions. Shows the
     fidelity horizon is policy- and model-dependent (NOT a universal ~30 steps).
  2. dreameval_scatter.png — imagined vs real return across the 13-policy
     epsilon spectrum, showing imagined return is flat (no ranking signal).

Numbers are inlined from the captured runs (raw logs under data/ are gitignored;
half-decorrelation steps use the sustained-crossing metric with bootstrap 68% CI
from app_eval.py / app_iris.py::fidelity). Run locally: python artifacts/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


# ---- DreamEval 13-policy data (captured from the run; see README) ----
EVAL_EPS = [0.0, 0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.583, 0.667, 0.75, 0.833, 0.917, 1.0]
EVAL_REAL = [8.9, 8.8, 8.4, 7.4, 6.1, 6.0, 4.7, 3.7, 3.6, 3.0, 2.2, 1.9, 1.7]
EVAL_IMAG = [0.5, 0.406, 0.375, 0.469, 0.469, 0.344, 0.438, 0.5, 0.438, 0.469, 0.438, 0.562, 0.281]
EVAL_SPEARMAN = 0.01

# ---- Fidelity-horizon runs (captured; sustained crossing + 68% bootstrap CI) ----
# Matched random-action protocol for the cross-model comparison; DIAMOND greedy
# is shown too to expose the policy dependence. "cross" = sustained-crossing step.
FID = {
    "diamond_greedy": {
        "label": "DIAMOND (diffusion), greedy policy", "color": "#2563eb", "ls": "-",
        "floor": 0.0020, "ceiling": 0.0075, "cross": 30, "ci": [30, 34],
        "curve": [
            0.002, 0.002, 0.0029, 0.003, 0.0034, 0.0034, 0.0036, 0.0036, 0.0037, 0.0037,
            0.0037, 0.0038, 0.0038, 0.0039, 0.004, 0.004, 0.0041, 0.0041, 0.0041, 0.0041,
            0.0042, 0.0043, 0.0043, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0044, 0.0051,
            0.0051, 0.0051, 0.0052, 0.0052, 0.0051, 0.0051, 0.0053, 0.0049, 0.0049, 0.0049,
            0.0049, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.0051, 0.0051, 0.0051,
            0.0051, 0.0051, 0.0051, 0.0051, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052, 0.0052,
        ],
    },
    "diamond_random": {
        "label": "DIAMOND (diffusion), random policy", "color": "#2563eb", "ls": "--",
        "floor": 0.0020, "ceiling": 0.0053, "cross": 10, "ci": [7, 14],
        "curve": [
            0.002, 0.002, 0.0029, 0.003, 0.0034, 0.0034, 0.0036, 0.0036, 0.0037, 0.0037,
            0.0038, 0.0038, 0.0038, 0.0039, 0.0039, 0.0041, 0.0042, 0.0043, 0.0044, 0.0043,
            0.0042, 0.0051, 0.0052, 0.0053, 0.0054, 0.0054, 0.0054, 0.0056, 0.0057, 0.0058,
            0.0057, 0.0058, 0.0058, 0.0059, 0.0058, 0.0058, 0.0059, 0.006, 0.0061, 0.0062,
            0.0063, 0.0065, 0.0065, 0.0065, 0.0064, 0.0065, 0.0065, 0.0065, 0.0065, 0.0066,
            0.0066, 0.0069, 0.0067, 0.0068, 0.0068, 0.0068, 0.007, 0.0069, 0.0069, 0.0069,
        ],
    },
    "iris_random": {
        "label": "IRIS (transformer), random policy", "color": "#dc2626", "ls": "-",
        "floor": 0.0005, "ceiling": 0.0026, "cross": 58, "ci": [21, 60],
        "curve": [
            0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0004, 0.0005, 0.0004, 0.0004, 0.0004, 0.0004, 0.0008, 0.0008, 0.0009, 0.0012,
            0.0014, 0.0015, 0.0014, 0.0015, 0.0014, 0.0015, 0.0014, 0.0014, 0.0014, 0.0014,
            0.0016, 0.0016, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013,
            0.0013, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0014, 0.0013, 0.0013, 0.0013,
            0.0012, 0.0012, 0.0012, 0.0013, 0.0013, 0.0014, 0.0014, 0.0017, 0.0017, 0.0019,
        ],
    },
}


# ---- Decoded-state (ball-drift) fidelity: the frame-type-fair metric ----
# Mean px distance between the imagined ball and the true ball, vs free-running
# step. Ceiling = mean distance between random unrelated real ball positions.
# Comparable across architectures (unlike pixel L1). null = no detection.
FID_BALL = {
    "diamond_greedy": {
        "label": "DIAMOND, greedy policy", "color": "#2563eb", "ls": "-",
        "ceiling": 20.23, "cross": None,
        "curve": [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, None, None, None, None, 4.63, 1.46,
                  1.89, 1.99, 2.38, 2.7, 2.96, 3.79, 3.84, 4.21, 4.39, 4.78, 5.57, 5.72,
                  6.22, 5.68, 5.11, 4.37, 3.89, 3.08, 3.08, 3.27, 3.5, 4.31, 5.09, 5.61,
                  6.39, 7.2, None, None, None, 5.32, 9.48, 9.69, 9.51, 8.25, 7.24, 6.45,
                  5.3, 4.77, 4.38, 5.13, 7.35, 7.43, 7.43, 9.01, 8.22, 8.05, 9.44],
    },
    "diamond_random": {
        "label": "DIAMOND, random policy", "color": "#2563eb", "ls": "--",
        "ceiling": 19.7, "cross": 16,
        "curve": [0.0, 0.0, 0.0, 0.0, 0.0, 0.03, 0.0, None, None, None, None, 0.0, 0.25,
                  7.31, 6.79, 10.26, 9.36, 13.79, 16.75, 18.31, 17.38, 16.99, 16.68, 16.47,
                  16.08, 15.69, 16.32, 16.02, 16.66, 17.81, 18.1, 15.93, 15.81, 15.63, 15.1,
                  14.62, 14.36, 17.98, 17.71, 20.04, 19.3, 18.33, 20.22, 19.97, 19.99, 20.12,
                  19.95, 19.83, 19.96, 20.04, 19.78, 20.57, 19.55, 20.67, 20.44, 20.62, 17.8,
                  16.63, 14.64, 14.63],
    },
    "iris_random": {
        "label": "IRIS, random policy", "color": "#dc2626", "ls": "-",
        "ceiling": 13.5, "cross": 20,
        "curve": [0.04, 0.26, 0.1, 0.55, 0.38, 0.0, None, None, None, None, None, 0.17, 0.4,
                  0.46, 0.24, 0.4, 3.94, 3.43, 6.5, 9.36, 11.36, 10.94, 11.28, 13.29, 13.51,
                  14.21, 12.72, 12.31, 13.03, 13.0, 13.4, 13.23, 12.35, 12.65, 12.81, 13.28,
                  12.14, 7.12, 7.77, 7.38, 7.58, 8.68, 10.18, 10.62, 10.78, 12.85, 14.05,
                  14.48, 14.02, 14.85, 14.65, 13.98, 15.86, 17.97, 18.2, 18.49, 18.2, 20.1,
                  19.24, 20.75],
    },
}


def _norm(curve, floor, ceiling):
    span = max(ceiling - floor, 1e-9)
    return [(v - floor) / span for v in curve]


def fidelity_ball_chart():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for key in ["diamond_greedy", "diamond_random", "iris_random"]:
        r = FID_BALL[key]
        yn = [np.nan if v is None else v / r["ceiling"] for v in r["curve"]]
        x = list(range(1, len(yn) + 1))
        cross = r["cross"]
        tag = f"half-decorr @ {cross}" if cross else "never half-decorrelates"
        ax.plot(x, yn, color=r["color"], ls=r["ls"], lw=2, label=f'{r["label"]} — {tag}')
    ax.axhline(0.5, color="#6b7280", ls=":", lw=1)
    ax.text(1.5, 0.53, "half-decorrelated", color="#6b7280", fontsize=8)
    ax.set_xlabel("free-running dream step")
    ax.set_ylabel("imagined-vs-true ball drift\n(0 = perfect, 1 = random-frame distance)")
    ax.set_title("Decoded-state fidelity: open world models track the ball, then lose it off-policy")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = HERE / "fidelity_ball.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def fidelity_chart():
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    for key in ["diamond_greedy", "diamond_random", "iris_random"]:
        r = FID[key]
        yn = _norm(r["curve"], r["floor"], r["ceiling"])
        x = list(range(1, len(yn) + 1))
        lab = f'{r["label"]} — half-decorr @ {r["cross"]} [{r["ci"][0]}-{r["ci"][1]}]'
        ax.plot(x, yn, color=r["color"], ls=r["ls"], lw=2, label=lab)
    ax.axhline(0.5, color="#6b7280", ls=":", lw=1)
    ax.text(1.5, 0.54, "half-decorrelated", color="#6b7280", fontsize=8)
    ax.set_xlabel("free-running dream step")
    ax.set_ylabel("divergence from reality\n(0 = 1-step error, 1 = each run's ceiling)")
    ax.set_title("Fidelity horizon is policy- and model-dependent (not a universal ~30)")
    ax.set_ylim(-0.15, 1.3)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = HERE / "fidelity_horizon.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def eval_chart():
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    sc = ax.scatter(EVAL_REAL, EVAL_IMAG, c=EVAL_EPS, cmap="viridis", s=70, edgecolor="k", linewidth=0.4)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("policy randomness (epsilon)")
    ax.set_xlabel("real return (ALE)")
    ax.set_ylabel("imagined return (world model)")
    ax.set_title(f"Imagined return does not rank policies (Spearman {EVAL_SPEARMAN})")
    ax.set_ylim(0, max(EVAL_IMAG) * 1.5)
    ax.grid(alpha=0.2)
    ax.text(0.04, 0.96, "real return spans 1.7–8.9;\nimagined return is flat ~0.4",
            transform=ax.transAxes, va="top", fontsize=9, color="#374151")
    fig.tight_layout()
    out = HERE / "dreameval_scatter.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    fidelity_ball_chart()
    fidelity_chart()
    eval_chart()
