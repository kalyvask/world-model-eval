"""
Generate the two headline figures for world-model-eval:
  1. fidelity_horizon.png  — DIAMOND vs IRIS divergence-vs-step, each normalized
     to its own floor->ceiling, showing both decorrelate at ~30 steps.
  2. dreameval_scatter.png  — imagined vs real return across the 13-policy
     epsilon spectrum, showing imagined return is flat (no ranking signal).

Reads the captured fidelity JSON from data/{diamond,iris}_fid_raw.txt.
Run locally:  python artifacts/make_charts.py
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"


def extract_json(path: Path) -> dict:
    """Pull the json.dumps(indent=2) block out of a captured modal-run log."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    # the printed block starts with a line that is exactly "{" and ends with "}"
    m = re.search(r"^\{\s*$.*?^\}\s*$", text, re.DOTALL | re.MULTILINE)
    if not m:
        raise ValueError(f"no JSON block found in {path}")
    return json.loads(m.group(0))


# ---- DreamEval 13-policy data (captured from the run; see README) ----
EVAL_EPS = [0.0, 0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.583, 0.667, 0.75, 0.833, 0.917, 1.0]
EVAL_REAL = [8.5, 8.6, 7.8, 7.5, 6.6, 5.8, 5.9, 3.8, 3.0, 4.2, 4.0, 1.8, 1.1]
EVAL_IMAG = [0.5, 0.375, 0.417, 0.375, 0.5, 0.25, 0.5, 0.375, 0.417, 0.417, 0.292, 0.333, 0.458]
EVAL_SPEARMAN = 0.22

# ---- Fidelity-horizon data (captured from the runs; see data/*_fid_raw.txt) ----
# Inlined so this script is self-contained on a fresh clone (the raw run logs
# under data/ are gitignored). If a raw log is present it takes precedence.
DIAMOND_FID = {
    "model": "DIAMOND", "one_step_error": 0.002, "decorrelated_ceiling": 0.0077,
    "half_decorrelation_step": 30,
    "divergence_curve": [
        0.002, 0.002, 0.0029, 0.003, 0.0034, 0.0034, 0.0036, 0.0036, 0.0037, 0.0037,
        0.0037, 0.0038, 0.0038, 0.0039, 0.0041, 0.0042, 0.0042, 0.0042, 0.0043, 0.0043,
        0.0044, 0.0047, 0.0046, 0.0047, 0.0047, 0.0047, 0.0047, 0.0047, 0.0047, 0.0062,
        0.0062, 0.0061, 0.0063, 0.0062, 0.006, 0.0061, 0.0065, 0.0056, 0.0056, 0.0056,
        0.0056, 0.0056, 0.0057, 0.0057, 0.0057, 0.0057, 0.0057, 0.0058, 0.0057, 0.0058,
        0.0058, 0.0058, 0.0058, 0.0058, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059,
    ],
}
IRIS_FID = {
    "model": "IRIS", "one_step_error": 0.0006, "decorrelated_ceiling": 0.0026,
    "half_decorrelation_step": 31,
    "divergence_curve": [
        0.0006, 0.0005, 0.0004, 0.0004, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
        0.0004, 0.0005, 0.0005, 0.0004, 0.0004, 0.0004, 0.0008, 0.0009, 0.0009, 0.0013,
        0.0013, 0.0014, 0.0013, 0.0014, 0.0014, 0.0013, 0.0013, 0.0014, 0.0014, 0.0014,
        0.0016, 0.0015, 0.0013, 0.0014, 0.0014, 0.0013, 0.0012, 0.0012, 0.0012, 0.0013,
        0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0014, 0.0012, 0.0011, 0.0011,
        0.0011, 0.0011, 0.0011, 0.0011, 0.0013, 0.0014, 0.0013, 0.0016, 0.0017, 0.0019,
    ],
}


def load_fid(raw_name: str, fallback: dict) -> dict:
    """Prefer a captured raw run log if present, else the inlined fallback."""
    path = DATA / raw_name
    if path.exists():
        try:
            return extract_json(path)
        except ValueError:
            pass
    return fallback


def fidelity_chart():
    d = load_fid("diamond_fid_raw.txt", DIAMOND_FID)
    i = load_fid("iris_fid_raw.txt", IRIS_FID)

    def norm(curve, floor, ceiling):
        span = max(ceiling - floor, 1e-9)
        return [(v - floor) / span for v in curve]

    dn = norm(d["divergence_curve"], d["one_step_error"], d["decorrelated_ceiling"])
    ihn = norm(i["divergence_curve"], i["one_step_error"], i["decorrelated_ceiling"])
    xd = list(range(1, len(dn) + 1))
    xi = list(range(1, len(ihn) + 1))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xd, dn, color="#2563eb", lw=2, label=f"DIAMOND (diffusion) — half-decorr @ {d['half_decorrelation_step']}")
    ax.plot(xi, ihn, color="#dc2626", lw=2, label=f"IRIS (transformer) — half-decorr @ {i['half_decorrelation_step']}")
    ax.axhline(0.5, color="#6b7280", ls="--", lw=1)
    ax.axvline(30, color="#9ca3af", ls=":", lw=1)
    ax.text(31, 0.6, "~30 steps", color="#6b7280", fontsize=9)
    ax.set_xlabel("free-running dream step")
    ax.set_ylabel("divergence from reality\n(0 = 1-step error, 1 = decorrelated)")
    ax.set_title("World-model fidelity horizon: ~30 steps across architecture + scale")
    ax.set_ylim(-0.05, 1.15)
    ax.legend(loc="upper left", fontsize=9, frameon=False)
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
    ax.text(0.04, 0.96, "real return spans 1.1–8.6;\nimagined return is flat ~0.4",
            transform=ax.transAxes, va="top", fontsize=9, color="#374151")
    fig.tight_layout()
    out = HERE / "dreameval_scatter.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    fidelity_chart()
    eval_chart()
