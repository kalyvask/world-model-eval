# world-model-eval

Experiments on an open generative **world model** (DIAMOND — a diffusion world
model that generates a playable Atari environment frame by frame), run headless
on Modal. Sibling to [`inside-the-agent`](../inside-the-agent).

## The fidelity horizon: near-perfect 1-step, short and policy-dependent

Free-running DIAMOND's dream from a real trajectory's context under the same
actions as the real env: the **one-step prediction error is 0.0020** (≈ the
natural consecutive-frame change of 0.0022 — near-perfect), but free-running
**half-decorrelates within tens of steps, and how fast depends on the policy**:
~30 steps under the agent's own greedy policy (68% bootstrap CI [30, 34]) vs
**~10 under random actions** ([7, 14]), non-overlapping over 24 trajectories. So
the horizon is real but it's an *in-distribution* number, not a constant.

The model decodes instantaneous state cleanly but neither sustains nor controls it:

- **Decode works** (ball-position R²≈0.73, leakage-corrected time-split): one
  faithful frame is enough; 1-step fidelity is excellent.
- **Multi-step policy evaluation fails**: it needs *sustained* fidelity, but the
  dream decorrelates within ~30 steps (DreamEval's imagined reward saturates by
  ~20–30 — the same horizon).
- **Steering fails too** (decode ≫ steer): the decoded ball direction moves the
  ball no more than a matched-norm random direction — bug-checked, it holds even
  injected post-normalization. See [`docs/steering_study.md`](docs/steering_study.md).

Code: `app_eval.py::fidelity`.

### Does the horizon generalize across architecture and scale? No.

Running the same measurement on **IRIS** (a VQ-VAE + Transformer world model,
several times larger than DIAMOND's 4.4M diffusion core) under a **matched
random-action protocol**: DIAMOND half-decorrelates by step **10** [7, 14], but
IRIS only reaches a *sustained* half-decorrelation at step **58** [21, 60] (10%
of bootstrap resamples never cross within 60 steps). Not identical.

Two things make a "universal ~30-step constant" untenable: the horizon is
policy-dependent (above), and **L1 frame divergence isn't comparable across the
two frame types** — IRIS's discrete VQ-VAE frames stay crisp and low-L1 even
when semantically wrong, while DIAMOND's continuous diffusion frames blur and
drift. Honest read: each model has its own policy-dependent fidelity horizon;
there is no shared ~30-step number. Code: `app_iris.py::fidelity`. (Both are
small/medium Atari models.)

![Fidelity horizon is policy- and model-dependent, not a universal ~30 steps](artifacts/fidelity_horizon.png)

*Each curve normalized to its own floor-to-ceiling (0 = 1-step error, 1 = that
run's decorrelated reference). Under a matched random-action protocol DIAMOND
(dashed) crosses half-decorrelation at ~10 and IRIS (red) only at the noisy edge
of the 60-step window; DIAMOND's "~30" (solid) holds only under its own greedy
policy. The earlier "DIAMOND ~30 ≈ IRIS ~31" compared mismatched policies with a
noise-sensitive first-touch crossing.*

---

The same DIAMOND-on-Modal infrastructure powers the applied study below,
explained by the fidelity horizon above:

### DreamEval — world model as a cheap policy evaluator

**Can the world model's *imagined* return rank policies the way the real env
does?** Score a spectrum of policies by imagined return (rolled out inside the
world model, scored by its reward model) and measure the correlation with real
return in the actual Atari env. The canonical applied use-case for world models
(policy evaluation without expensive real trials), validated on Atari.

**Result (DIAMOND-Breakout): imagined return is *not* a reliable policy
evaluator here.** Across an epsilon-greedy spectrum (eps 0 = good → 1 = random),
real return falls cleanly (8.5 → 1.1) but imagined return stays **flat at ~0.4
regardless of policy quality** (random ≈ good). The rank correlation is sample-
fragile: at 7 coarse policies it looks promising (Spearman 0.78, p=0.04) but
that is driven by the good-vs-random extremes + small n — on a **finer 13-policy
grid it collapses to Spearman 0.22 (p=0.47)**. The dream's imagined reward
**saturates by ~20–30 steps** (the rollout goes inert after the ball is lost),
so it captures only the coarsest good-vs-random distinction, not a usable
ranking.

![Imagined vs real return across the 13-policy spectrum; imagined return is flat](artifacts/dreameval_scatter.png)

*Real return falls cleanly from 8.6 to 1.1 across the epsilon spectrum (color),
but imagined return stays flat at ~0.4 regardless of policy quality. Spearman
0.22 (p=0.47) at 13 policies: no usable ranking signal.*

Takeaway: a small open world model decodes state well (ball-position R²≈0.73)
but its imagined rollouts are too low-fidelity to rank policies at fine
resolution. It **decodes but does not faithfully *simulate***.

Code: `modal_deploy/app_eval.py` · plan: [`BUILD_PLAN.md`](BUILD_PLAN.md)

---

Infra: DIAMOND on Modal L40S — hydra load + `eval` resolver,
`make_atari_env`, seeded `WorldModelEnv` (collect real ALE frames → imagine
under the agent's policy). Tiny 4.4M-param denoiser; cheap to run.

Status: DreamEval E1–E3 done. Strengthening (more rollouts, then a 13-policy
grid) reversed the n=7 positive: imagined return does not reliably rank policies
(Spearman 0.22, p=0.47 at 13 policies). Honest negative; decode ≫ simulate.

Headline numbers are bug-checked: decode R² is leakage-corrected (time-ordered
split, not random), the fidelity crossing uses a sustained metric + bootstrap CI
over trajectories, and the cross-architecture comparison uses a matched
action protocol — which removed an earlier, confounded "~30 ≈ ~31" claim.
