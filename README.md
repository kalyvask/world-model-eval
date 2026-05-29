# world-model-eval

Experiments on an open generative **world model** (DIAMOND — a diffusion world
model that generates a playable Atari environment frame by frame), run headless
on Modal. Sibling to [`inside-the-agent`](../inside-the-agent).

Two studies share the same DIAMOND-on-Modal infrastructure:

### 1. DreamEval — world model as a cheap policy evaluator (current)

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

Takeaway: a small open world model decodes state well but its imagined rollouts
are too low-fidelity to rank policies at fine resolution. Combined with the
steering study below, the repo is a two-sided study of the **limits of a small
world model: it decodes (R²=0.89) but does not faithfully *simulate*** — neither
clean steering nor reliable policy evaluation holds.

Code: `modal_deploy/app_eval.py` · plan: [`BUILD_PLAN.md`](BUILD_PLAN.md)

### 2. Steering study — decode ≫ steer (done)

Interpretability + steering of the world model. **Finding:** game state is
linearly **decodable** from the UNet activations (ball position R² = 0.89) but
**not cleanly steerable** — adding the probe/contrastive direction moves the
ball about as much as a matched-norm *random* direction does. The same
decode ≫ steer pattern holds in the sibling `inside-the-agent` (LLM agent).

Code: `modal_deploy/app_diamond.py` · plan + results:
[`docs/steering_study.md`](docs/steering_study.md)

---

Infra (both studies): DIAMOND on Modal L40S — hydra load + `eval` resolver,
`make_atari_env`, seeded `WorldModelEnv` (collect real ALE frames → imagine
under the agent's policy). Tiny 4.4M-param denoiser; cheap to run.

Status: DreamEval E1–E3 done. Strengthening (more rollouts, then a 13-policy
grid) reversed the n=7 positive: imagined return does not reliably rank policies
(Spearman 0.22, p=0.47 at 13 policies). Honest negative; decode ≫ simulate.
