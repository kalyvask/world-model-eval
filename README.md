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

**Result (DIAMOND-Breakout):** across an epsilon-greedy policy spectrum
(eps 0 = good → 1 = random), imagined return ranks policies in significant
agreement with real return — **Spearman 0.78 (p = 0.04), Pearson 0.81** (n = 7
policies, 16 imagined rollouts × horizon 120 vs 5 real episodes × 300). The
signal is in the *ranking*, not the magnitude (imagined returns are compressed
on sparse Breakout reward), but the extremes separate cleanly. The world model
is a usable cheap policy *evaluator*.

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

Status: DreamEval E2 done (imagined return ranks policies, Spearman 0.78). Next:
strengthen with more rollouts + horizon-vs-correlation analysis (E3).
