# DreamEval — build plan

**Can a small world model cheaply predict which policy is better — without
running the real environment?**

The canonical applied use-case for world models (the "DreamDojo / policy
evaluation without expensive real trials" thesis), tested concretely on
DIAMOND-Breakout: score a spectrum of policies by their **imagined return**
(rolled out inside the world model, scored by its reward model) and check how
well that ranks them against their **real return** in the actual Atari env.

This repo also contains a prior **steering study** (decode ≫ steer); see
[`docs/steering_study.md`](docs/steering_study.md). Both reuse the same
DIAMOND-on-Modal infrastructure (`modal_deploy/app_diamond.py` for steering,
`modal_deploy/app_eval.py` for DreamEval).

## The claim to test

If `imagined_return(policy)` correlates strongly with `real_return(policy)`
across policies, the world model is a usable cheap evaluator. The payoff is in
domains where real rollouts are expensive (robots); **Atari is the validation
testbed** — real rollouts are cheap here, so we use them as ground truth. We do
**not** claim cost savings on Atari itself; we test whether imagined return
*ranks* policies the way real return does.

## Method

- **Policy spectrum (quality knob):** epsilon-greedy on the pretrained DIAMOND
  actor-critic, epsilon in {0, .1, .25, .4, .6, .8, 1.0}. eps=0 = good policy,
  eps=1 = uniform random. A monotonic quality range, no training required.
- **Real return:** run each policy in real ALE Breakout (`make_atari_env`),
  average undiscounted return.
- **Imagined return:** run each policy inside `WorldModelEnv`, sum the
  reward-model rewards, average. (FIRE burn-in to launch the ball, applied in
  both real and imagined for consistency.)
- **Result:** Spearman (headline — ranking is what an evaluator needs) +
  Pearson, across the spectrum. Scatter (imagined vs real) is the headline plot.
- **Horizon analysis:** correlation vs imagined horizon H — how far can you
  trust the dream for ranking?

## Phases

- **E1 — load + both return functions** (`app_eval.py`): reuse the proven
  DIAMOND load; `smoke` = good (eps0) vs random (eps1) in real + dream; good
  should beat random in both, else add/adjust FIRE burn-in. ← here.
- **E2 — the experiment:** sweep the epsilon spectrum; Spearman/Pearson + scatter.
- **E3 — horizon + cost:** correlation vs imagined H; step-count/wall-clock of
  imagined vs real eval.
- **E4 — writeup + chart.**

## Honesty guardrails

- No cost-savings claim on Atari; frame it as validating that imagined return
  ranks policies like real return.
- Headline Spearman (rank), since reward scales differ between dream and real.
- Control: the random policy (eps=1) must land at the bottom of BOTH rankings.
