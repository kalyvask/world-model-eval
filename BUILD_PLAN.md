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

- **E1 — load + both return functions** ✅ (`app_eval.py`): pipeline runs;
  real env ranks correctly (good 2.0 > random 0.0). FIRE burn-in launches the
  ball in both real and dream.
- **E2 — the experiment** ✅: 7 coarse policies → Spearman 0.78 (p=0.04). Looked
  promising, but see E3.
- **E3 — strengthen** ✅ (and it reversed the result): more rollouts (n_imag=32)
  gave 0.71 (p=0.07); a **13-policy grid collapsed it to Spearman 0.22 (p=0.47)**.
  Imagined return is flat ~0.4 across the spectrum (random ≈ good); the dream's
  reward **saturates by ~20–30 steps** (flat horizon curve). **Honest verdict:
  imagined return is NOT a reliable policy evaluator on DIAMOND-Breakout** — the
  n=7 positive was small-n / good-vs-random-extremes fragility.
- Consolidated: a small world model **decodes** state (R²=0.89, steering study)
  but does not faithfully **simulate** — neither clean steering nor reliable
  policy eval holds. Decode ≫ simulate.
- **E4 — writeup + chart.**

## Honesty guardrails

- No cost-savings claim on Atari; frame it as validating that imagined return
  ranks policies like real return.
- Headline Spearman (rank), since reward scales differ between dream and real.
- Control: the random policy (eps=1) must land at the bottom of BOTH rankings.
