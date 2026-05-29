# world-model-steering

Interpretability and live steering of an open **world model** — a model that
generates a playable environment frame by frame.

Sibling to [`inside-the-agent`](../inside-the-agent) (which steered an LLM
browser agent). Same methodology — capture internal activations, find a
concept direction, hook a layer, add/clamp it, observe — pointed at a
generative world model instead of a language model.

Targets: **DIAMOND** (diffusion world model, playable Atari) first, then
**Oasis** (playable Minecraft) for the visual demo. Directions come from linear
probes / contrastive activations, not SAEs (linear probes edit better and no
pretrained SAE exists for these models).

See [`BUILD_PLAN.md`](BUILD_PLAN.md) for the full plan, method rationale, prior
art, and the grounded DIAMOND API notes.

## Findings (DIAMOND-Breakout)

Pipeline (all headless on Modal, L40S): load DIAMOND → seed from real ALE
frames → imagine under the agent's own policy → capture UNet
(`inner_model.unet`) activations → CV-label the ball by frame differencing →
probe / steer.

1. **Game state is linearly decodable from the world model's activations.** A
   ridge probe on the pooled UNet activation predicts ball position on a
   held-out split: **ball_x R² = 0.89, ball_y R² = 0.76** (n=373, label std
   ~14/8 px). This replicates the known result that DIAMOND carries
   approximately linear game-state representations.

2. **The decoded direction does not give clean steering.** Adding the probe
   direction (or a difference-of-means "ball-right minus ball-left" vector)
   back into the activation *moves* the ball, but a **matched-norm random
   direction moves it about as much**, and the effect is non-monotonic past a
   small magnitude. There is weak sign-dependent control at moderate magnitude
   (+α and −α push opposite ways) but the signal-to-noise vs the random
   control is poor.

**Takeaway: decode ≫ steer.** The same pattern holds in the sibling
[`inside-the-agent`](../inside-the-agent) (an LLM browser agent): interpretable
directions read concepts out cleanly, but adding them back perturbs more than
it controls. Clean concept-steering of a generative world model by linear
activation addition is materially harder than decoding it.

Status: P1–P4 complete (decode result + steering characterization). P5 (live
HUD) and P6 (Oasis port) are optional given the steer finding.
