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

Status: scaffolding (P1 — load + step DIAMOND headless on Modal).
