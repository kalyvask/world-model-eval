# world-model-steering — build plan

Interpretability + live steering of an open **world model** (a model that
generates a playable environment frame-by-frame). Sibling project to
`inside-the-agent` (which steered an LLM browser agent); this points the same
"capture activations → find a direction → hook a layer → add/clamp → observe"
methodology at a generative world model.

## Thesis

An **interactive interpretability + steering cockpit for a world model**: find
concept directions in the model's internal activations, then steer them *live*
and watch the simulated world change (clamp "ball-present" → the ball vanishes;
amplify a motion direction → the world drifts). The contribution is the live/
interactive steering tool + a **characterization of which concepts steer
cleanly vs which break the simulation**.

## Method decision: linear probes / contrastive directions, NOT SAEs

For *editing/steering* the literature is blunt: linear-probe directions hit
~88% edit success vs SAE latents at ~41%
([Are Sparse Autoencoders Useful?](https://arxiv.org/pdf/2502.16681)). And
there is no pretrained SAE for these world models anyway. So we get directions
by:
- **Contrastive / difference-of-means**: mean activation over states with the
  concept (ball present) minus mean over states without it → steering vector.
- **Linear probes**: train a linear classifier to detect a labeled concept
  (ball x-position bucket, paddle position, brick present, score digit) from a
  captured activation; the probe weight is the steering direction.

Steering = register a forward hook on a UNet layer and add `alpha * direction`
during denoising (mirrors `inside-the-agent`'s residual-stream hook).

## Prior art (cite, differentiate from)

[What Do World Models Learn in RL?](https://arxiv.org/pdf/2603.21546) already
probes DIAMOND + IRIS on Breakout/Pong, finds ~linear game-state
representations, and does *offline* causal interventions. We differentiate by:
(1) **live/interactive** steering, not offline analysis; (2) a steerability
characterization (which concepts hold vs break the sim); (3) extension to a
richer world (Oasis-Minecraft) the paper didn't cover.

## Target models (decided: DIAMOND first → Oasis upgrade)

- **DIAMOND** (NeurIPS 2024, https://github.com/eloialonso/diamond): diffusion
  world model, open weights, playable Atari (Breakout/Pong) + standalone CS:GO.
  Validated linear representations → low-risk direction-finding. Start here.
- **Oasis 500M** (https://github.com/etched-ai/open-oasis, HF Etched/oasis-500m):
  playable Minecraft, DiT + ViT autoencoder, 20fps. Flashier, novel, heavier.
  Port here once the pipeline works on DIAMOND.

## DIAMOND API notes (grounded from source)

- Load: `hf_hub_download(repo_id="eloialonso/diamond", filename="atari_100k/models/<Game>.pt")`;
  hydra `compose(config_name="trainer")`; `agent = Agent(instantiate(cfg.agent, num_actions=N))`;
  `agent.load(path_ckpt)`.
- World model core: `agent.denoiser` (UNet in `src/models/diffusion/inner_model.py`) = hook point.
- Headless step: `WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_cfg)` →
  `reset()` → `step(act: LongTensor (n,))` → `(next_obs, rew, term, trunc, info)`.
  Internally `self.sampler.sample(obs_buffer, act_buffer)`.
- Seeding: WorldModelEnv needs initial real frames (a data loader) to condition
  the first prediction. The `--pretrained` play path downloads seed data; we
  need the equivalent (download a seed segment, or seed from a real ALE env —
  atari deps are needed anyway for `num_actions`).

## Results (P1-P4 complete)

- P1 ✅ DIAMOND Breakout loads headless on Modal (4.41M-param denoiser, L40S).
- P2 ✅ seed (real ALE frames) + imagine-step + capture UNet acts (1,64,64,64).
- P3 ✅ **ball position linearly decodable**: ridge probe on pooled UNet
  activation, ball_x R²=0.89 / ball_y R²=0.76 (held-out). Driving imagination
  with the agent's policy + FIRE burn-in was needed for dynamic rollouts;
  frame-diff CV isolates the moving ball from static bricks.
- P4 ✅ **decode ≫ steer (bug-checked, holds)**: the first pass added the
  direction to the UNet output, but `InnerModel.forward` runs `norm_out`
  (GroupNorm) on the next line — it subtracts the per-group mean and rescales,
  washing an additive offset away. Fixed by injecting on `norm_out`'s **output**
  (post-GroupNorm), with the perturbation scaled to a fraction of the activation
  norm per denoise step. Re-run (signed frac sweep + matched-norm random
  control): the finding **survives the fix**. The semantic ball_x direction does
  not move the measured ball more than a random direction of the same norm —
  random frac=0.25 shifted ball_x by −20.6 px vs the semantic direction's
  −6.4 px, and +frac barely moved it. So decode ≫ steer is real, not an artifact
  of the original (pre-norm) hook placement. Caveat: at magnitudes large enough
  to change the output the frame degrades, so this is evidence against *clean*
  steerability, not a precise causal null. Code: `app_diamond.py::steer_ball`.
- P5 / P6: optional given the P4 finding (a "steer the dream" cockpit is only
  compelling if steering is clean; Oasis port would test if decode≫steer holds
  on a richer model).

## Phased plan

- **P1 — Load + step headless on Modal.** Clone DIAMOND in the image, install
  deps + Atari ROMs, download a pretrained checkpoint (start: Breakout), build
  WorldModelEnv, reset + step N frames with a fixed action, return frames.
  De-risks the whole integration. ← START HERE.
- **P2 — Capture activations.** Register a forward hook on an `inner_model`
  UNet layer; capture the activation at the last denoising step per generated
  frame. Confirm shapes + that they vary with game state.
- **P3 — Label + probe.** Roll out the world model (or a real ALE env) with
  ground-truth state (ball xy, paddle x, bricks, score). Train linear probes /
  compute contrastive directions per concept. Report probe accuracy (sanity:
  game state is linearly decodable).
- **P4 — Steer.** Add `alpha * direction` in the hook during generation; sweep
  alpha; classify outcome (concept changed cleanly vs sim broke). The "watch
  the ball vanish" result. Controls: random direction, wrong-sign (mirror
  inside-the-agent's control discipline).
- **P5 — Live HUD.** Stream generated frames + sliders that set per-concept
  alpha live (reuse inside-the-agent's HUD/ws pattern + the I1 live-steering
  idea). The demo-day "steer the dream" cockpit.
- **P6 — Oasis port.** Repeat P2-P5 on Oasis-Minecraft for the flashy demo.

## Repo layout

```
modal_deploy/app_diamond.py   Modal server: load + step DIAMOND, capture/steer
probes/                       linear probe + contrastive direction training
steering/                     direction application + alpha sweep + controls
hud/                          live steering cockpit (later)
data/                         captured activations, probes, rollouts
```

## Constraints

- Local scaffold only; do NOT create/push the GitHub repo until there's a real
  result (same "no push until findings" discipline as inside-the-agent).
- GPU on Modal (sponsored by AMP for CS153). DIAMOND Atari is small (L40S fine).
