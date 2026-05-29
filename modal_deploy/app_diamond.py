"""
B1 — DIAMOND Atari diffusion world model on Modal: load, step headless, and
capture UNet activations (P1 + P2).

DIAMOND: https://github.com/eloialonso/diamond (NeurIPS 2024).
  - world model core = agent.denoiser (UNet in src/models/diffusion/inner_model.py)
  - hook point = agent.denoiser.inner_model.unet (d_blocks/mid/u_blocks)
  - load: hydra compose("trainer") -> Agent(instantiate(cfg.agent, num_actions=N)) -> agent.load(ckpt)
  - ckpt: hf_hub_download("eloialonso/diamond", "atari_100k/models/<Game>.pt")
  - seed (from play.py): make_atari_env -> collect real frames into Dataset ->
    BatchSampler -> DataLoader -> WorldModelEnv(denoiser, rew_end_model, dl, cfg)
  - step: wm_env.step(act:(n,) long) -> (next_obs, rew, term, trunc, info);
    the sampler calls denoiser.denoise (-> inner_model) once per denoising step.

Run:
  modal run modal_deploy/app_diamond.py::introspect      # P1: load + UNet structure
  modal run modal_deploy/app_diamond.py::capture         # P2: rollout + capture acts
"""
import modal

# Clone DIAMOND into the image and install its deps + Atari ROMs. The repo's
# src/ is added to sys.path at runtime; config/ is loaded via hydra by abs path.
image = (
    modal.Image.debian_slim(python_version="3.10")
    # git for the clone; libgl1 + libglib2.0-0 because DIAMOND's env modules
    # import opencv (cv2), which needs libGL.so.1 (absent from debian_slim).
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/eloialonso/diamond /root/diamond",
        "pip install -r /root/diamond/requirements.txt",
        # Atari ROMs (the agent/env modules import ale-py; num_actions etc.)
        "pip install 'autorom[accept-rom-license]' || true",
        "AutoROM --accept-license || true",
    )
    .pip_install("huggingface_hub")
)

app = modal.App("world-model-steering-diamond")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Start with Breakout (small action set, validated linear representations in the
# probing literature). ALE Breakout has 4 actions.
GAME = "Breakout"
NUM_ACTIONS = 4


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/cache": hf_volume},
    timeout=900,
    scaledown_window=300,
)
class DiamondWorldModel:
    @modal.enter()
    def load(self):
        import os
        import sys
        from pathlib import Path

        os.environ.setdefault("HF_HOME", "/cache/hf")
        sys.path.insert(0, "/root/diamond/src")
        os.chdir("/root/diamond")

        import torch
        from huggingface_hub import hf_hub_download
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        from agent import Agent
        from coroutines.collector import make_collector, NumToCollect
        from data import BatchSampler, collate_segments_to_batch, Dataset
        from envs import make_atari_env, WorldModelEnv

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # DIAMOND's configs use a custom ${eval:...} OmegaConf resolver (for
        # arithmetic in cfg.env / cfg.world_model_env). It's registered in the
        # repo's normal entrypoints; we compose() directly, so register it here.
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        # Hydra: load the repo's trainer config by absolute dir (cwd-independent).
        with initialize_config_dir(version_base="1.3", config_dir="/root/diamond/config"):
            cfg = compose(config_name="trainer")
        self.cfg = cfg

        # Real ALE env (also gives num_actions, matching the checkpoint head).
        print(f"Building real Atari env for {GAME}...")
        os.environ["ENV_TRAIN_ID"] = f"{GAME}NoFrameskip-v4"
        os.environ["ENV_TEST_ID"] = f"{GAME}NoFrameskip-v4"
        test_env = make_atari_env(num_envs=1, device=self.device, **cfg.env.test)
        self.num_actions = int(test_env.num_actions)

        print(f"Instantiating Agent (num_actions={self.num_actions})...")
        agent = Agent(instantiate(cfg.agent, num_actions=self.num_actions)).to(self.device)

        print(f"Downloading pretrained checkpoint for {GAME}...")
        ckpt = hf_hub_download(
            repo_id="eloialonso/diamond",
            filename=f"atari_100k/models/{GAME}.pt",
            cache_dir="/cache",
        )
        agent.load(ckpt)
        agent.eval()
        self.agent = agent

        # Seed the world model with real frames (play.py recipe). The WM
        # conditions on the last `num_steps_conditioning` real frames.
        n_collect = 128
        ds_dir = Path(f"/cache/wms_dataset/{GAME}_{n_collect}")
        ds_dir.mkdir(parents=True, exist_ok=True)
        dataset = Dataset(ds_dir)
        dataset.load_from_default_path()
        if len(dataset) == 0:
            print(f"Collecting {n_collect} real-env steps to seed the world model...")
            collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
            collector.send(NumToCollect(steps=n_collect))
            dataset.save_to_default_path()
        n_cond = cfg.agent.denoiser.inner_model.num_steps_conditioning
        bs = BatchSampler(dataset, 0, 1, 1, n_cond, None, False)
        dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
        wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
        self.wm_env = WorldModelEnv(
            agent.denoiser, agent.rew_end_model, dl, wm_env_cfg,
            return_denoising_trajectory=False,
        )

        # Persistent capture hook on the UNet. The diffusion sampler calls the
        # inner model once per denoising step; we keep the list and take the
        # last (lowest-sigma, most-resolved) activation per env.step.
        self._cap = []
        unet = self.agent.denoiser.inner_model.unet

        def cap_hook(module, inp, out):
            # UNet.forward returns a tuple (x, *_); keep the feature tensor x.
            t = out[0] if isinstance(out, (tuple, list)) else out
            self._cap.append(t.detach())
            return out

        self._cap_handle = unet.register_forward_hook(cap_hook)
        print(f"Loaded DIAMOND {GAME} world model + seeded WorldModelEnv on {self.device}.")

    @modal.method()
    def introspect(self):
        """Report the denoiser/UNet structure so we know where to hook in P2."""
        import torch

        denoiser = self.agent.denoiser
        inner = getattr(denoiser, "inner_model", denoiser)

        def n_params(m):
            return sum(p.numel() for p in m.parameters())

        # Top-level named children of the inner UNet (candidate hook points).
        top_children = [(name, type(mod).__name__) for name, mod in inner.named_children()]

        # A deeper sample of named_modules to see the block structure.
        all_named = [name for name, _ in inner.named_modules()]
        return {
            "denoiser_type": type(denoiser).__name__,
            "inner_model_type": type(inner).__name__,
            "denoiser_params_M": round(n_params(denoiser) / 1e6, 2),
            "inner_params_M": round(n_params(inner) / 1e6, 2),
            "inner_top_children": top_children,
            "n_named_modules": len(all_named),
            "named_modules_sample": all_named[:60],
            "device": str(self.device),
            "game": GAME,
        }

    @modal.method()
    def capture(self, n_frames: int = 16, action: int = 1):
        """Roll out the world model `n_frames` steps with a fixed action and
        capture the UNet activation at the final denoising step of each frame.

        Returns shapes + a cross-frame variation stat (std of per-channel means
        across frames > 0 means the activations track the evolving game state).
        Saves first/last frames + per-frame activation vectors to the volume.
        """
        from pathlib import Path

        import numpy as np
        import torch
        from PIL import Image

        wm = self.wm_env
        reset_out = wm.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out

        out_dir = Path(f"/cache/wms_capture/{GAME}")
        out_dir.mkdir(parents=True, exist_ok=True)

        frame_acts = []
        a_shape = None
        saved = 0
        lo = hi = 0.0
        next_obs = obs
        for i in range(n_frames):
            self._cap.clear()
            act_t = torch.tensor([action % self.num_actions], dtype=torch.long, device=self.device)
            step_out = wm.step(act_t)
            next_obs = step_out[0]
            if self._cap:
                a = self._cap[-1].float()  # (1, C, H, W) at the final denoise step
                a_shape = list(self._cap[-1].shape)
                frame_acts.append(a.mean(dim=(2, 3)).squeeze(0).cpu().numpy())  # (C,)
            # frame -> uint8 image (handle [-1,1] or [0,1] range)
            fr = next_obs.detach().float().squeeze(0).cpu()  # (C, H, W)
            lo, hi = float(fr.min()), float(fr.max())
            img = (fr.clamp(-1, 1) + 1) / 2 if lo < -0.01 else fr.clamp(0, 1)
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if i == 0 or i == n_frames - 1:
                Image.fromarray(img_np).save(out_dir / f"frame_{i:03d}.png")
                saved += 1
            obs = next_obs

        acts = np.stack(frame_acts) if frame_acts else np.zeros((0, 0))
        np.savez_compressed(out_dir / "unet_acts.npz", acts=acts)
        try:
            hf_volume.commit()
        except Exception:
            pass
        std_across = float(acts.std(axis=0).mean()) if acts.size else 0.0
        return {
            "game": GAME,
            "n_frames": n_frames,
            "num_actions": self.num_actions,
            "act_layer": "inner_model.unet",
            "act_shape_per_denoise": a_shape,
            "per_frame_vec_dim": int(acts.shape[1]) if acts.ndim == 2 and acts.size else None,
            "act_std_across_frames": std_across,
            "frame_shape": list(next_obs.shape),
            "frame_value_range": [lo, hi],
            "saved_dir": str(out_dir),
            "saved_frames": saved,
        }


@app.local_entrypoint()
def introspect():
    import json

    wm = DiamondWorldModel()
    info = wm.introspect.remote()
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def capture(n_frames: int = 16, action: int = 1):
    import json

    wm = DiamondWorldModel()
    info = wm.capture.remote(n_frames=n_frames, action=action)
    print(json.dumps(info, indent=2))
