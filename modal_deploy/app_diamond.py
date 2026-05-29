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
    .pip_install("huggingface_hub", "scikit-learn")
)

app = modal.App("world-model-steering-diamond")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Start with Breakout (small action set, validated linear representations in the
# probing literature). ALE Breakout has 4 actions.
GAME = "Breakout"
NUM_ACTIONS = 4


def _detect_ball_paddle(frame_chw, prev_chw=None):
    """CV labels from a generated Breakout frame (3,H,W) in [-1,1].

    Paddle: brightest pixels in the bottom band (no bricks there).
    Ball: the MOVING bright object, via frame differencing in the play area
    (robust to the static bright brick mass). Returns None for the ball on the
    first frame or when there's no motion.
    """
    import numpy as np

    g = ((frame_chw.mean(0) + 1.0) / 2.0).clip(0, 1)  # (H,W) grayscale [0,1]
    H, W = g.shape

    # Paddle: bright pixels in the bottom band.
    pad_x = None
    band = g[int(H * 0.86):, :]
    pys, pxs = np.nonzero(band > 0.5)
    if pxs.size > 2:
        pad_x = float(pxs.mean())

    # Ball: the moving bright object (frame diff), in the play area below the
    # bricks and above the paddle. Static bricks cancel out in the difference.
    ball_x = ball_y = None
    if prev_chw is not None:
        pg = ((prev_chw.mean(0) + 1.0) / 2.0).clip(0, 1)
        y0, y1 = int(H * 0.40), int(H * 0.84)
        d = np.abs(g[y0:y1, :] - pg[y0:y1, :])
        if d.size and float(d.max()) > 0.08:
            bys, bxs = np.nonzero(d >= d.max() * 0.7)
            if bxs.size > 0:
                ball_x = float(bxs.mean())
                ball_y = float(bys.mean()) + y0
    return ball_x, ball_y, pad_x


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
        self._steer = None  # (1,C,H,W) tensor added to the UNet output when set
        unet = self.agent.denoiser.inner_model.unet

        def cap_hook(module, inp, out):
            # UNet.forward returns a tuple (x, *_); keep the feature tensor x.
            x = out[0] if isinstance(out, (tuple, list)) else out
            self._cap.append(x.detach())
            if self._steer is not None:
                x = x + self._steer
                return ((x,) + tuple(out[1:])) if isinstance(out, (tuple, list)) else x
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

    @modal.method()
    def probe(self, n_frames: int = 400, pool: int = 16):
        """P3: is game state linearly decodable from the UNet activation?

        Roll out the world model with random actions (so ball/paddle move),
        capture the full spatial UNet activation per frame (adaptive-pooled to
        pool x pool to keep the probe tractable), CV-detect ball/paddle in the
        generated frame, then ridge-regress activation -> position on a held-out
        split. High test R^2 => the activation linearly encodes that concept
        (and gives us a steering direction for P4). Saves (X, labels) for P4.
        """
        from pathlib import Path

        import numpy as np
        import torch
        import torch.nn.functional as F
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        wm = self.wm_env
        ac = self.agent.actor_critic
        lstm_dim = ac.lstm.hidden_size
        hx = torch.zeros(1, lstm_dim, device=self.device)
        cx = torch.zeros(1, lstm_dim, device=self.device)
        reset_out = wm.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out

        feats, bx, by, px = [], [], [], []
        frame_l1 = []
        prev_fr = None
        n_fire = 8  # FIRE burn-in to launch the ball
        for i in range(n_frames):
            # Drive imagination with the pretrained policy (on-distribution),
            # forcing FIRE early to launch the ball.
            with torch.no_grad():
                ac_out = ac.predict_act_value(obs, (hx, cx))
            hx, cx = ac_out.hx_cx
            if i < n_fire:
                a = torch.tensor([1], dtype=torch.long, device=self.device)  # FIRE
            else:
                a = torch.distributions.Categorical(logits=ac_out.logits_act).sample().long()
            self._cap.clear()
            step_out = wm.step(a)
            next_obs = step_out[0]
            obs = next_obs
            if not self._cap:
                continue
            act = self._cap[-1].float()  # (1, 64, 64, 64)
            pooled = F.adaptive_avg_pool2d(act, (pool, pool)).flatten().cpu().numpy()
            fr = next_obs.detach().float().squeeze(0).cpu().numpy()  # (3, H, W)
            if prev_fr is not None:
                frame_l1.append(float(np.abs(fr - prev_fr).mean()))
            bxc, byc, pxc = _detect_ball_paddle(fr, prev_fr)
            prev_fr = fr
            feats.append(pooled)
            bx.append(bxc); by.append(byc); px.append(pxc)

        X = np.asarray(feats, dtype=np.float32)

        def fit(label_list, name):
            y = np.array([v if v is not None else np.nan for v in label_list], dtype=float)
            mask = ~np.isnan(y)
            if mask.sum() < 60:
                return {"concept": name, "n": int(mask.sum()), "test_r2": None, "note": "too few detections"}
            Xs, ys = X[mask], y[mask]
            Xtr, Xte, ytr, yte = train_test_split(Xs, ys, test_size=0.25, random_state=0)
            n_comp = int(min(128, Xtr.shape[0] - 1, Xtr.shape[1]))
            pipe = make_pipeline(StandardScaler(), PCA(n_components=n_comp), Ridge(alpha=10.0))
            pipe.fit(Xtr, ytr)
            return {
                "concept": name,
                "n": int(mask.sum()),
                "test_r2": round(float(r2_score(yte, pipe.predict(Xte))), 3),
                "label_std_px": round(float(ys.std()), 2),
            }

        results = [fit(bx, "ball_x"), fit(by, "ball_y"), fit(px, "paddle_x")]

        out_dir = Path(f"/cache/wms_capture/{GAME}")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "probe_data.npz",
            X=X,
            ball_x=np.array([v if v is not None else np.nan for v in bx]),
            ball_y=np.array([v if v is not None else np.nan for v in by]),
            paddle_x=np.array([v if v is not None else np.nan for v in px]),
            pool=pool,
        )
        try:
            hf_volume.commit()
        except Exception:
            pass
        return {
            "game": GAME,
            "n_frames": n_frames,
            "feat_dim": int(X.shape[1]) if X.size else 0,
            "pool": pool,
            "detect_rate_ball": round(float(np.mean([v is not None for v in bx])), 3),
            "detect_rate_paddle": round(float(np.mean([v is not None for v in px])), 3),
            "frame_l1_mean": round(float(np.mean(frame_l1)), 4) if frame_l1 else 0.0,
            "feat_std_across_frames": round(float(X.std(axis=0).mean()), 5) if X.size else 0.0,
            "label_sample_ballx_bally_padx": [
                [None if v is None else round(v, 1) for v in t]
                for t in list(zip(bx[:14], by[:14], px[:14]))
            ],
            "probes": results,
        }

    @modal.method()
    def steer_ball(self, n_probe: int = 300, n_eval: int = 48, alpha: float = 16.0):
        """P4: does adding the ball_x probe direction MOVE the ball in the dream?

        (1) probing rollout -> ridge probe -> full-res unit direction for ball_x
        (back-projected through scaler+PCA, upsampled 16x16 -> 64x64);
        (2) replay one fixed action sequence + fixed seed at several steering
        magnitudes (0, +/-alpha, +/-2alpha) plus a matched-norm random-direction
        control; report the mean ball_x at each. A monotonic shift with alpha
        (and a flat random control) = the direction causally moves the ball.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        from sklearn.decomposition import PCA
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        pool = 16
        ac = self.agent.actor_critic
        dev = self.device

        def reset_obs():
            r = self.wm_env.reset()
            return r[0] if isinstance(r, (tuple, list)) else r

        # ---- (1) probing rollout -> difference-of-means direction ----
        # Capture FULL-res (C,H,W) activations + ball_x; the steering direction
        # is mean(act | ball far right) - mean(act | ball far left). This keeps
        # the localized spatial structure (where the ball is in the activation),
        # unlike a pooled ridge decode-direction.
        torch.manual_seed(0)
        hx = torch.zeros(1, ac.lstm.hidden_size, device=dev)
        cx = torch.zeros_like(hx)
        obs = reset_obs()
        prev = None
        acts, yb = [], []
        for i in range(n_probe):
            with torch.no_grad():
                o = ac.predict_act_value(obs, (hx, cx))
            hx, cx = o.hx_cx
            a = (torch.tensor([1], device=dev) if i < 8
                 else torch.distributions.Categorical(logits=o.logits_act).sample()).long()
            self._cap.clear(); self._steer = None
            obs = self.wm_env.step(a)[0]
            if not self._cap:
                continue
            acts.append(self._cap[-1].float().squeeze(0).half().cpu())  # (C,H,W) f16
            fr = obs.detach().float().squeeze(0).cpu().numpy()
            bxc, _, _ = _detect_ball_paddle(fr, prev); prev = fr
            yb.append(bxc if bxc is not None else np.nan)
        yb = np.array(yb)
        idx = np.where(~np.isnan(yb))[0]
        order = idx[np.argsort(yb[idx])]              # ascending ball_x
        k = max(10, len(order) // 3)                   # terciles
        low_idx, high_idx = order[:k], order[-k:]
        mean_low = torch.stack([acts[j] for j in low_idx]).float().mean(0)    # (C,H,W)
        mean_high = torch.stack([acts[j] for j in high_idx]).float().mean(0)
        d = (mean_high - mean_low).unsqueeze(0).to(dev)                        # (1,C,H,W)
        d = d / (d.norm() + 1e-8)                                              # unit norm
        act_norm = float(acts[-1].float().norm())
        x_low = float(np.mean(yb[low_idx])); x_high = float(np.mean(yb[high_idx]))

        # ---- (2) one fixed action sequence + seed, swept over steering ----
        def run(actions, steer):
            torch.manual_seed(123)
            hxr = torch.zeros(1, ac.lstm.hidden_size, device=dev); cxr = torch.zeros_like(hxr)
            obs = reset_obs(); prev = None; bxs = []; rec = []
            self._steer = steer
            for i in range(n_eval):
                if actions is None:
                    with torch.no_grad():
                        o = ac.predict_act_value(obs, (hxr, cxr))
                    hxr, cxr = o.hx_cx
                    a = (torch.tensor([1], device=dev) if i < 8
                         else torch.distributions.Categorical(logits=o.logits_act).sample()).long()
                    rec.append(int(a.item()))
                else:
                    a = torch.tensor([actions[i]], dtype=torch.long, device=dev)
                self._cap.clear()
                obs = self.wm_env.step(a)[0]
                fr = obs.detach().float().squeeze(0).cpu().numpy()
                bxc, _, _ = _detect_ball_paddle(fr, prev); prev = fr
                bxs.append(bxc)
            self._steer = None
            vals = [v for v in bxs if v is not None]
            return (float(np.mean(vals)) if vals else None, len(vals), rec)

        _, _, actions = run(None, None)  # record a fixed action sequence
        rows = []
        for mult in [0.0, 1.0, -1.0, 2.0, -2.0]:
            steer = None if mult == 0 else (mult * alpha) * d
            mbx, ndet, _ = run(actions, steer)
            rows.append({"alpha": round(mult * alpha, 1), "mean_ball_x": round(mbx, 2) if mbx is not None else None, "n_detected": ndet})
        base = next((r["mean_ball_x"] for r in rows if r["alpha"] == 0.0), None)
        for r in rows:
            r["delta_vs_base"] = (round(r["mean_ball_x"] - base, 2) if (r["mean_ball_x"] is not None and base is not None) else None)
        # matched-norm random-direction control at +alpha
        torch.manual_seed(7)
        rnd = torch.randn_like(d); rnd = rnd / (rnd.norm() + 1e-8)
        rmbx, rnd_n, _ = run(actions, alpha * rnd)
        return {
            "game": GAME, "concept": "ball_x", "n_probe": n_probe, "n_eval": n_eval,
            "dir_method": "difference_of_means_full_res",
            "dom_contrast_ball_x_low_high": [round(x_low, 1), round(x_high, 1)],
            "act_norm": round(act_norm, 1), "steer_dir_norm": "unit",
            "baseline_mean_ball_x": base,
            "steer_curve": rows,
            "random_dir_control": {"alpha": alpha, "mean_ball_x": round(rmbx, 2) if rmbx is not None else None,
                                   "delta_vs_base": round(rmbx - base, 2) if (rmbx is not None and base is not None) else None},
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


@app.local_entrypoint()
def probe(n_frames: int = 400, pool: int = 16):
    import json

    wm = DiamondWorldModel()
    info = wm.probe.remote(n_frames=n_frames, pool=pool)
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def steer_ball(n_probe: int = 300, n_eval: int = 48, alpha: float = 16.0):
    import json

    wm = DiamondWorldModel()
    info = wm.steer_ball.remote(n_probe=n_probe, n_eval=n_eval, alpha=alpha)
    print(json.dumps(info, indent=2))
