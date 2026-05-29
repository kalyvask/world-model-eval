"""
IRIS scale comparison: does the ~30-step fidelity horizon (measured on DIAMOND's
4.4M diffusion world model) extend on a larger, different-architecture world
model? IRIS = VQ-VAE tokenizer + autoregressive Transformer (Micheli et al.
2023), same Atari-100k / ALE pairing, so the fidelity experiment ports cleanly.

IRIS pins torch 1.11 (no sm_89 kernels) -> run on A100 (sm_80), not L40S.

Run:
  modal run modal_deploy/app_iris.py::smoke      # load + real-env + wm sanity
  modal run modal_deploy/app_iris.py::fidelity   # free-run dream vs real divergence
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/eloialonso/iris /root/iris",
        # gym==0.21 (pinned by IRIS) won't install under modern pip: its sdist
        # metadata won't parse under pip>=24, and it needs old setuptools/wheel
        # to build. Canonical fix: downgrade pip + build tools, then install gym
        # without build isolation so it uses the old setuptools.
        "python -m pip install 'pip==23.0.1'",
        "pip install 'setuptools==65.5.0' 'wheel==0.38.4'",
        "pip install --no-build-isolation 'gym==0.21.0'",
        # IRIS installs torch separately (not in requirements.txt). 1.13.1+cu117
        # has cp310 wheels, runs on A100 (sm_80), and is close to IRIS's 1.11 API.
        "pip install torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117",
        "pip install -r /root/iris/requirements.txt",
        "pip install 'autorom[accept-rom-license]' || true",
        "AutoROM --accept-license || true",
        "pip install huggingface_hub",
        # torch 1.13 / torchvision 0.14 were built against NumPy 1.x; IRIS's
        # reqs pull NumPy 2 -> "_ARRAY_API not found". Pin back to 1.x (last wins).
        "pip install 'numpy==1.26.4'",
    )
)

app = modal.App("world-model-eval-iris")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

GAME = "Breakout"


def _fidelity_stats(all_div, ceil_diffs, n_boot=2000, seed=0):
    """Floor/ceiling-normalized half-decorrelation step from per-trajectory
    divergence curves (n_traj x horizon) + per-traj decorrelated-ceiling diffs.
    Matches app_eval.py::_fidelity_stats so DIAMOND and IRIS are scored
    identically. Reports first-touch, sustained, and smoothed crossings plus a
    bootstrap 68% CI over trajectories; boot_frac_never_crosses high => the
    crossing is unreliable / the dream barely decorrelates within the window.
    """
    import numpy as np

    div = np.array(all_div, dtype=float)            # (n_traj, horizon)
    ceil_arr = np.array(ceil_diffs, dtype=float)
    n_traj, H = div.shape
    mean_div = div.mean(0)
    floor = float(mean_div[0])
    ceiling = float(ceil_arr.mean())

    def first_cross(curve, thr):
        for k, v in enumerate(curve):
            if v >= thr:
                return k + 1
        return None

    def sustained_cross(curve, thr):
        for k in range(len(curve)):
            if all(curve[j] >= thr for j in range(k, len(curve))):
                return k + 1
        return None

    thresh = floor + 0.5 * (ceiling - floor)
    cross_first = first_cross(mean_div, thresh)
    cross_sustained = sustained_cross(mean_div, thresh)
    sm = np.convolve(mean_div, np.ones(3) / 3.0, mode="same")
    cross_smoothed = first_cross(sm, thresh)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        ix = rng.integers(0, n_traj, n_traj)
        md = div[ix].mean(0)
        fl, ce = float(md[0]), float(ceil_arr[ix].mean())
        c = first_cross(md, fl + 0.5 * (ce - fl))
        boots.append(c if c is not None else H + 1)  # "never crosses" -> beyond window
    boots = np.array(boots, dtype=float)
    ci_lo, ci_hi = int(np.percentile(boots, 16)), int(np.percentile(boots, 84))
    frac_no_cross = round(float((boots > H).mean()), 3)

    return {
        "one_step_error": round(floor, 4),
        "decorrelated_ceiling": round(ceiling, 4),
        "half_decorrelation_step": cross_first,
        "half_decorrelation_step_sustained": cross_sustained,
        "half_decorrelation_step_smoothed": cross_smoothed,
        "half_decorrelation_ci68": [ci_lo, ci_hi],
        "boot_frac_never_crosses": frac_no_cross,
        "divergence_curve": [round(float(v), 4) for v in mean_div],
    }


@app.cls(
    image=image,
    gpu="A100-40GB",  # torch 1.11 supports sm_80; L40S (sm_89) is too new for it
    volumes={"/cache": hf_volume},
    timeout=1800,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],
)
class IrisWorldModel:
    @modal.enter()
    def load(self):
        import os
        import sys
        from functools import partial
        from pathlib import Path

        os.environ.setdefault("HF_HOME", "/cache/hf")
        sys.path.insert(0, "/root/iris/src")
        os.chdir("/root/iris")

        import torch
        from huggingface_hub import hf_hub_download
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from agent import Agent
        from envs import SingleProcessEnv, WorldModelEnv
        from models.actor_critic import ActorCritic
        from models.world_model import WorldModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)

        with initialize_config_dir(config_dir="/root/iris/config"):
            cfg = compose(config_name="trainer", overrides=[f"env.train.id={GAME}NoFrameskip-v4",
                                                            f"env.test.id={GAME}NoFrameskip-v4"])
        self.cfg = cfg

        env_fn = partial(instantiate, config=cfg.env.test)
        self.test_env = SingleProcessEnv(env_fn)
        self.num_actions = int(self.test_env.num_actions)

        tokenizer = instantiate(cfg.tokenizer)
        world_model = WorldModel(
            obs_vocab_size=tokenizer.vocab_size,
            act_vocab_size=self.num_actions,
            config=instantiate(cfg.world_model),
        )
        actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=self.num_actions)
        agent = Agent(tokenizer, world_model, actor_critic).to(self.device)

        ckpt = hf_hub_download(
            repo_id="eloialonso/iris",
            filename=f"pretrained_models/{GAME}.pt",
            cache_dir="/cache",
        )
        agent.load(Path(ckpt), device=self.device)
        agent.eval()
        self.agent = agent
        self.wm_env = WorldModelEnv(
            tokenizer=agent.tokenizer, world_model=agent.world_model,
            device=self.device, env=env_fn(),
        )
        print(f"IRIS ready: {GAME}, num_actions={self.num_actions}, device={self.device}")

    @modal.method()
    def smoke(self):
        """Load sanity: real-env reset (obs shape/range), num_actions, and a
        world-model reset+step round trip (confirms the transformer imagines)."""
        import numpy as np
        import torch

        reset_out = self.test_env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        obs_t = torch.as_tensor(obs).to(self.device) if not torch.is_tensor(obs) else obs.to(self.device)

        info = {
            "game": GAME, "num_actions": self.num_actions, "device": str(self.device),
            "real_obs_type": type(obs).__name__,
            "real_obs_shape": list(obs_t.shape),
            "real_obs_dtype": str(obs_t.dtype),
            "real_obs_range": [float(obs_t.float().min()), float(obs_t.float().max())],
        }
        # world-model round trip — IRIS tokenizer wants NCHW float [0,1]
        def to_np(x):
            return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else np.asarray(x)

        try:
            wm = self.wm_env
            obs_nchw = obs_t.permute(0, 3, 1, 2).float().div(255.0)  # (1,3,64,64) [0,1]
            wm_obs = wm.reset_from_initial_observations(obs_nchw)
            wm_obs2, rew, done, _ = wm.step(0)
            r, s2 = to_np(wm_obs), to_np(wm_obs2)
            info["wm_reset_shape"] = list(r.shape)
            info["wm_reset_range"] = [float(r.min()), float(r.max())]
            info["wm_step_shape"] = list(s2.shape)
            info["wm_step_range"] = [float(s2.min()), float(s2.max())]
            info["wm_step_reward"] = float(np.asarray(rew).sum())
            info["wm_step_ok"] = True
        except Exception as e:
            info["wm_step_ok"] = False
            info["wm_error"] = f"{type(e).__name__}: {e}"
        return info


    @modal.method()
    def fidelity(self, n_traj: int = 8, horizon: int = 60, fire: int = 8):
        """Same measurement as DIAMOND's app_eval.py::fidelity, on IRIS: seed the
        world model from a real post-burn-in frame, free-run under a fixed action
        sequence, and measure normalized frame divergence vs step. Output format
        matches DIAMOND's so the horizons are directly comparable.
        """
        import numpy as np
        import torch

        dev = self.device

        def to_nchw01(o):
            t = o if torch.is_tensor(o) else torch.as_tensor(o)
            t = t.to(dev).float()
            if t.dim() == 3:
                t = t.unsqueeze(0)
            if t.shape[-1] == 3:               # NHWC -> NCHW
                t = t.permute(0, 3, 1, 2)
            if float(t.max()) > 1.5:           # [0,255] -> [0,1]
                t = t / 255.0
            return t

        def l1(a, b):
            return float((a - b).abs().mean().item())

        def real_step(a):
            try:
                out = self.test_env.step(np.array([a]))
            except Exception:
                out = self.test_env.step(a)
            obs = out[0]
            done = out[2] if len(out) > 2 else False
            try:
                done = bool(np.asarray(done).any())
            except Exception:
                done = bool(done)
            return obs, done

        all_div, real_consec, ceil_diffs = [], [], []
        need = fire + horizon + 1
        for tr in range(n_traj):
            torch.manual_seed(700 + tr)
            rng = np.random.default_rng(700 + tr)
            actions = [1] * fire + [int(rng.integers(0, self.num_actions)) for _ in range(horizon)]
            reset_out = self.test_env.reset()
            obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
            frames = [to_nchw01(obs)]
            ok = True
            for t in range(need - 1):
                obs, done = real_step(actions[t])
                frames.append(to_nchw01(obs))
                if done:
                    ok = False
                    break
            if not ok or len(frames) < need:
                continue

            self.wm_env.reset_from_initial_observations(frames[fire])
            divs = []
            for k in range(horizon):
                imagined, _, _, _ = self.wm_env.step(actions[fire + k])
                img = to_nchw01(imagined)
                divs.append(l1(img, frames[fire + k + 1]))
            all_div.append(divs)
            for t in range(1, len(frames)):
                real_consec.append(l1(frames[t], frames[t - 1]))
            ceil_diffs.append(l1(frames[fire], frames[fire + horizon]))

        if not all_div:
            return {"game": GAME, "model": "IRIS", "error": "no full-length trajectories", "n_traj": n_traj}
        stats = _fidelity_stats(all_div, ceil_diffs)
        stats.update({
            "game": GAME, "model": "IRIS", "policy": "random",
            "n_traj_used": len(all_div), "horizon": horizon,
            "real_consecutive_frame_diff": round(float(np.mean(real_consec)), 4),
        })
        return stats


@app.local_entrypoint()
def smoke():
    import json
    print(json.dumps(IrisWorldModel().smoke.remote(), indent=2))


@app.local_entrypoint()
def fidelity(n_traj: int = 8, horizon: int = 60):
    import json
    print(json.dumps(IrisWorldModel().fidelity.remote(n_traj=n_traj, horizon=horizon), indent=2))
