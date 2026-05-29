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


@app.local_entrypoint()
def smoke():
    import json
    print(json.dumps(IrisWorldModel().smoke.remote(), indent=2))
