"""
B1 / P1 — load a pretrained DIAMOND Atari diffusion world model headless on
Modal and introspect it. De-risks the whole integration (image builds, deps +
ROMs install, checkpoint downloads + loads) and surfaces the UNet layer names
we'll hook in P2 for activation capture / steering.

DIAMOND: https://github.com/eloialonso/diamond (NeurIPS 2024).
  - world model core = agent.denoiser (UNet in src/models/diffusion/inner_model.py)
  - load: hydra compose("trainer") -> Agent(instantiate(cfg.agent, num_actions=N)) -> agent.load(ckpt)
  - ckpt: hf_hub_download("eloialonso/diamond", "atari_100k/models/<Game>.pt")

Run:
  modal run modal_deploy/app_diamond.py::introspect
"""
import modal

# Clone DIAMOND into the image and install its deps + Atari ROMs. The repo's
# src/ is added to sys.path at runtime; config/ is loaded via hydra by abs path.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
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

        os.environ.setdefault("HF_HOME", "/cache/hf")
        sys.path.insert(0, "/root/diamond/src")
        os.chdir("/root/diamond")

        import torch
        from huggingface_hub import hf_hub_download
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        from agent import Agent

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Hydra: load the repo's trainer config by absolute dir so cwd/file
        # location doesn't matter.
        with initialize_config_dir(version_base="1.3", config_dir="/root/diamond/config"):
            cfg = compose(config_name="trainer")
        self.cfg = cfg

        print(f"Instantiating Agent (num_actions={NUM_ACTIONS})...")
        agent = Agent(instantiate(cfg.agent, num_actions=NUM_ACTIONS)).to(self.device)

        print(f"Downloading pretrained checkpoint for {GAME}...")
        ckpt = hf_hub_download(
            repo_id="eloialonso/diamond",
            filename=f"atari_100k/models/{GAME}.pt",
            cache_dir="/cache",
        )
        agent.load(ckpt)
        agent.eval()
        self.agent = agent
        print(f"Loaded DIAMOND {GAME} world model on {self.device}.")

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


@app.local_entrypoint()
def introspect():
    import json

    wm = DiamondWorldModel()
    info = wm.introspect.remote()
    print(json.dumps(info, indent=2))
