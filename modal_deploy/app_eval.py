"""
DreamEval — does a small world model's *imagined* return rank policies the same
way the *real* environment does?

DIAMOND-Breakout on Modal. Policy spectrum = epsilon-greedy on the pretrained
actor-critic (eps 0 = good, eps 1 = random). For each policy we measure:
  - real_return:     rollout in the real ALE env (ground truth)
  - imagined_return: rollout inside WorldModelEnv, summed reward-model reward
and report the Spearman/Pearson correlation across the spectrum.

Run:
  modal run modal_deploy/app_eval.py::smoke        # one policy, sanity
  modal run modal_deploy/app_eval.py::run_eval     # full spectrum + correlation
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.10")
    # git for the clone; libgl1+libglib2.0-0 for DIAMOND's opencv import.
    .apt_install("git", "libgl1", "libglib2.0-0")
    .run_commands(
        "git clone https://github.com/eloialonso/diamond /root/diamond",
        "pip install -r /root/diamond/requirements.txt",
        "pip install 'autorom[accept-rom-license]' || true",
        "AutoROM --accept-license || true",
    )
    .pip_install("huggingface_hub", "scipy")
)

app = modal.App("world-model-eval")
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

GAME = "Breakout"


def _fidelity_stats(all_div, ceil_diffs, n_boot=2000, seed=0):
    """Floor/ceiling-normalized half-decorrelation step from per-trajectory
    divergence curves (n_traj x horizon) + per-traj decorrelated-ceiling diffs.

    The original "first step >= threshold" crossing is noise-sensitive when the
    curve hovers near the threshold, so report it alongside a *sustained*
    crossing (first step from which the curve stays above) and a 3-point smoothed
    crossing, plus a bootstrap 68% CI over trajectories. boot_frac_never_crosses
    = fraction of bootstrap resamples that never reach the threshold within the
    window (high => the crossing is unreliable / the dream barely decorrelates).
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
    gpu="L40S",
    volumes={"/cache": hf_volume},
    timeout=1800,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("hf-token")],
)
class DreamEval:
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

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        with initialize_config_dir(version_base="1.3", config_dir="/root/diamond/config"):
            cfg = compose(config_name="trainer")
        self.cfg = cfg

        os.environ["ENV_TRAIN_ID"] = f"{GAME}NoFrameskip-v4"
        os.environ["ENV_TEST_ID"] = f"{GAME}NoFrameskip-v4"
        self.test_env = make_atari_env(num_envs=1, device=self.device, **cfg.env.test)
        self.num_actions = int(self.test_env.num_actions)

        agent = Agent(instantiate(cfg.agent, num_actions=self.num_actions)).to(self.device)
        ckpt = hf_hub_download(
            repo_id="eloialonso/diamond",
            filename=f"atari_100k/models/{GAME}.pt",
            cache_dir="/cache",
        )
        agent.load(ckpt)
        agent.eval()
        self.agent = agent

        n_collect = 128
        ds_dir = Path(f"/cache/wms_dataset/{GAME}_{n_collect}")
        ds_dir.mkdir(parents=True, exist_ok=True)
        dataset = Dataset(ds_dir)
        dataset.load_from_default_path()
        if len(dataset) == 0:
            collector = make_collector(self.test_env, agent.actor_critic, dataset, epsilon=0)
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
        print(f"DreamEval ready: {GAME}, num_actions={self.num_actions}, device={self.device}")

    # -----------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------
    def _zeros_hidden(self):
        import torch
        h = self.agent.actor_critic.lstm.hidden_size
        return torch.zeros(1, h, device=self.device), torch.zeros(1, h, device=self.device)

    def _eps_action(self, logits, epsilon, rng):
        import torch
        if rng.random() < epsilon:
            return torch.randint(0, self.num_actions, (1,), device=self.device).long()
        return logits.argmax(dim=-1).long()  # greedy exploit

    @staticmethod
    def _scalar(x):
        try:
            return float(x.sum().item())
        except Exception:
            return float(x)

    @staticmethod
    def _done(term, trunc):
        def anyf(v):
            try:
                return bool(v.any().item())
            except Exception:
                return bool(v)
        return anyf(term) or anyf(trunc)

    def _rollout_return(self, env, epsilon, horizon, seed, fire_burn_in=8):
        """One episode in `env` under an epsilon-greedy actor-critic policy;
        returns undiscounted summed reward. Works for both the real ALE env and
        WorldModelEnv (same reset/step gymnasium-style API)."""
        import random

        import torch

        rng = random.Random(seed)
        torch.manual_seed(seed)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        hx, cx = self._zeros_hidden()
        total = 0.0
        for t in range(horizon):
            with torch.no_grad():
                o = self.agent.actor_critic.predict_act_value(obs, (hx, cx))
            hx, cx = o.hx_cx
            if t < fire_burn_in:
                a = torch.ones(1, dtype=torch.long, device=self.device)  # FIRE to launch
            else:
                a = self._eps_action(o.logits_act, epsilon, rng)
            step_out = env.step(a)
            obs, rew, term, trunc = step_out[0], step_out[1], step_out[2], step_out[3]
            total += self._scalar(rew)
            if self._done(term, trunc):
                break
        return total

    def _real_return(self, epsilon, n_eps, horizon):
        import numpy as np
        rets = [self._rollout_return(self.test_env, epsilon, horizon, seed=s) for s in range(n_eps)]
        return float(np.mean(rets)), [round(r, 2) for r in rets]

    def _imagined_return(self, epsilon, n_roll, horizon):
        import numpy as np
        rets = [self._rollout_return(self.wm_env, epsilon, horizon, seed=1000 + s) for s in range(n_roll)]
        return float(np.mean(rets)), [round(r, 2) for r in rets]

    # -----------------------------------------------------------------------
    # endpoints
    # -----------------------------------------------------------------------
    @modal.method()
    def smoke(self, horizon: int = 80):
        """Sanity: returns for the good policy (eps=0) vs random (eps=1) in both
        the real env and the dream. Good policy should beat random in both."""
        out = {}
        for name, eps in [("good_eps0", 0.0), ("random_eps1", 1.0)]:
            rr, rl = self._real_return(eps, 3, horizon)
            ir, il = self._imagined_return(eps, 3, horizon)
            out[name] = {"real_mean": round(rr, 3), "real": rl,
                         "imagined_mean": round(ir, 3), "imagined": il}
        return {"game": GAME, "horizon": horizon, "results": out}

    @modal.method()
    def run_eval(self, epsilons: list = None, n_real: int = 5, n_imag: int = 8,
                 real_horizon: int = 400, imag_horizon: int = 80):
        import numpy as np
        from scipy.stats import pearsonr, spearmanr

        if epsilons is None:
            epsilons = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        rows = []
        for eps in epsilons:
            rr, _ = self._real_return(eps, n_real, real_horizon)
            ir, _ = self._imagined_return(eps, n_imag, imag_horizon)
            rows.append({"epsilon": eps, "real_return": round(rr, 3), "imagined_return": round(ir, 3)})
        real = [r["real_return"] for r in rows]
        imag = [r["imagined_return"] for r in rows]
        sp = spearmanr(real, imag)
        pe = pearsonr(real, imag)
        return {
            "game": GAME,
            "n_real_eps": n_real, "n_imag_rollouts": n_imag,
            "real_horizon": real_horizon, "imag_horizon": imag_horizon,
            "rows": rows,
            "spearman": round(float(sp.correlation), 3),
            "spearman_p": round(float(sp.pvalue), 4),
            "pearson": round(float(pe.statistic), 3),
        }

    # -----------------------------------------------------------------------
    # E3: strengthened correlation + correlation-vs-horizon curve
    # -----------------------------------------------------------------------
    def _rollout_cumreward(self, env, epsilon, horizon, seed, fire_burn_in=8):
        """Like _rollout_return but returns the cumulative-reward trajectory
        (list, one entry per executed step) so we can read the return at any
        horizon checkpoint from a single rollout."""
        import random

        import torch

        rng = random.Random(seed)
        torch.manual_seed(seed)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        hx, cx = self._zeros_hidden()
        total = 0.0
        cum = []
        for t in range(horizon):
            with torch.no_grad():
                o = self.agent.actor_critic.predict_act_value(obs, (hx, cx))
            hx, cx = o.hx_cx
            if t < fire_burn_in:
                a = torch.ones(1, dtype=torch.long, device=self.device)
            else:
                a = self._eps_action(o.logits_act, epsilon, rng)
            step_out = env.step(a)
            obs, rew, term, trunc = step_out[0], step_out[1], step_out[2], step_out[3]
            total += self._scalar(rew)
            cum.append(total)
            if self._done(term, trunc):
                break
        return cum

    def _imagined_curve(self, epsilon, n_roll, horizon, checkpoints):
        """Mean imagined cumulative return at each horizon checkpoint."""
        import numpy as np
        cps = sorted(checkpoints)
        per_cp = {h: [] for h in cps}
        for r in range(n_roll):
            cum = self._rollout_cumreward(self.wm_env, epsilon, horizon, seed=2000 + r)
            for h in cps:
                per_cp[h].append(cum[min(h, len(cum)) - 1] if cum else 0.0)
        return {h: float(np.mean(per_cp[h])) for h in cps}

    @modal.method()
    def run_eval_curve(self, epsilons: list = None, n_real: int = 10, real_horizon: int = 300,
                       n_imag: int = 32, imag_horizon: int = 150, checkpoints: list = None):
        import numpy as np
        from scipy.stats import pearsonr, spearmanr

        if epsilons is None:
            epsilons = [0.0, 0.1, 0.25, 0.4, 0.6, 0.8, 1.0]
        if checkpoints is None:
            checkpoints = [30, 60, 90, 120, 150]
        checkpoints = [h for h in sorted(checkpoints) if h <= imag_horizon]

        real_returns, rows = [], []
        imag_at = {h: [] for h in checkpoints}
        for eps in epsilons:
            rr, _ = self._real_return(eps, n_real, real_horizon)
            real_returns.append(rr)
            cum = self._imagined_curve(eps, n_imag, imag_horizon, checkpoints)
            for h in checkpoints:
                imag_at[h].append(cum[h])
            rows.append({"epsilon": eps, "real_return": round(rr, 3),
                         "imagined_at": {h: round(cum[h], 3) for h in checkpoints}})

        curve = []
        for h in checkpoints:
            sp = spearmanr(real_returns, imag_at[h])
            pe = pearsonr(real_returns, imag_at[h])
            curve.append({"horizon": h,
                          "spearman": round(float(sp.correlation), 3),
                          "spearman_p": round(float(sp.pvalue), 4),
                          "pearson": round(float(pe.statistic), 3)})
        best = max(curve, key=lambda c: (c["spearman"] if c["spearman"] == c["spearman"] else -9))
        return {
            "game": GAME, "epsilons": epsilons,
            "n_real": n_real, "real_horizon": real_horizon,
            "n_imag": n_imag, "imag_horizon": imag_horizon,
            "real_returns": [round(r, 3) for r in real_returns],
            "imagined_at_best_horizon": [round(v, 3) for v in imag_at[best["horizon"]]],
            "rows": rows,
            "horizon_curve": curve,
            "best_horizon": best,
        }

    # -----------------------------------------------------------------------
    # Fidelity horizon: free-running dream vs real, same actions
    # -----------------------------------------------------------------------
    @modal.method()
    def fidelity(self, n_traj: int = 8, horizon: int = 60, policy: str = "greedy"):
        """Seed the world model from a real trajectory's context, free-run it
        forward under the SAME actions as the real env, and measure normalized
        frame divergence vs dream step. References: the one-step error (floor),
        the natural consecutive-real-frame change, and the divergence between
        unrelated real frames (decorrelated ceiling). The fidelity horizon = the
        dream step at which divergence reaches halfway to the ceiling.

        policy: "greedy" drives the real trajectory with the pretrained
        actor-critic (in-distribution); "random" uses uniform-random actions to
        match IRIS's app_iris.py::fidelity, so the DIAMOND vs IRIS horizon
        comparison is apples-to-apples on action distribution.
        """
        import numpy as np
        import torch

        n_cond = int(self.cfg.agent.denoiser.inner_model.num_steps_conditioning)
        wm = self.wm_env
        ac = self.agent.actor_critic
        dev = self.device

        def l1(a, b):
            return float((a - b).abs().mean().item())

        all_div, real_step_diffs, ceil_diffs = [], [], []
        need = n_cond + horizon + 1
        for tr in range(n_traj):
            torch.manual_seed(500 + tr)
            rng = np.random.default_rng(500 + tr)  # only used when policy="random"
            reset_out = self.test_env.reset()
            obs = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
            hx, cx = self._zeros_hidden()
            frames = [obs.detach().clone()]
            acts = []
            for t in range(need):
                if policy == "random":
                    a = (torch.ones(1, dtype=torch.long, device=dev) if t < 8
                         else torch.tensor([int(rng.integers(0, self.num_actions))],
                                           dtype=torch.long, device=dev))
                else:  # greedy: advance the actor-critic LSTM every step, as before
                    with torch.no_grad():
                        o = ac.predict_act_value(obs, (hx, cx))
                    hx, cx = o.hx_cx
                    a = torch.ones(1, dtype=torch.long, device=dev) if t < 8 else o.logits_act.argmax(-1).long()
                step_out = self.test_env.step(a)
                obs = step_out[0]
                frames.append(obs.detach().clone())
                acts.append(a)
                if self._done(step_out[2], step_out[3]):
                    break
            if len(frames) < need:
                continue  # episode ended early; skip for a clean fixed-length compare

            # seed the dream from the real context, then free-run
            wm.obs_buffer = torch.stack(frames[0:n_cond], dim=1).to(dev)
            wm.act_buffer = torch.stack(acts[0:n_cond], dim=1).to(dev)
            divs = []
            for k in range(horizon):
                t = n_cond + k                      # predicting real frame index t
                wm.act_buffer[:, -1] = acts[t - 1]
                imagined, _ = wm.predict_next_obs()
                wm.obs_buffer = wm.obs_buffer.roll(-1, dims=1)
                wm.act_buffer = wm.act_buffer.roll(-1, dims=1)
                wm.obs_buffer[:, -1] = imagined
                divs.append(l1(imagined.squeeze(0), frames[t].squeeze(0)))
            all_div.append(divs)
            for t in range(1, n_cond + horizon):
                real_step_diffs.append(l1(frames[t].squeeze(0), frames[t - 1].squeeze(0)))
            ceil_diffs.append(l1(frames[n_cond].squeeze(0), frames[n_cond + horizon].squeeze(0)))

        if not all_div:
            return {"game": GAME, "error": "no full-length trajectories", "n_traj": n_traj}
        stats = _fidelity_stats(all_div, ceil_diffs)
        stats.update({
            "game": GAME, "model": "DIAMOND", "policy": policy,
            "n_traj_used": len(all_div), "horizon": horizon, "n_cond": n_cond,
            "real_consecutive_frame_diff": round(float(np.mean(real_step_diffs)), 4),
        })
        return stats


@app.local_entrypoint()
def smoke(horizon: int = 80):
    import json
    print(json.dumps(DreamEval().smoke.remote(horizon=horizon), indent=2))


@app.local_entrypoint()
def run_eval(n_real: int = 5, n_imag: int = 8, real_horizon: int = 400, imag_horizon: int = 80):
    import json
    info = DreamEval().run_eval.remote(
        n_real=n_real, n_imag=n_imag, real_horizon=real_horizon, imag_horizon=imag_horizon
    )
    print(json.dumps(info, indent=2))


@app.local_entrypoint()
def fidelity(n_traj: int = 8, horizon: int = 60, policy: str = "both"):
    import json
    wm = DreamEval()
    if policy == "both":
        out = {
            "greedy": wm.fidelity.remote(n_traj=n_traj, horizon=horizon, policy="greedy"),
            "random": wm.fidelity.remote(n_traj=n_traj, horizon=horizon, policy="random"),
        }
    else:
        out = wm.fidelity.remote(n_traj=n_traj, horizon=horizon, policy=policy)
    print(json.dumps(out, indent=2))


@app.local_entrypoint()
def run_eval_curve(n_real: int = 10, real_horizon: int = 300, n_imag: int = 32, imag_horizon: int = 150,
                   n_eps: int = 7):
    import json
    # Denser epsilon grid (more policies) tightens the n-limited Spearman.
    epsilons = [round(i / (n_eps - 1), 3) for i in range(n_eps)] if n_eps > 1 else [0.0]
    checkpoints = sorted({max(10, imag_horizon // 3), max(10, 2 * imag_horizon // 3), imag_horizon})
    info = DreamEval().run_eval_curve.remote(
        epsilons=epsilons, n_real=n_real, real_horizon=real_horizon,
        n_imag=n_imag, imag_horizon=imag_horizon, checkpoints=checkpoints,
    )
    print(json.dumps(info, indent=2))
