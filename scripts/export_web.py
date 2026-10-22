"""Export the hold policy's actor (deterministic mean action) as a float32 blob for the website, plus a
reference rollout the browser build is checked against. Writes website/shared/live/{policy.bin,policy.json,ref.json}."""

import json
from pathlib import Path

import numpy as np
import yaml

from rlarm.common import load_policy, make_env

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "website/shared/live"
run = ROOT / yaml.safe_load((ROOT / "configs/demo.yaml").read_text())["run"]
cfg = yaml.safe_load((run / "config.yaml").read_text())
model = load_policy(cfg, run / "model.zip")
actor = model.actor
lin = [m for m in actor.latent_pi if m.__class__.__name__ == "Linear"] + [actor.mu]
scale = actor.features_extractor.scale
mats = [(m.weight.detach().numpy().astype(np.float32), m.bias.detach().numpy().astype(np.float32)) for m in lin]


def forward(obs: dict) -> np.ndarray:
    """Numpy copy of the exported network (what the browser runs)."""
    ag, dg = obs["achieved_goal"], obs["desired_goal"]
    x = np.r_[obs["observation"], ag, dg, (dg - ag) * scale].astype(np.float32)
    for i, (w, b) in enumerate(mats):
        x = w @ x + b
        if i < len(mats) - 1:
            x = np.maximum(x, 0)
    return np.tanh(x)


u = make_env(cfg).unwrapped
obs, _ = u.reset(seed=0)
rng = np.random.default_rng(1)
from rlarm.kinematics import fk  # noqa: E402

ref = {"q0": u.data.qpos[:3].tolist(), "steps": []}
worst = 0.0
for t in range(200):
    if t % 60 == 0:
        q = rng.uniform(u.lim[:, 0], u.lim[:, 1])
        u.goal = np.asarray(fk(q), float)
        u.data.mocap_pos[u.mocap] = u.goal
        obs = u._obs()
    a_sb3 = model.predict(obs, deterministic=True)[0]
    a = forward(obs)
    worst = max(worst, float(np.abs(a - a_sb3).max()))
    ref["steps"].append({"goal": u.goal.tolist(), "obs": obs["observation"].tolist(), "action": a.tolist(), "tip": obs["achieved_goal"].tolist()})
    obs, *_ = u.step(a)
print("max |numpy - SB3 action| =", worst)
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "policy.bin").write_bytes(b"".join(w.tobytes() + b.tobytes() for w, b in mats))
(OUT / "policy.json").write_text(json.dumps({"layers": [list(w.shape) for w, _ in mats], "scale": scale, "max_delta": cfg["env"]["max_delta"], "vel_scale": cfg["env"]["vel_scale"], "vel_clip": cfg["env"]["vel_clip"], "frame_skip": cfg["env"]["frame_skip"], "home": cfg["arm"]["home"]}))
(OUT / "ref.json").write_text(json.dumps(ref))
print([list(w.shape) for w, _ in mats], (OUT / "policy.bin").stat().st_size, "bytes")
