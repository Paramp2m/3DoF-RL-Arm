"""Live demo: drag the green target and the trained hold policy follows it.

Window: double-click the green ball, then Ctrl+right-drag it. Close the window or Ctrl+C to stop.
--headless moves the target along a scripted path instead and prints the mean tip-target distance
over the second half (one JSON line). Settings: configs/demo.yaml.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def target_path(lim: np.ndarray, dt: float, steps: int, script: dict) -> np.ndarray:
    """(steps, 3) target positions: random joint-space waypoints (same distribution as training goals)
    joined by smooth moves at <= max_joint_speed, with a dwell at each; passed through FK so every
    point is reachable."""
    from rlarm.kinematics import fk

    rng = np.random.default_rng(script["seed"])
    dwell = max(1, round(script["dwell_s"] / dt))
    q = rng.uniform(lim[:, 0], lim[:, 1])
    qs = [q] * dwell
    while len(qs) < steps:
        nxt = rng.uniform(lim[:, 0], lim[:, 1])
        n = max(1, int(np.ceil(np.abs(nxt - q).max() / (script["max_joint_speed"] * dt))))
        s = (1 - np.cos(np.pi * np.arange(1, n + 1) / n)) / 2  # ease in/out
        qs += list(q + s[:, None] * (nxt - q)) + [nxt] * dwell
        q = nxt
    return np.array([fk(x) for x in qs[:steps]])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--headless", action="store_true", help="scripted target, no window")
    p.add_argument(
        "--steps", type=int, default=0, help="control steps (0 = until the window closes)"
    )
    p.add_argument("--gif", help="headless only: also save a GIF of the first steps here")
    a = p.parse_args()
    if a.headless and a.steps <= 0:
        p.error("--headless needs --steps > 0")
    os.environ["MUJOCO_GL"] = "egl" if a.headless else "glfw"  # a window needs GLFW

    from rlarm.common import apply_threads, load_policy, make_env

    demo = yaml.safe_load((ROOT / a.config).read_text())
    run = ROOT / demo["run"]
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    apply_threads({**cfg, "torch_threads": 1})
    model = load_policy(cfg, run / "model.zip")
    u = make_env(cfg, "rgb_array" if a.gif else None).unwrapped
    obs, _ = u.reset(seed=demo["script"]["seed"])
    dt = u.ecfg["frame_skip"] * u.model.opt.timestep
    targets = target_path(u.lim, dt, a.steps, demo["script"]) if a.headless else None

    viewer = None
    if not a.headless:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(u.model, u.data)
    dists, frames, t = [], [], 0
    try:
        while not a.steps or t < a.steps:
            t0 = time.perf_counter()
            if viewer is not None:
                if not viewer.is_running():
                    break
                with viewer.lock():
                    goal = u.data.mocap_pos[u.mocap].copy()  # where the user dragged it
            else:
                goal = targets[t]
                u.data.mocap_pos[u.mocap] = goal
            u.goal = goal
            obs = u._obs()  # re-read the goal before acting
            # episode truncation is ignored: the demo is one long episode
            obs, _, _, _, info = u.step(model.predict(obs, deterministic=True)[0])
            dists.append(info["distance"])
            if a.gif and t % demo["gif"]["stride"] == 0 and len(frames) < demo["gif"]["max_frames"]:
                frames.append(u.render())
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            t += 1
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        u.close()
    if frames:
        import imageio.v2 as imageio

        imageio.mimsave(a.gif, frames, fps=round(1 / (dt * demo["gif"]["stride"])), loop=0)
    d = np.array(dists) * 1000
    print(
        json.dumps(
            {
                "run": demo["run"],
                "steps": len(d),
                "mean_dist_mm": round(float(d.mean()), 2) if len(d) else None,
                "mean_dist_mm_second_half": round(float(d[len(d) // 2 :].mean()), 2)
                if len(d)
                else None,
            }
        )
    )


if __name__ == "__main__":
    main()
