"""Render one frame of an env to a PNG (<=640 px)."""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco

from rlarm.arm import load_model
from rlarm.common import load_config, make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=480)
    a = p.parse_args()
    cfg, size = load_config(a.config), min(a.size, 640)
    if "arm" in cfg:  # raw arm model (Phase 1): home pose, target placed at the tip
        m = load_model(cfg)
        d = mujoco.MjData(m)
        d.qpos[:] = cfg["arm"]["home"]
        mujoco.mj_forward(m, d)
        d.mocap_pos[0] = d.site("tip").xpos + [0.08, 0.0, 0.08]
        mujoco.mj_forward(m, d)
        with mujoco.Renderer(m, size, size) as r:
            r.update_scene(d, camera="main")
            img = r.render()
    else:
        env = make_env(cfg, "rgb_array", width=size, height=size)
        env.reset(seed=0)
        img = env.render()
    imageio.imwrite(a.out, img)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
