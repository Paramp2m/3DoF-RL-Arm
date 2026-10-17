"""Render one frame of an env to a PNG (<=640 px)."""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio

from rlarm.common import load_config, make_env


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=480)
    a = p.parse_args()
    env = make_env(
        load_config(a.config), "rgb_array", width=min(a.size, 640), height=min(a.size, 640)
    )
    env.reset(seed=0)
    imageio.imwrite(a.out, env.render())
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
