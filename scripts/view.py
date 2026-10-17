"""Interactive MuJoCo viewer for the arm. Drag the green target: double-click it, then Ctrl+right-drag."""

import argparse

import mujoco
import mujoco.viewer

from rlarm.arm import load_model
from rlarm.common import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/arm.yaml")
    cfg = load_config(p.parse_args().config)
    m = load_model(cfg)
    d = mujoco.MjData(m)
    d.qpos[:] = d.ctrl[:3] = cfg["arm"]["home"]
    mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    main()
