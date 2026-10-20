import sys

import numpy as np

from rlarm.common import ROOT, load_config

sys.path.insert(0, str(ROOT / "scripts"))
from imitate import collect


def test_collect_does_not_depend_on_worker_count():
    cfg = load_config(ROOT / "configs/obstacle_h2.yaml")
    jobs = [(100000 + i, 1.0, i == 0, 10) for i in range(3)]
    one, two = collect(cfg, None, jobs, 1), collect(cfg, None, jobs, 2)
    for (x1, y1, tr1, s1), (x2, y2, tr2, s2) in zip(one, two, strict=True):
        assert np.array_equal(np.asarray(y1), np.asarray(y2)) and s1 == s2 and len(x1) == len(x2)
    assert one[0][2] is not None and one[1][2] is None  # transitions only where keep
