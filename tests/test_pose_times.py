"""The instant of the sweep each pose stands for (KissSLAM._pose_time_fraction), without any dataset.

    python tests/test_pose_times.py

(a) A scan deskewed with a motion is expressed at its last point: kiss_icp's own preprocessor, given a pure
    translation v over the sweep, leaves the point measured at the LAST instant where it was and moves the one of
    the first instant by -v.  So the fraction is 1.
(b) No motion (deskew off / identity): the fraction is the mean normalised time of the points inside the range crop,
    in the reader's unit (s or ns alike).
"""
import sys
from types import SimpleNamespace

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.preprocess import get_preprocessor

from kiss_slam.slam import KissSLAM


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    return ok


def main():
    results = []
    # (a) kiss_icp 1.3.0: which instant a deskewed scan is expressed at.
    cfg = KISSConfig()
    cfg.data.deskew, cfg.data.min_range, cfg.data.max_range = True, 0.0, 100.0
    pre = get_preprocessor(cfg)
    pts = np.array([[10.0, 0, 0], [0, 10.0, 0], [-10.0, 0, 0]])
    ts = np.array([0.0, 0.05, 0.1])
    delta = np.eye(4)
    delta[0, 3] = 1.0                                      # 1 m along x over the sweep
    out = pre.preprocess(pts, ts, delta)
    results.append(check("deskew keeps the last point in place", np.allclose(out[2], pts[2]), f"{out[2]}"))
    results.append(check("deskew moves the first point by -v", np.allclose(out[0], pts[0] - [1, 0, 0]), f"{out[0]}"))

    # The fraction itself, on a stub carrying only what _pose_time_fraction reads.
    stub = SimpleNamespace(odometry=SimpleNamespace(config=SimpleNamespace(data=SimpleNamespace(min_range=1.0, max_range=50.0))))
    f = lambda frame, t, d: KissSLAM._pose_time_fraction(stub, frame, t, d)
    rng = np.random.default_rng(0)
    n = 10000
    frame = rng.normal(size=(n, 3)) * 10
    t_s = np.sort(rng.uniform(0, 0.1, n))
    results.append(check("(a) deskewed with a motion -> 1", f(frame, t_s, delta) == 1.0))
    results.append(check("(b) deskew off -> mean time", abs(f(frame, t_s, None) - 0.5) < 0.02, f"{f(frame, t_s, None):.3f}"))
    results.append(check("(b) identity motion = deskew off", f(frame, t_s, np.eye(4)) == f(frame, t_s, None)))
    t_ns = (t_s - t_s[0]) * 1e9                            # Ouster bag: ns from the first point
    results.append(check("(b) same fraction in ns", abs(f(frame, t_ns, None) - f(frame, t_s, None)) < 1e-9))
    near_late = frame.copy()                               # points measured late all out of range -> earlier mean
    near_late[t_s > 0.05] = 0.0
    results.append(check("(b) only points inside the range crop count", f(near_late, t_s, None) < 0.3,
                         f"{f(near_late, t_s, None):.3f}"))
    print("\nRESULT:", "PASS" if all(results) else "FAIL")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
