"""Το deskew σε περισσότερα περάσματα (#017): με 1 πέρασμα είναι ακριβώς το KISS.

    python tests/test_deskew_refine.py

(a) _register_frame_deskew_refined με passes=1 == KissICP.register_frame, σε 150 πραγματικές σαρώσεις.
(b) με passes=2 η τροχιά αλλάζει (η αλλαγή όντως ενεργοποιείται) και μένει πεπερασμένη.
"""
import warnings
from pathlib import Path

import numpy as np
warnings.simplefilter("ignore")

from kiss_icp.datasets import dataset_factory
from kiss_slam.config import load_config
from kiss_slam.slam import KissSLAM

N, TOL = 150, 1e-9


def run(passes, force_refined):
    cfg = load_config(Path("configs/indoor_fast.yaml"))
    cfg.deskew_refine.passes = passes
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    slam = KissSLAM(cfg)
    if force_refined:
        slam.odometry.register_frame = slam._register_frame_deskew_refined   # και με passes=1
    for i in range(N):
        xyz, ts = ds[i][:2]
        slam.process_scan(xyz, ts)
    return np.array(slam.poses)


upstream = run(1, False)
refined1 = run(1, True)
refined2 = run(2, False)
d1 = np.abs(upstream - refined1).max()
d2 = np.abs(upstream - refined2).max()
ok = d1 < TOL and d2 > 1e-6 and np.isfinite(refined2).all()
print(f"(a) passes=1 vs KISS register_frame   max|Δ| = {d1:.1e}   {'PASS' if d1 < TOL else 'FAIL'}")
print(f"(b) passes=2 vs KISS                  max|Δ| = {d2:.1e}   {'PASS' if d2 > 1e-6 else 'FAIL'} (πρέπει να διαφέρει)")
print("RESULT:", "PASS" if ok else "FAIL")
