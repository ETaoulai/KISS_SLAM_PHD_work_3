#!/usr/bin/env python3
"""Ground-truth deskew motion per scan, for the oracle-deskew diagnostic (#086) - NOT a method, it uses the ground truth.

    python scripts/make_oracle_motion.py <ground truth> <frame> <reference run dir> <out.npz>

For every scan of the reference run (its pose_times.csv: stamp = first point, span_s = sweep length) the motion during
that sweep, D_k = G(start_k)^-1 . G(end_k), with G the ground truth in the LiDAR frame as evaluate_official.load_gt gives it
(SLERP / linear interpolation).  Scans outside the ground truth get NaN (run_ncd.py --oracle-deskew keeps the method's
own deskew there).  The same scans in the same order as any run of the sequence (the readers are deterministic).
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import interpolate  # noqa: E402
from evaluate_official import load_gt  # noqa: E402


def main():
    gt, frame, run, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    gt_t, gt_T, frame = load_gt(gt, frame)
    rows = list(csv.DictReader(open(sorted(run.glob("*/pose_times.csv"))[-1])))
    start = np.array([float(r["stamp"]) for r in rows])
    end = start + np.array([float(r["span_s"]) for r in rows])
    ok0, G0 = interpolate(gt_t, gt_T, start)
    ok1, G1 = interpolate(gt_t, gt_T, end)
    motion = np.full((len(rows), 4, 4), np.nan)
    both = ok0 & ok1
    D = np.linalg.inv(G0[both[ok0]]) @ G1[both[ok1]]
    motion[both] = D
    np.savez(out, motion=motion)
    rot = np.degrees(np.arccos(np.clip((np.trace(D[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
    print(f"{out}: {both.sum()} / {len(rows)} scans with a ground-truth motion; per-sweep rotation median {np.median(rot):.2f} deg, "
          f"translation median {np.median(np.linalg.norm(D[:, :3, 3], axis=1)):.3f} m")


if __name__ == "__main__":
    main()
