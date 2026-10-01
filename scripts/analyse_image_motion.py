#!/usr/bin/env python3
"""The image motion of every scan against the ground-truth sweep motion (#086): bias, scale, noise, per axis.

    python scripts/analyse_image_motion.py <run dir with image_motions.npz> <oracle_motion.npz>

image_motions.npz: written by any run since #086 (motion (N,4,4) as estimated, inliers); oracle_motion.npz: make_oracle_motion.py
(the ground-truth motion during each sweep).  Rotations as rotation vectors in the sensor frame (deg per sweep):
  error e = rotvec(D^T M)  (D ground truth, M image);  bias = mean(e) per axis;  scale = slope of M on D per axis (1 = right
  size);  noise = std(e) and the share of the error left after removing bias and scale;  translation the same in mm.
Then the error against inliers and against the rotation per sweep (the rotation speed).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    run, orc = Path(sys.argv[1]), sys.argv[2]
    f = sorted(run.glob("*/image_motions.npz"))[-1]
    d = np.load(f); M, n, scan = d["motion"], d["inliers"], d["scan"]
    D = np.load(orc)["motion"][scan]
    ok = np.isfinite(M).all(axis=(1, 2)) & np.isfinite(D).all(axis=(1, 2))
    M, D, n = M[ok], D[ok], n[ok]
    rm = np.degrees(R.from_matrix(M[:, :3, :3]).as_rotvec())
    rd = np.degrees(R.from_matrix(D[:, :3, :3]).as_rotvec())
    e = np.degrees(R.from_matrix(np.transpose(D[:, :3, :3], (0, 2, 1)) @ M[:, :3, :3]).as_rotvec())
    print(f"{run.name}: {ok.sum()} / {len(ok)} scans with both motions; inliers median {np.median(n):.0f}")
    print(f"  rotation per sweep (truth): |rd| median {np.median(np.linalg.norm(rd, axis=1)):.2f} deg;  error |e| median "
          f"{np.median(np.linalg.norm(e, axis=1)):.3f} deg, 90% {np.percentile(np.linalg.norm(e, axis=1), 90):.3f} deg")
    for i, ax in enumerate("xyz"):
        A = np.c_[rd[:, i], np.ones(len(rd))]
        (slope, icpt), *_ = np.linalg.lstsq(A, rm[:, i], rcond=None)
        resid = rm[:, i] - (slope * rd[:, i] + icpt)
        print(f"  rot {ax}: truth std {rd[:, i].std():.3f}  bias {e[:, i].mean():+.4f}  scale {slope:.3f}  error rms {np.sqrt((e[:, i]**2).mean()):.4f}"
              f"  after bias+scale {resid.std():.4f} deg")
    te = (np.transpose(D[:, :3, :3], (0, 2, 1)) @ (M[:, :3, 3] - D[:, :3, 3])[..., None])[..., 0] * 1000
    for i, ax in enumerate("xyz"):
        A = np.c_[D[:, i, 3], np.ones(len(D))]
        (slope, icpt), *_ = np.linalg.lstsq(A, M[:, i, 3], rcond=None)
        print(f"  trans {ax}: truth std {D[:, i, 3].std()*1000:.1f} mm  bias {te[:, i].mean():+.1f}  scale {slope:.3f}  error rms {np.sqrt((te[:, i]**2).mean()):.1f} mm")
    en = np.linalg.norm(e, axis=1); sp = np.linalg.norm(rd, axis=1)
    print("  rotation error by inliers:   " + "  ".join(f"{lo}-{hi}: {np.median(en[(n >= lo) & (n < hi)]):.3f} ({((n >= lo) & (n < hi)).sum()})"
                                                 for lo, hi in ((0, 50), (50, 100), (100, 200), (200, 400), (400, 10**6)) if ((n >= lo) & (n < hi)).sum() > 10))
    q = np.percentile(sp, [0, 25, 50, 75, 100])
    print("  rotation error by speed:     " + "  ".join(f"{q[i]:.1f}-{q[i+1]:.1f} deg: {np.median(en[(sp >= q[i]) & (sp <= q[i+1])]):.3f}" for i in range(4)))
    print(f"  relative error |e| / |rd| median {np.median(en / np.maximum(sp, 1e-3)):.2f}")


if __name__ == "__main__":
    main()
