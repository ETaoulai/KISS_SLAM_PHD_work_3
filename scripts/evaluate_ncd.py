#!/usr/bin/env python3
"""Newer College 2020: runs of scripts/run_ncd.py against the ground truth, grouped by arm.

    python scripts/evaluate_ncd.py <sequence dir> <run dir> [<run dir> ...]

Each run dir is the out dir of one run_ncd.py call; its name up to the first "_" is the arm
(kiss_a -> kiss, sift_s2 -> sift).  The ground truth is ground_truth/registered_poses.csv in the
LiDAR frame (T_CL as kiss_icp) and in its OWN world frame, so z is up and the vertical error
means height; each trajectory is aligned to it rigidly (Umeyama, no scale).

Per run: ATE RMSE, RPE over 1 s (translation, rotation), path length vs GT, vertical RMSE,
the KITTI relative error the pipeline printed, and ms/scan.  Per arm: mean ± σ over its runs.
CLAUDE.md: ATE of one run is not a measurement (#037); read RPE and path length first.
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import find_tum, load_tum, rpe, umeyama_rigid  # noqa: E402


def gt_world(seq):
    g = np.genfromtxt(Path(seq) / "ground_truth" / "registered_poses.csv", delimiter=",", skip_header=1)
    T = np.tile(np.eye(4), (len(g), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(g[:, 5:9]).as_matrix()
    T[:, :3, 3] = g[:, 2:5]
    T_CL = np.eye(4)
    T_CL[:3, :3] = Rotation.from_quat([0, 0, 0.924, 0.383]).as_matrix()
    T_CL[:3, 3] = [-0.084, -0.025, 0.050]
    return g[:, 0] + g[:, 1] * 1e-9, T @ T_CL


def evaluate(gt_t, gt_T, run):
    st, est = load_tum(find_tum(run))
    k = np.clip(np.searchsorted(gt_t, st), 1, len(gt_t) - 1)
    k = np.where(np.abs(gt_t[k - 1] - st) < np.abs(gt_t[k] - st), k - 1, k)
    ok = np.abs(gt_t[k] - st) < 5e-3
    est, gt, st = est[ok], gt_T[k[ok]], st[ok]
    est = umeyama_rigid(est[:, :3, 3], gt[:, :3, 3]) @ est
    err = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1)
    rt, rr, _ = rpe(est, gt, st, 1.0)
    log = (run.parent / f"{run.name}.log").read_text(errors="replace") if (run.parent / f"{run.name}.log").exists() else ""
    num = lambda pat: float(m[1]) if (m := re.search(pat, log)) else np.nan
    return dict(
        n=len(est), matched=ok.mean(),
        ate=np.sqrt((err ** 2).mean()), ate_max=err.max(),
        rpe_t=rt.mean() * 100, rpe_r=rr.mean(),
        path=np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1).sum(),
        gt_path=np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum(),
        z_rmse=np.sqrt(((est[:, 2, 3] - gt[:, 2, 3]) ** 2).mean()),
        kitti=num(r"Average Translation Error\s+([\d.]+)"),
        ms=num(r"Average Runtime\s+([\d.]+)"),
        fail=num(r"image motion: \d+/\d+ scans \((\d+) fell back"),
    )


COLS = [("ate", "ATE [m]", "{:.3f}"), ("rpe_t", "RPE 1 s [cm]", "{:.2f}"), ("rpe_r", "RPE 1 s [deg]", "{:.3f}"),
        ("path", "path [m]", "{:.1f}"), ("z_rmse", "z RMSE [m]", "{:.3f}"), ("kitti", "KITTI [%]", "{:.2f}"),
        ("ms", "ms/scan", "{:.0f}"), ("fail", "image fails", "{:.0f}")]


def main():
    seq, runs = sys.argv[1], [Path(r) for r in sys.argv[2:]]
    gt_t, gt_T = gt_world(seq)
    res = {r.name: evaluate(gt_t, gt_T, r) for r in runs}
    gt_path = next(iter(res.values()))["gt_path"]
    print(f"GT path length {gt_path:.1f} m, {len(gt_t)} poses; matched scans per run: "
          f"{min(v['matched'] for v in res.values()) * 100:.1f} % or more\n")
    head = f"{'run':<10}" + "".join(f"{h:>15}" for _, h, _ in COLS)
    print(head)
    for name, v in res.items():
        print(f"{name:<10}" + "".join(f"{f.format(v[c]):>15}" for c, _, f in COLS))
    print("\nPer arm, mean ± σ over its runs:")
    print(head.replace("run ", "arm "))
    for arm in dict.fromkeys(n.split("_")[0] for n in res):
        vs = [v for n, v in res.items() if n.split("_")[0] == arm]
        cell = lambda c, f: (f.format(np.mean([v[c] for v in vs])) + (f" ±{f.format(np.std([v[c] for v in vs], ddof=1))}"
                             if len(vs) > 1 else ""))
        print(f"{arm + f' ({len(vs)})':<10}" + "".join(f"{cell(c, f):>15}" for c, _, f in COLS))


if __name__ == "__main__":
    main()
