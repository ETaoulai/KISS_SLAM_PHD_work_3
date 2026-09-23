#!/usr/bin/env python3
"""Runs of scripts/run_ncd.py against a ground truth, grouped by arm.

    python scripts/evaluate_ncd.py <ground truth> <run dir> [<run dir> ...] [--frame=ncd2020|ncd2021|spires|none]

ground truth: a Newer College 2020 sequence dir (its ground_truth/registered_poses.csv), or a file:
              "#sec,nsec,x,y,z,qx,qy,qz,qw" csv (2020, 2021) or TUM "t x y z qx qy qz qw" (2021 tum_format,
              Oxford Spires).
--frame: pose of the LiDAR in the ground-truth frame, applied as T_world_lidar = T_world_gt @ T:
         ncd2020  T_CL of kiss_icp's loader (the 2020 GT is a camera frame)   [default for a 2020 dir]
         ncd2021  T_base_os-sensor, os_imu_lidar_transforms.yaml of the 2021 release (0.001, 0, 0.091)
         spires   T_base_lidar of Oxford Spires (0, 0, 0.124, 180 deg about z; scripts/evaluate_gt.py)
         none     the ground truth is already the LiDAR                            [default for a file]

Each run dir is the out dir of one run_ncd.py call; its name up to the first "_" is the arm
(kiss_a -> kiss, sift_s2 -> sift).  The ground truth is interpolated at the scan times (SLERP) in
its own world frame, so z is up and the vertical error means height; each trajectory is aligned to
it rigidly (Umeyama, no scale).

Per run: ATE RMSE, RPE over 1 s (translation, rotation), path length vs GT, vertical RMSE, the KITTI
relative error (kiss_icp's sequence_error), ms/scan and image-motion failures from the log.  Per arm:
mean ± σ over its runs.  CLAUDE.md: ATE of one run is not a measurement (#037); read RPE and path
length first.
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum, rpe, umeyama_rigid  # noqa: E402


def se3(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


FRAMES = {
    "ncd2020": se3([-0.084, -0.025, 0.050], [0, 0, 0.924, 0.383]),
    "ncd2021": se3([0.001, 0.000, 0.091], [0, 0, 0, 1]),
    "none": np.eye(4),
}


def load_gt(path, frame):
    path = Path(path)
    if path.is_dir():
        path = path / "ground_truth" / "registered_poses.csv"
        frame = frame or "ncd2020"
    first = path.read_text().lstrip().split("\n", 1)[0]
    if "," in first:                                                    # #sec,nsec,x,y,z,qx,qy,qz,qw
        g = np.genfromtxt(path, delimiter=",", comments="#")
        t, T = g[:, 0] + g[:, 1] * 1e-9, np.array([se3(r[2:5], r[5:9]) for r in g])
    else:                                                               # TUM
        t, T = load_tum(path)
    frame = frame or "none"
    T = base_to_lidar(T) if frame == "spires" else T @ FRAMES[frame]
    return t, T, frame


def evaluate(gt_t, gt_T, run):
    from kiss_icp.metrics import sequence_error

    st, est = load_tum(find_tum(run))
    ok, gt = interpolate(gt_t, gt_T, st)
    est, st = est[ok], st[ok]
    kitti_t, _ = sequence_error(gt, est)
    est = umeyama_rigid(est[:, :3, 3], gt[:, :3, 3]) @ est
    err = np.linalg.norm(est[:, :3, 3] - gt[:, :3, 3], axis=1)
    rt, rr, _ = rpe(est, gt, st, 1.0)
    log = run.parent / f"{run.name}.log"
    log = log.read_text(errors="replace") if log.exists() else ""
    num = lambda pat: float(m[1]) if (m := re.search(pat, log)) else np.nan
    return dict(
        n=len(est), matched=ok.mean(),
        ate=np.sqrt((err ** 2).mean()), ate_max=err.max(),
        rpe_t=rt.mean() * 100, rpe_r=rr.mean(),
        path=np.linalg.norm(np.diff(est[:, :3, 3], axis=0), axis=1).sum(),
        gt_path=np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum(),
        z_rmse=np.sqrt(((est[:, 2, 3] - gt[:, 2, 3]) ** 2).mean()),
        kitti=kitti_t,
        ms=num(r"Average Runtime\s+([\d.]+)"),
        fail=num(r"image motion: \d+/\d+ scans \((\d+) fell back"),
    )


COLS = [("ate", "ATE [m]", "{:.3f}"), ("rpe_t", "RPE 1 s [cm]", "{:.2f}"), ("rpe_r", "RPE 1 s [deg]", "{:.3f}"),
        ("path", "path [m]", "{:.1f}"), ("z_rmse", "z RMSE [m]", "{:.3f}"), ("kitti", "KITTI [%]", "{:.2f}"),
        ("ms", "ms/scan", "{:.0f}"), ("fail", "image fails", "{:.0f}")]


def main():
    frame = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--frame=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gt_t, gt_T, frame = load_gt(args[0], frame)
    res = {Path(r).name: evaluate(gt_t, gt_T, Path(r)) for r in args[1:]}
    v0 = next(iter(res.values()))
    print(f"GT {args[0]} (frame {frame}): {len(gt_t)} poses, path over the scans {v0['gt_path']:.1f} m; "
          f"scans inside the GT span: {min(v['matched'] for v in res.values()) * 100:.1f} % or more\n")
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
