#!/usr/bin/env python3
"""Runs against the ground truth with the official protocol (evo), every pose at the instant it stands for.

    python scripts/evaluate_official.py <ground truth> <run dir> [<run dir> ...] [--frame=ncd2020|ncd2021|spires|none]
                                        [--out=<dir>]

The protocol of the Oxford Spires localisation benchmark (scripts/localisation_benchmark/*.py of
ori-drs/oxford_spires_dataset):   evo_ape tum gt_lidar.txt <estimate>_tum.txt --align --t_max_diff 0.01
i.e. the translation APE (RMSE) after a rigid SE(3) Umeyama alignment, estimate and GT associated by stamp within
10 ms, no time offset.  Newer College papers report the same APE plus the evo RPE over 1 m.  Here, per run:

  ate     evo APE, translation, --align, t_max_diff 0.01                       [the official number]
  rpe_t   evo RPE, translation, delta 1 m, all pairs, pairs from the reference [cm]
  rpe_r   the same pairs, rotation angle                                        [deg]
  rpe1s_* the same relative error over 1 s (rpe_seconds: evo has no time step), as the RPE 1 s of #037-#060
  rte/rre the KITTI relative error over 100-800 m sub-trajectories, % and deg/100 m (#083); NaN below 100 m of path
  path    length of the associated estimate, gt_path of the associated GT; z_rmse after the alignment

The time offset (#045, #046).  Every stamp in the GT files is the instant of the pose.  The scan stamps of the Hesai and
Ouster bags and of the Newer College 2020 pcds are the FIRST point of the sweep (measured 25/9: min point time - stamp
= 0.0 ms, max = +99.9 ms / +98.3 ms), while a deskewed scan is expressed at its LAST point (kiss_icp 1.3.0), and a
scan registered raw (no deskew) near its mean point time.  So each pose gets its own time, from the run itself: no
search, nothing tuned on the metric.
  - runs with pose_times.csv (written since 25/9): its *_poses_posetime_tum.txt, exact per scan;
  - older runs: the recorded stamps + the convention of the arm (config.yml data.deskew, slam_config.yaml
    image_deskew.enabled, two_start.csv): deskewed scans +END, raw scans (deskew off, the first scans, the
    constant-velocity start kept by two_start) +MEAN; the image-motion failures of a run (identity deskew, count in
    its log) are not listed per scan and keep +END.
The estimate with these times is written to <out>/<run>_posetime_tum.txt and the GT, in the LiDAR frame, to
<out>/gt_lidar.txt, so every number here can be reproduced with the evo command line (printed with -v).

Which GT samples exist (measured 25/9).  Newer College: 10 Hz, on every scan stamp.  Oxford Spires: a 20 Hz grid plus
the stamps of ~77 % of the scans (church_03: 23 % of the scan stamps have no GT sample, the nearest is 17-24 ms away);
evo drops those poses, as in the benchmark (column "assoc.").  A pose at the mean point time (no deskew) falls between
the samples (~48 ms from any on Newer College 2021) and evo cannot associate it: when fewer than half of the poses
inside the GT span associate within 10 ms, the GT is interpolated (SLERP) at the pose times instead, and the run is
flagged (gt_interp = 1).  The GT has no holes to bridge: its longest gap is 0.3 s (underground_hard), 0.2 s
elsewhere.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import find_tum, interpolate, load_tum  # noqa: E402
from evaluate_ncd import load_gt  # noqa: E402

T_MAX_DIFF = 0.01       # evo_ape --t_max_diff of the Oxford Spires benchmark
MIN_ASSOCIATED = 0.5    # below this share the pose times fall between the GT samples: interpolate the GT (docstring)
# Offsets from the scan stamp (s) for runs without pose_times.csv, measured on the data 25/9:
# END = last point, MEAN = mean normalised time of the points in range x the sweep span.
SWEEP = {"spires": dict(end=0.0999, mean=0.0500),      # Hesai QT64, christ-church-03, 300 scans
         "ncd2021": dict(end=0.0999, mean=0.0475),     # Ouster OS0-128, math_easy
         "ncd2020": dict(end=0.0983, mean=0.0492),     # Ouster OS1-64, 01_short
         "none": dict(end=0.0999, mean=0.0500)}


def to_traj(t, T):
    return PoseTrajectory3D(poses_se3=list(T), timestamps=np.asarray(t, dtype=np.float64))


def write_tum(path, t, T):
    from scipy.spatial.transform import Rotation
    rows = [np.r_[s, M[:3, 3], Rotation.from_matrix(M[:3, :3]).as_quat()] for s, M in zip(t, T)]
    np.savetxt(path, np.array(rows), fmt="%.6f")


def _yaml(run, name):
    hits = sorted(Path(run).glob(f"**/{name}"))
    return (yaml.safe_load(open(hits[-1])) or {}) if hits else {}


def pose_times(run, frame):
    """(times, poses, source): the run's poses at the instant each stands for (see the module docstring)."""
    exact = sorted(Path(run).glob("**/*_poses_posetime_tum.txt"))
    if exact:
        t, T = load_tum(exact[-1])
        return t, T, "exact"
    stamps, T = load_tum(find_tum(run))
    kiss_cfg, slam_cfg = _yaml(run, "config.yml"), _yaml(run, "slam_config.yaml")
    deskew = bool(kiss_cfg.get("data", {}).get("deskew", True))
    image = bool(slam_cfg.get("image_deskew", {}).get("enabled", False))
    sweep = SWEEP[frame]
    off = np.full(len(stamps), sweep["end"] if (deskew or image) else sweep["mean"])
    off[: 1 if image else 2] = sweep["mean"]           # no motion yet: first scan (image) / first two (last_delta)
    two = sorted(Path(run).glob("**/two_start.csv"))
    if image and two:
        import csv
        for row in csv.DictReader(open(two[-1])):
            if row["kept"] == "cv" and int(row["scan"]) < len(off):
                off[int(row["scan"])] = sweep["mean"]
    return stamps + off, T, "arm"


def rpe_seconds(ref, est, delta=1.0, tol=0.05):
    """RPE over `delta` seconds on associated trajectories: (translation m, rotation deg), means over the pairs.

    evo has no time unit for the RPE step (frames, m, deg, rad), and "10 frames" is not 1 s where evo dropped poses
    (Oxford Spires: ~77 % associate).  Pairs: every pose i with the pose j nearest to t_i + delta, kept if
    |t_j - t_i - delta| <= tol * delta (all pairs, like evo --all_pairs); error E = (Q_i^-1 Q_j)^-1 (P_i^-1 P_j) as evo's
    RPE, Q the reference, P the estimate.
    """
    t = np.asarray(est.timestamps)
    P, Q = np.array(est.poses_se3), np.array(ref.poses_se3)
    k = np.searchsorted(t, t + delta).clip(1, len(t) - 1)
    j = np.where(np.abs(t[k - 1] - t - delta) < np.abs(t[k] - t - delta), k - 1, k)
    i = np.arange(len(t))
    keep = (j > i) & (np.abs(t[j] - t[i] - delta) <= tol * delta)
    i, j = i[keep], j[keep]
    E = np.linalg.inv(np.linalg.inv(Q[i]) @ Q[j]) @ (np.linalg.inv(P[i]) @ P[j])
    cos = np.clip((np.trace(E[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    return float(np.linalg.norm(E[:, :3, 3], axis=1).mean()), float(np.degrees(np.arccos(cos)).mean())


def evaluate_official(gt_t, gt_T, run, frame, out_dir=None, verbose=False):
    run = Path(run)
    t, T, source = pose_times(run, frame)
    ref, est = to_traj(gt_t, gt_T), to_traj(t, T)
    inside = (t >= gt_t[0] - T_MAX_DIFF) & (t <= gt_t[-1] + T_MAX_DIFF)
    try:
        ref_a, est_a = sync.associate_trajectories(ref, est, max_diff=T_MAX_DIFF)
        gt_interp = est_a.num_poses < MIN_ASSOCIATED * inside.sum()
    except sync.SyncException:                          # not one pose within 10 ms of a GT sample
        gt_interp = True
    if gt_interp:
        ok, G = interpolate(gt_t, gt_T, t)
        ref_a, est_a = to_traj(t[ok], G), to_traj(t[ok], T[ok])
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tum(out_dir / f"{run.name}_posetime_tum.txt", t, T)
        if not (out_dir / "gt_lidar.txt").exists():
            write_tum(out_dir / "gt_lidar.txt", gt_t, gt_T)
        if verbose:
            print(f"  evo_ape tum {out_dir / 'gt_lidar.txt'} {out_dir / f'{run.name}_posetime_tum.txt'} "
                  f"--align --t_max_diff {T_MAX_DIFF}" + ("   (GT interpolated here: evo_ape alone would drop poses)"
                                                         if gt_interp else ""))

    def rpe(relation, delta, unit, all_pairs, from_ref):
        m = metrics.RPE(relation, delta=delta, delta_unit=unit, all_pairs=all_pairs, pairs_from_reference=from_ref)
        m.process_data((ref_a, est_a))
        return float(np.mean(m.error))

    tr, rot = metrics.PoseRelation.translation_part, metrics.PoseRelation.rotation_angle_deg
    rpe_t = rpe(tr, 1.0, metrics.Unit.meters, True, True)
    rpe_r = rpe(rot, 1.0, metrics.Unit.meters, True, True)
    rpe1s_t, rpe1s_r = rpe_seconds(ref_a, est_a, 1.0)
    # RTE / RRE (#083): the KITTI odometry metric most LiDAR-odometry papers report (KISS-ICP, CT-ICP, Traj-LO) - the relative
    # translation error in % and rotation error in deg per 100 m, averaged over all sub-trajectories of 100, 200, ... 800 m
    # (kiss_icp.metrics.sequence_error, start every 10 poses), on the same associated pose pairs.  Undefined below 100 m of path.
    rte = rre = float("nan")
    if ref_a.path_length >= 100.0:
        from kiss_icp.metrics import sequence_error
        rte, rre = sequence_error(np.array(ref_a.poses_se3), np.array(est_a.poses_se3))
        rre *= 100.0                                     # deg/m -> deg/100 m

    est_al = PoseTrajectory3D(poses_se3=list(est_a.poses_se3), timestamps=est_a.timestamps)
    est_al.align(ref_a, correct_scale=False)            # evo_ape --align: Umeyama SE(3)
    ape = metrics.APE(tr)
    ape.process_data((ref_a, est_al))
    z = est_al.positions_xyz[:, 2] - ref_a.positions_xyz[:, 2]
    return dict(
        n=est_a.num_poses, matched=est_a.num_poses / max(inside.sum(), 1), gt_interp=float(gt_interp),
        exact_times=float(source == "exact"),
        ate=float(ape.get_statistic(metrics.StatisticsType.rmse)),
        rpe_t=rpe_t * 100, rpe_r=rpe_r, rpe1s_t=rpe1s_t * 100, rpe1s_r=rpe1s_r,
        rte=float(rte), rre=float(rre),
        path=float(est_a.path_length), gt_path=float(ref_a.path_length),
        z_rmse=float(np.sqrt((z ** 2).mean())),
        pose_minus_stamp=float(np.median(t - load_tum(find_tum(run))[0])),
    )


COLS = [("ate", "APE [m]", "{:.3f}"), ("rpe_t", "RPE 1 m [cm]", "{:.2f}"), ("rpe_r", "RPE 1 m [deg]", "{:.3f}"),
        ("rpe1s_t", "RPE 1 s [cm]", "{:.2f}"), ("rpe1s_r", "RPE 1 s [deg]", "{:.3f}"), ("rte", "RTE [%]", "{:.2f}"),
        ("rre", "RRE [deg/100m]", "{:.3f}"), ("path", "path [m]", "{:.1f}"),
        ("z_rmse", "z RMSE [m]", "{:.3f}"), ("matched", "assoc.", "{:.3f}"), ("gt_interp", "GT interp", "{:.0f}"),
        ("pose_minus_stamp", "pose-stamp [s]", "{:+.4f}")]


def main():
    frame = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--frame=")), None)
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    gt_t, gt_T, frame = load_gt(args[0], frame)
    res = {Path(r).name: evaluate_official(gt_t, gt_T, Path(r), frame, out, "-v" in sys.argv) for r in args[1:]}
    print(f"\nGT {args[0]} (frame {frame}): {len(gt_t)} poses; evo, t_max_diff {T_MAX_DIFF}, SE(3) alignment\n")
    head = f"{'run':<16}" + "".join(f"{h:>16}" for _, h, _ in COLS)
    print(head)
    for name, v in res.items():
        print(f"{name:<16}" + "".join(f"{f.format(v[c]):>16}" for c, _, f in COLS))


if __name__ == "__main__":
    main()
