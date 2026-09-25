#!/usr/bin/env python3
"""Runs on Hilti SLAM Challenge 2021 sequences, scored with the official protocol of the challenge.

    python scripts/evaluate_hilti.py <reference> <run dir> [<run dir> ...] [--calib=<calibration.yaml>] [--out=<dir>]

reference: one of the challenge's ground-truth files, name unchanged (the suffix says which point it is):
           <seq>_pole.txt (pole tip, total station) · <seq>_prism.txt (prism) · <seq>_imu.txt (dense, IMU frame).
           Here: /home/photogrammetry/kiss_data/hilti_2021/<seq>/ground_truth/ (docs/datasets.md).
--calib:   calibration.yaml of the dataset (default /home/photogrammetry/kiss_data/hilti_2021/calibration.yaml,
           Hugging Face Hilti-Research/hilti-slam-challenge-2021, "Calibration V2 26.08.2021").

The official protocol (Hilti-Research/hilti-slam-challenge-2021, evaluation-evo/evaluation.py), reproduced step by step:
  1. the estimate is the trajectory of the IMU (the frame "imu" of calibration.yaml), TUM format;
  2. every estimated pose is multiplied by T_imu_ref of the reference type: pole tip / prism (fixed matrices in
     evaluation.py) or identity for *_imu.txt;
  3. estimate and reference associated by stamp with max_diff = 1 s (sparse references: the device stands still on each
     control point);
  4. SE(3) Umeyama alignment, no scale;
  5. APE of the translation: rmse, mean, median, std, min, max (the challenge reports these).
Our poses are of the Ouster frame os_sensor (frame_id of /os_cloud_node/points), at the instant each stands for
(evaluate_official.pose_times, #061): T_world_imu = T_world_os_sensor · (T_imu_os_sensor)^-1, with
T_imu_os_sensor = T_imu_os_lidar · T_os_lidar_os_sensor from calibration.yaml (quaternions x, y, z, w: the identity of
"imu" is [0, 0, 0, 1]).  The IMU trajectory is written to <out>/<run>_imu_tum.txt, so evaluation.py itself can be run
on it:   evaluation.py <out>/<run>_imu_tum.txt <reference>
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from evo.core import metrics, sync
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_official import pose_times, write_tum  # noqa: E402

CALIB = Path("/home/photogrammetry/kiss_data/hilti_2021/calibration.yaml")
MAX_DIFF = 1.0          # evaluation.py: sync.associate_trajectories(traj_ref, traj_est, max_diff=1)
# evaluation.py, apply_pole_tip_calibration: T_imu_ref by the suffix of the reference file.
T_IMU_REF = {
    "pole.txt": np.array([[0.176566, -0.984288, 0.00121622, -0.00938425],
                          [-0.984256, -0.17655, 0.00837907, -0.0148401],
                          [-0.0080327, -0.00267653, -0.999964, 1.66722],
                          [0, 0, 0, 1.0]]),
    "prism.txt": np.array([[0.176566, -0.984288, 0.00121622, -0.00594496],
                           [-0.984256, -0.17655, 0.00837907, -0.00721288],
                           [-0.0080327, -0.00267653, -0.999964, 0.272943],
                           [0, 0, 0, 1.0]]),
    "imu.txt": np.eye(4),
}


def extrinsic(sensor):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(sensor["extrinsics"]["quaternion"]).as_matrix()     # x, y, z, w
    T[:3, 3] = sensor["extrinsics"]["translation"]
    return T


def T_imu_os_sensor(calib=CALIB):
    s = yaml.safe_load(open(calib))["sensors"]
    assert s["os_lidar"]["parent"] == "imu" and s["os_sensor"]["parent"] == "os_lidar"
    return extrinsic(s["os_lidar"]) @ extrinsic(s["os_sensor"])


def evaluate_hilti(reference, run, calib=CALIB, out_dir=None):
    reference, run = Path(reference), Path(run)
    kind = reference.name.split("_")[-1].lower()
    if kind not in T_IMU_REF:
        raise SystemExit(f"{reference.name}: the name must end in _pole.txt, _prism.txt or _imu.txt (as released)")
    t, T_lidar, source = pose_times(run, "none")
    T_imu = T_lidar @ np.linalg.inv(T_imu_os_sensor(calib))          # the IMU trajectory the challenge asks for
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        write_tum(Path(out_dir) / f"{run.name}_imu_tum.txt", t, T_imu)
    return dict(exact_times=source == "exact", **score(reference, t, T_imu))


def score(reference, t, T_imu):
    """Steps 2-5 of the protocol on an IMU trajectory (times t, poses T_imu)."""
    reference = Path(reference)
    kind = reference.name.split("_")[-1].lower()
    est = PoseTrajectory3D(poses_se3=list(np.asarray(T_imu) @ T_IMU_REF[kind]), timestamps=np.asarray(t, dtype=np.float64))
    ref = file_interface.read_tum_trajectory_file(str(reference))
    ref_s, est_s = sync.associate_trajectories(ref, est, MAX_DIFF)
    est_s.align(ref_s, correct_scale=False, correct_only_scale=False)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref_s, est_s))
    stats = ape.get_all_statistics()
    return dict(kind=kind, n_ref=ref.num_poses, n_matched=ref_s.num_poses,
                max_dt=float(np.max(np.abs(ref_s.timestamps - est_s.timestamps))), **{k: float(v) for k, v in stats.items()})


def main():
    calib = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--calib=")), CALIB)
    out = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--out=")), None)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    print(f"reference {args[0]}  (Hilti 2021 protocol: IMU trajectory, T_imu_ref, max_diff {MAX_DIFF} s, SE(3), APE)\n")
    print(f"{'run':<18}{'type':>10}{'matched':>10}{'max |dt| s':>12}{'rmse m':>10}{'mean m':>10}{'median m':>10}{'max m':>10}")
    for r in args[1:]:
        v = evaluate_hilti(args[0], r, calib, out)
        print(f"{Path(r).name:<18}{v['kind']:>10}{v['n_matched']:>5}/{v['n_ref']:<4}{v['max_dt']:>12.3f}"
              f"{v['rmse']:>10.4f}{v['mean']:>10.4f}{v['median']:>10.4f}{v['max']:>10.4f}")


if __name__ == "__main__":
    main()
