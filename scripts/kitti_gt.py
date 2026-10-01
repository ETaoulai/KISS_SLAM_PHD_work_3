#!/usr/bin/env python3
"""KITTI odometry ground truth in the Velodyne frame, as TUM with times, for evaluate_official.py (#084).

    python scripts/kitti_gt.py <kitti root> <odometry seq> <raw drive dir (sync)> <first raw frame> <out gt.txt>

Odometry sequence 07 = raw drive 2011_09_30_0027, sync frames 0-1100.  The odometry poses P_i are those of the rectified
left camera (cam0) in the frame of the first; the Velodyne pose is  T_i = Tr^-1 . P_i . Tr  with Tr = R_rect_00 . T_cam0_velo
(raw calib_velo_to_cam.txt, calib_cam_to_cam.txt: the odometry calib.txt Tr).  Each pose is stamped with the sync scan's
timestamps.txt: the camera trigger, the laser facing forward - mid-sweep, the instant the ground truth stands for.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kiss_slam.tools.kitti_raw import _read_stamps  # noqa: E402


def calib(path):
    d = {}
    for line in open(path):
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                d[k.strip()] = np.array([float(x) for x in v.split()])
            except ValueError:
                pass
    return d


def main():
    root, seq, drive, first, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
    date = drive.parent
    vc, cc = calib(date / "calib_velo_to_cam.txt"), calib(date / "calib_cam_to_cam.txt")
    T_cv = np.eye(4)
    T_cv[:3, :3], T_cv[:3, 3] = vc["R"].reshape(3, 3), vc["T"]
    R_rect = np.eye(4)
    R_rect[:3, :3] = cc["R_rect_00"].reshape(3, 3)
    Tr = R_rect @ T_cv
    P = np.loadtxt(root / "dataset" / "poses" / f"{seq}.txt").reshape(-1, 3, 4)
    P = np.concatenate([P, np.tile([[[0, 0, 0, 1]]], (len(P), 1, 1))], axis=1)
    T = np.linalg.inv(Tr) @ P @ Tr
    t = _read_stamps(drive / "velodyne_points" / "timestamps.txt")[first:first + len(P)]
    rows = [[ti, *Ti[:3, 3], *Rotation.from_matrix(Ti[:3, :3]).as_quat()] for ti, Ti in zip(t, T)]
    np.savetxt(out, np.array(rows), fmt="%.9f")
    print(f"{out}: {len(rows)} poses, {t[0]:.3f} .. {t[-1]:.3f} s, path {np.linalg.norm(np.diff(T[:, :3, 3], axis=0), axis=1).sum():.1f} m")


if __name__ == "__main__":
    main()
