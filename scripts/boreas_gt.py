#!/usr/bin/env python3
"""Boreas ground truth (applanix/lidar_poses.csv) as TUM in the LiDAR frame, for evaluate_official.py (#085).

    python scripts/boreas_gt.py <seq dir> <out gt.txt>

lidar_poses.csv: the post-processed GNSS/INS pose of the LiDAR at every scan time (GPSTime in microseconds = the scan file
name, mid-sweep), ENU position (easting, northing, altitude) and roll / pitch / heading.  Rotation LiDAR -> ENU:
R = (Rx(roll) . Ry(pitch) . Rz(heading))^T, here as transpose(scipy from_euler("xyz", [roll, pitch, heading])) - chosen
on the data, not assumed: with it the GNSS velocity seen in the LiDAR frame keeps one direction (spread 0.6 deg over 9703
moving samples, at 45.7 deg, cf. the 42.6 deg yaw of calib/T_applanix_lidar.txt); the other conventions spread 1-47 deg.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    seq, out = Path(sys.argv[1]), Path(sys.argv[2])
    g = np.genfromtxt(seq / "applanix" / "lidar_poses.csv", delimiter=",", skip_header=1)
    Rm = np.transpose(Rotation.from_euler("xyz", g[:, 7:10]).as_matrix(), (0, 2, 1))
    q = Rotation.from_matrix(Rm).as_quat()
    rows = np.column_stack([g[:, 0] * 1e-6, g[:, 1:4], q])
    np.savetxt(out, rows, fmt="%.9f")
    print(f"{out}: {len(rows)} poses, {rows[0, 0]:.3f} .. {rows[-1, 0]:.3f} s, "
          f"path {np.linalg.norm(np.diff(g[:, 1:4], axis=0), axis=1).sum():.1f} m")


if __name__ == "__main__":
    main()
