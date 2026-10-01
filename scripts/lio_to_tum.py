#!/usr/bin/env python3
"""A LiDAR-inertial reference run (COIN-LIO / FAST-LIO2, #083) as one of our run folders.

    python scripts/lio_to_tum.py <run dir> <name> <params yaml> [--extrinsic-key=mapping]
    python scripts/lio_to_tum.py <run dir> <name> none --topic=/robot/dlo/odom_node/odom --raw-scan      (DLO, #085)

<run dir>/odometry.bag holds the method's /Odometry (nav_msgs/Odometry), recorded by
/home/photogrammetry/baselines/lio_docker/run_in_container.sh.  FAST-LIO2 and COIN-LIO (built on it) estimate the IMU
("body") pose and publish it stamped at the END of each LiDAR scan (lidar_end_time).  Our evaluators expect the LiDAR's
pose, so each pose becomes  T_world_lidar = T_world_imu . T_imu_lidar  with T_imu_lidar = (extrinsic_R, extrinsic_T) of
the params file the method ran with (FAST-LIO convention: the LiDAR's pose in the IMU frame).  The stamp is the instant
the pose stands for, so the trajectory is written as *_poses_posetime_tum.txt (used as is by evaluate_official.pose_times)
and as *_poses_tum.txt.
DLO (#085, LiDAR only): its pose is the LiDAR's (params "none": no extrinsic), stamped with the scan header (sweep start) and
not deskewed, so --raw-scan writes config.yml deskew false and no posetime file: evaluate_official.pose_times then puts each
pose at the mean sweep time, as KISS without deskew / GenZ-ICP.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from rosbags.highlevel import AnyReader
from scipy.spatial.transform import Rotation


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run, name, params = Path(args[0]), args[1], Path(args[2])
    key = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--extrinsic-key=")), "mapping")
    topic = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--topic=")), "/Odometry")
    raw_scan = "--raw-scan" in sys.argv
    T_il = np.eye(4)
    if str(params) != "none":
        cfg = yaml.safe_load(open(params))[key]
        T_il[:3, :3] = np.array(cfg["extrinsic_R"], dtype=float).reshape(3, 3)
        T_il[:3, 3] = np.array(cfg["extrinsic_T"], dtype=float)

    rows = []
    with AnyReader([run / "odometry.bag"]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            p, q = m.pose.pose.position, m.pose.pose.orientation
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T[:3, 3] = [p.x, p.y, p.z]
            T = T @ T_il                                              # IMU pose -> LiDAR pose
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            rows.append([t, *T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat()])
    rows = np.array(rows)
    np.savetxt(run / f"{name}_poses_tum.txt", rows, fmt="%.9f")
    if raw_scan:
        (run / "config.yml").write_text("# lio_to_tum.py --raw-scan: poses of raw scans at the sweep start stamp\ndata:\n  deskew: false\n")
    else:
        np.savetxt(run / f"{name}_poses_posetime_tum.txt", rows, fmt="%.9f")
    print(f"{run}: {len(rows)} poses, {rows[0, 0]:.3f} .. {rows[-1, 0]:.3f} s; T_imu_lidar from {params} [{key}]")


if __name__ == "__main__":
    main()
