"""Newer College 2020 (Ouster OS1-64) scans as .pcd, with intensity, ring and per-point time.

    <seq>/raw_format/ouster_scan/cloud_<sec>_<nsec>.pcd     (unzipped raw_format/ouster_zip_files)
    <seq>/ground_truth/registered_poses.csv                  (#sec,nsec,x,y,z,qx,qy,qz,qw)

kiss_icp's NewerCollegeDataset reads only x, y, z (pyntcloud) and makes up the point times from
the 64 x 1024 layout.  The image-motion deskew (`image_deskew`) needs what the files carry:
intensity, ring and the measured time `t` of every point.  `__getitem__` returns the 4-tuple
(xyz, timestamps, intensity, ring) that SlamPipeline._next and KissSLAM.process_scan take.

- timestamps: ABSOLUTE, scan stamp (file name) + `t` (ns from the start of that sweep), in
  seconds.  intensity_deskew fits one motion over two consecutive scans and needs the previous
  scan's points to be earlier than the current one's; with `t` alone both scans run 0-0.1 s and
  every motion came out wrong (~100 deg, ~17 m per scan).  KISS normalises the times itself.
- intensity: the Ouster signal runs 0 to ~1100 (median 150-450 on 01_short), while the
  intensity panorama clips at 255 (it was built for the Hesai 0-255 scale).  Multiplied by one
  fixed `intensity_scale` (default 255/1024) for every scan: a change of units, no per-scan
  normalisation.
- points without a return (range 0, xyz = 0) are dropped.
- gt_poses: one per scan (the file has a pose at every scan time), moved from the camera frame
  to the LiDAR frame with the same T_CL as kiss_icp's loader and expressed relative to the
  first one, so the pipeline's own evaluation (ATE, KITTI relative error) runs.
"""
import os
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

_PCD_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"), ("t", "<u4"),
    ("reflectivity", "<u2"), ("ring", "u1"), ("noise", "<u2"), ("range", "<u4"),
])
_NAME = re.compile(r"^cloud_(\d+)_(\d+)\.pcd$")


def read_pcd(path):
    """Structured array of one binary Ouster .pcd (fields as in _PCD_DTYPE)."""
    raw = Path(path).read_bytes()
    header_end = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = raw[:header_end].decode()
    fields = header.split("FIELDS", 1)[1].split("\n", 1)[0].split()
    if fields != list(_PCD_DTYPE.names):
        raise ValueError(f"{path}: unexpected PCD fields {fields}")
    n = int(header.split("POINTS", 1)[1].split("\n", 1)[0])
    return np.frombuffer(raw, _PCD_DTYPE, count=n, offset=header_end)


class NewerCollege2020Pcd:
    def __init__(self, data_dir, intensity_scale=255.0 / 1024.0):
        self.data_dir = Path(data_dir)
        self.sequence_id = self.data_dir.name
        self.intensity_scale = intensity_scale
        scan_dir = self.data_dir / "raw_format" / "ouster_scan"
        names = [f for f in os.listdir(scan_dir) if _NAME.match(f)]
        stamp = lambda f: (int(_NAME.match(f)[1]), int(_NAME.match(f)[2]))
        self.scan_files = [scan_dir / f for f in sorted(names, key=stamp)]
        self.stamps = np.array([s + ns * 1e-9 for s, ns in map(stamp, sorted(names, key=stamp))])
        gt_file = self.data_dir / "ground_truth" / "registered_poses.csv"
        if gt_file.exists():
            self.gt_poses = self._load_gt(gt_file)
        self.use_global_visualizer = True

    def _load_gt(self, gt_file):
        g = np.genfromtxt(gt_file, delimiter=",", skip_header=1)
        t = g[:, 0] + g[:, 1] * 1e-9
        k = np.clip(np.searchsorted(t, self.stamps), 1, len(t) - 1)
        k = np.where(np.abs(t[k - 1] - self.stamps) < np.abs(t[k] - self.stamps), k - 1, k)
        if np.abs(t[k] - self.stamps).max() > 1e-3:
            raise ValueError(f"{gt_file}: some scans have no ground-truth pose at their time")
        poses = np.tile(np.eye(4), (len(k), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(g[k, 5:9]).as_matrix()      # x, y, z, w
        poses[:, :3, 3] = g[k, 2:5]
        T_CL = np.eye(4)                                                    # as kiss_icp datasets/ncd.py
        T_CL[:3, :3] = Rotation.from_quat([0, 0, 0.924, 0.383]).as_matrix()
        T_CL[:3, 3] = [-0.084, -0.025, 0.050]
        poses = poses @ T_CL
        return np.linalg.inv(poses[0]) @ poses

    def __len__(self):
        return len(self.scan_files)

    def get_frames_timestamps(self):
        return self.stamps

    def __getitem__(self, idx):
        a = read_pcd(self.scan_files[idx])
        a = a[a["range"] > 0]
        xyz = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float64)
        return (xyz, self.stamps[idx] + a["t"].astype(np.float64) * 1e-9,
                a["intensity"].astype(np.float64) * self.intensity_scale, a["ring"].astype(np.int64))
