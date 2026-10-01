"""KITTI raw Velodyne HDL-64E drives (sync .bin or extract .txt) with intensity, ring and per-point time (#084).

    <drive>/velodyne_points/data/<10 digits>.{bin,txt}       x y z reflectance
    <drive>/velodyne_points/timestamps_start.txt / _end.txt   sweep start (laser facing backwards) / end (backwards again)

KITTI stores no ring and no point time, which the intensity panorama needs (as ncd_pcd.py).  Both are recovered:
- ring: the points are stored laser by laser, top laser (+2 deg) first, each laser sweeping its azimuth from the front
  (checked on drive 0027: yaw step +0.18 deg, elevation +2.1 -> -24 deg over the file).  A new laser starts where the
  azimuth, counted 0-360 deg from the front, falls back by more than 180 deg.  Ring 0 = top laser, as the Ouster.
- time: the HDL-64E turns clockwise and a sweep starts and ends facing backwards (KITTI raw devkit), so the fraction of
  the sweep is (180 deg - yaw) / 360 deg - kiss_icp's KITTIRawDataset.get_timestamps - and the absolute point time is
  start + fraction * (end - start), with the sweep's own measured start and end.
- intensity: reflectance 0-1; the panorama clips at 255, so run_ncd.py uses --intensity-scale=255 for KITTI.

The sync scans (= KITTI odometry; odometry sequence 07 = drive 2011_09_30_0027, frames 0-1100) and the extract scans are
the SAME raw sweeps, point for point (checked on drive 0027: identical clouds, 0.0 mm; extract only adds a few frames at the
ends and stores text).  Neither is motion compensated: KISS's own deskew improves both alike (#084).
Ground truth: scripts/kitti_gt.py (odometry poses in the Velodyne frame).
- correct=True (#085): the vertical-angle correction of the KITTI Velodyne intrinsics that IMLS-SLAM, CT-ICP and KISS-ICP apply
  (each point turned upwards about the horizontal axis perpendicular to it, 0.205 deg in kiss_icp's _correct_kitti_scan, the
  same call as kiss_icp's KITTI loaders).  Azimuth, hence ring and time, unchanged.
"""
from datetime import datetime
from pathlib import Path

import numpy as np


def _read_stamps(path):
    def sec(line):
        day, frac = line.strip().split(".")
        return datetime.strptime(day, "%Y-%m-%d %H:%M:%S").timestamp() + float("0." + frac)
    return np.array([sec(l) for l in open(path) if l.strip()])


def rings(xyz):
    """Ring index of every point (0 = top laser) from the laser-by-laser order of a KITTI scan."""
    a = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0            # azimuth from the front, 0-360
    starts = np.flatnonzero(np.diff(a) < -180.0) + 1                     # the next laser starts again at the front
    ring = np.zeros(len(xyz), dtype=np.int64)
    ring[starts] = 1
    return np.cumsum(ring)


class KittiRaw:
    def __init__(self, data_dir, first=0, last=None, correct=False):
        self.data_dir = Path(data_dir)
        self.correct = correct
        self.sequence_id = self.data_dir.name
        v = self.data_dir / "velodyne_points"
        files = sorted((v / "data").glob("*.bin")) or sorted((v / "data").glob("*.txt"))
        start, end = _read_stamps(v / "timestamps_start.txt"), _read_stamps(v / "timestamps_end.txt")
        sl = slice(first, last)
        self.scan_files, self.start, self.end = files[sl], start[sl], end[sl]
        self.stamps = self.start                                           # the scan stamp = sweep start, as the Ouster
        self.use_global_visualizer = True

    def __len__(self):
        return len(self.scan_files)

    def get_frames_timestamps(self):
        return self.stamps

    def __getitem__(self, idx):
        f = self.scan_files[idx]
        p = np.fromfile(f, np.float32).reshape(-1, 4).astype(np.float64) if f.suffix == ".bin" else np.loadtxt(f, ndmin=2)
        xyz, refl = p[:, :3], p[:, 3]
        ring = rings(xyz)
        frac = np.clip(0.5 * (1.0 - np.arctan2(xyz[:, 1], xyz[:, 0]) / np.pi), 0.0, 1.0)   # (180 - yaw) / 360
        t = self.start[idx] + frac * (self.end[idx] - self.start[idx])
        if self.correct:
            from kiss_icp.pybind import kiss_icp_pybind
            xyz = np.asarray(kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(xyz)))
        keep = np.linalg.norm(xyz, axis=1) > 0.0
        return xyz[keep], t[keep], refl[keep], ring[keep]
