"""Το deskew + αρχική θέση ICP από την κίνηση της εικόνας intensity ως ρύθμιση του KissSLAM (#018-#032).

    python tests/test_image_deskew.py

(a) image_deskew.enabled=False: η τροχιά ίδια με το γνήσιο upstream KissSLAM (original_slam_files/slam_original.py),
    σε 150 πραγματικές σαρώσεις.
(b) enabled=True + motion_file (runs/i3_motion_car_sp_st0.05_floor.npz): οι πρώτες 150 θέσεις ίδιες με το override
    register_frame του scripts/run_i3_deskew.py («0 init fixed»), που αναπαράγεται εδώ αυτολεξεί.
(c) online (motion_file=None), 60 σαρώσεις: οι κινήσεις που μετρά ο ενσωματωμένος εκτιμητής ίδιες με το npz του
    precompute_i3_motion.py --model=car --subpixel --stuck=0.05 --floor-only (ίδιος εκτιμητής, ίδιο seed).
"""
import warnings
from pathlib import Path

import numpy as np
warnings.simplefilter("ignore")

from kiss_icp.datasets import dataset_factory
from kiss_slam.config import load_config
from kiss_slam.original_slam_files.slam_original import KissSLAM as UpstreamKissSLAM
from kiss_slam.slam import KissSLAM
from kiss_slam.tools.point_cloud2 import read_point_cloud_raw
import kiss_slam.intensity_deskew as _idsk

# (c) compares with an npz made before fast_test: its fixed 400-hypothesis RANSAC (same random draws).
_idsk.RANSAC_CONF = None

N, N_ONLINE, TOL, TOL_MOTION = 150, 60, 1e-9, 1e-6
CFG, BAG = Path("configs/indoor_fast.yaml"), Path("data/church_02_cut.bag")
MOTION = "runs/i3_motion_car_sp_st0.05_floor.npz"


def dataset(raw=False):
    ds = dataset_factory(dataloader="rosbag", data_dir=BAG, sequence=None, topic="/hesai/pandar", meta=None)
    if raw:
        ds.read_point_cloud = read_point_cloud_raw
    return ds


def config(**image_deskew):
    cfg = load_config(CFG)
    for k, v in image_deskew.items():
        setattr(cfg.image_deskew, k, v)
    return cfg


class Upstream(UpstreamKissSLAM):
    """Το γνήσιο upstream process_scan· μόνο η κλήση του loop closer προσαρμόζεται στο τρέχον API του
    kiss_slam.loop_closer (λίστα δεκτών closures αντί για ένα 4-tuple), που άλλαξε σε αυτό το repo (#014)."""

    def compute_closures(self, query_id, query):
        accepted = self.closer.compute(query_id, query, self.local_map_graph)
        for source_id, target_id, pose_constraint in accepted:
            self.closures.append((source_id, target_id))
            self.optimizer.add_factor(source_id, target_id, pose_constraint, np.eye(6))
        if accepted:
            self.optimize_pose_graph()


def run_upstream():
    ds, slam = dataset(), Upstream(config())
    for i in range(N):
        slam.process_scan(*ds[i][:2])
    return np.array(slam.poses)


def run_disabled():
    ds, slam = dataset(), KissSLAM(config(enabled=False))
    for i in range(N):
        slam.process_scan(*ds[i][:2])
    return np.array(slam.poses)


def run_reference_override():
    """scripts/run_i3_deskew.py::register_frame με WARMUP=0, INIT=True, SIGMA="fixed" — αυτολεξεί."""
    motion = np.load(MOTION)["motion"]
    ok = ~np.isnan(motion[:, 0, 0])
    deltas = [motion[k] if ok[k] else np.eye(4) for k in range(len(motion))]
    ds, slam = dataset(), KissSLAM(config(enabled=False))
    odo = slam.odometry
    inv = np.linalg.inv
    k = {"i": 0}

    def register_frame(frame, timestamps):
        i = k["i"]; k["i"] += 1
        delta = deltas[i]
        frame = odo.preprocessor.preprocess(frame, timestamps, delta)
        source, frame_downsample = odo.voxelize(frame)
        sigma = odo.adaptive_threshold.get_threshold()
        initial_guess = odo.last_pose @ delta
        new_pose = odo.registration.align_points_to_map(
            points=source, voxel_map=odo.local_map, initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma, kernel=sigma,
        )
        odo.local_map.update(frame_downsample, new_pose)
        odo.last_delta = inv(odo.last_pose) @ new_pose
        odo.last_pose = new_pose
        return frame, source

    odo.register_frame = register_frame
    for i in range(N):
        slam.process_scan(*ds[i][:2])
    return np.array(slam.poses)


def run_motion_file():
    ds, slam = dataset(), KissSLAM(config(enabled=True, motion_file=MOTION))
    for i in range(N):
        slam.process_scan(*ds[i][:2])
    return np.array(slam.poses)


def run_online_motions():
    """Οι κινήσεις που χρησιμοποιεί ο ενσωματωμένος εκτιμητής (ταυτοτική όπου απέτυχε)."""
    ds, slam = dataset(raw=True), KissSLAM(config(enabled=True, motion_file=None))
    used, orig = [], slam._image_motion

    def recording(*args):
        M = orig(*args); used.append(np.eye(4) if M is None else M); return M   # None = failed (#054)

    slam._image_motion = recording
    for i in range(N_ONLINE):
        xyz, ts, inten, ring = ds[i]
        slam.process_scan(xyz, ts, inten, ring)
    return np.array(used), slam.n_image_motion_failures


upstream = run_upstream()
disabled = run_disabled()
d_a = np.abs(upstream - disabled).max()

reference = run_reference_override()
integrated = run_motion_file()
d_b = np.abs(reference - integrated).max()

online, n_fail = run_online_motions()
pre = np.load(MOTION)["motion"][:N_ONLINE]
pre_ok = ~np.isnan(pre[:, 0, 0])
pre = np.where(pre_ok[:, None, None], pre, np.eye(4))
d_c = np.abs(online - pre).max()

ok = d_a < TOL and d_b < TOL and d_c < TOL_MOTION and n_fail == int((~pre_ok).sum())
print(f"(a) enabled=False vs upstream KissSLAM      max|Δ| = {d_a:.1e}   {'PASS' if d_a < TOL else 'FAIL'}")
print(f"(b) motion_file vs run_i3_deskew override   max|Δ| = {d_b:.1e}   {'PASS' if d_b < TOL else 'FAIL'}")
print(f"(c) online motions vs precomputed npz       max|Δ| = {d_c:.1e}   {'PASS' if d_c < TOL_MOTION else 'FAIL'}"
      f"   (αποτυχίες {n_fail} online / {int((~pre_ok).sum())} npz σε {N_ONLINE} σαρώσεις)")
print("RESULT:", "PASS" if ok else "FAIL")
