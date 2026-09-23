#!/usr/bin/env python3
"""#023: KISS-SLAM με deskew από την εικόνα intensity ΚΑΙ μετατόπιση ICP διορθωμένη από την εικόνα.

Ίδιο με το run_i3_deskew.py (deskew με την κίνηση της εικόνας, runs/i3_motion_car_sp.npz), με μία προσθήκη μετά τον
ICP: η μετατόπιση της σάρωσης (inv(last_pose)·new_pose) αναλύεται στις ιδιοκατευθύνσεις του πίνακα πληροφορίας
point-to-plane H = mean(n nᵀ) της σάρωσης· όπου η ιδιοτιμή < τ (η γεωμετρία δεν «κλειδώνει»), η συνιστώσα παίρνεται
από την κίνηση της εικόνας. Η στροφή μένει του ICP. τ ≥ 1: όλη η μετατόπιση από την εικόνα.

    python scripts/run_i4_fusion.py <tau> [out_dir]
"""
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

TAU = float(sys.argv[1])
OUT = sys.argv[2] if len(sys.argv) > 2 else f"runs/indoor_detail_i4_tau{TAU:g}"
CFG, MOTION = "configs/indoor_detail.yaml", "runs/i3_motion_car_sp.npz"
os.environ["KISS_SLAM_OUT_DIR"] = str(Path(OUT).resolve())

from kiss_icp.datasets import dataset_factory  # noqa: E402

from kiss_slam.pipeline import SlamPipeline  # noqa: E402


def weak_directions(frame):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(frame)).voxel_down_sample(0.25)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
    n = np.asarray(pc.normals)
    w, V = np.linalg.eigh(n.T @ n / len(n))
    return V[:, w < TAU]


def main():
    motion = np.load(MOTION)["motion"]
    ok = ~np.isnan(motion[:, 0, 0])
    deltas = [motion[k] if ok[k] else np.eye(4) for k in range(len(motion))]
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    pipe = SlamPipeline(dataset=ds, config_file=Path(CFG), use_intensity=False)
    odo = pipe.kiss_slam.odometry
    inv = np.linalg.inv
    k = {"i": 0, "fused": 0}

    def register_frame(frame, timestamps):
        """KissICP.register_frame (kiss_icp 1.3.0): deskew από την εικόνα + μετατόπιση ICP διορθωμένη από την εικόνα."""
        i = k["i"]; k["i"] += 1
        frame = odo.preprocessor.preprocess(frame, timestamps, deltas[i])
        source, frame_downsample = odo.voxelize(frame)
        sigma = odo.adaptive_threshold.get_threshold()
        initial_guess = odo.last_pose @ odo.last_delta
        new_pose = odo.registration.align_points_to_map(
            points=source, voxel_map=odo.local_map, initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma, kernel=sigma,
        )
        if ok[i] and i > 0:                                   # ← η προσθήκη
            delta = inv(odo.last_pose) @ new_pose
            V = weak_directions(frame)
            if V.shape[1]:
                delta[:3, 3] += V @ (V.T @ (deltas[i][:3, 3] - delta[:3, 3]))
                new_pose = odo.last_pose @ delta
                k["fused"] += 1
        odo.adaptive_threshold.update_model_deviation(inv(initial_guess) @ new_pose)
        odo.local_map.update(frame_downsample, new_pose)
        odo.last_delta = inv(odo.last_pose) @ new_pose
        odo.last_pose = new_pose
        return frame, source

    odo.register_frame = register_frame
    print(f"KissSLAM| Ι-4: deskew εικόνας + μετατόπιση ICP από την εικόνα όπου λ < {TAU} → {OUT}")
    pipe.run().print()
    print(f"KissSLAM| διορθώθηκαν {k['fused']} από {k['i']} σαρώσεις")


if __name__ == "__main__":
    main()
