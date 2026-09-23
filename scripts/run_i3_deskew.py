#!/usr/bin/env python3
"""Ι-3 (#019): KISS-SLAM με deskew από την κίνηση που μετρά η εικόνα intensity — όλα τα άλλα ίδια.

Ίδια δομή με το run_oracle_deskew.py (#012), με MIA διαφορά: η κίνηση της σάρωσης k για το deskew δεν έρχεται από
το GT αλλά από το runs/i3_motion.npz (precompute_i3_motion.py: ζεύγος (k−1, k), αιτιακά, χωρίς GT). Όπου η εκτίμηση
απέτυχε: κανένα deskew (ταυτοτική), που είναι καλύτερο από τη μαντεψιά του KISS (#012).

    python scripts/run_i3_deskew.py [config] [out_dir] [motion.npz, default runs/i3_motion.npz] [bag] [warmup] [init]
    warmup: οι πρώτες N σαρώσεις με το deskew του ίδιου του KISS (μπάλωμα, #029 — δεν είναι λύση)
    init:   η κίνηση της εικόνας ΚΑΙ ως αρχική θέση του ICP (last_pose·M), όχι μόνο ως deskew (#030: το deskew και η
            αρχική θέση πρέπει να συμφωνούν, αλλιώς η εκκίνηση ξεφεύγει)
    σ (7ο όρισμα, #031): "adaptive" (default, KISS: από την απόκλιση αρχικής θέσης–αποτελέσματος — με init πέφτει 2.4 → 0.6)·
            "kiss": η απόκλιση μετριέται έναντι της πρόβλεψης σταθερής ταχύτητας του KISS (σ όπως χωρίς init)·
            "fixed": σ = αρχική τιμή (2.0), χωρίς προσαρμογή
    curve (8ο όρισμα, #033): deskew κάθε σημείου με τη θέση του σαρωτή τη δική του στιγμή πάνω στην καμπύλη
            (params/t_start του npz), αντί για την ομοιόμορφη κατανομή του delta από τον KISS
"""
import os
import sys
from pathlib import Path

import numpy as np

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/indoor_detail.yaml"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/indoor_detail_i3_deskew"
MOTION = sys.argv[3] if len(sys.argv) > 3 else "runs/i3_motion.npz"
BAG = sys.argv[4] if len(sys.argv) > 4 else "data/church_02_cut.bag"
WARMUP = int(sys.argv[5]) if len(sys.argv) > 5 else 0
INIT = len(sys.argv) > 6 and sys.argv[6] == "init"
SIGMA = sys.argv[7] if len(sys.argv) > 7 else "adaptive"
CURVE = len(sys.argv) > 8 and sys.argv[8] == "curve"
os.environ["KISS_SLAM_OUT_DIR"] = str(Path(OUT).resolve())

from kiss_icp.datasets import dataset_factory  # noqa: E402

from kiss_slam.pipeline import SlamPipeline  # noqa: E402


def main():
    npz = np.load(MOTION); motion = npz["motion"]
    ok = ~np.isnan(motion[:, 0, 0])
    if CURVE:
        from kiss_slam.intensity_deskew import deskew_curve
        params, t_start = npz["params"], npz["t_start"]
    deltas = [motion[k] if ok[k] else np.eye(4) for k in range(len(motion))]
    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG),
                         sequence=None, topic="/hesai/pandar", meta=None)
    pipe = SlamPipeline(dataset=ds, config_file=Path(CFG), use_intensity=False)
    odo = pipe.kiss_slam.odometry
    inv = np.linalg.inv
    k = {"i": 0}
    pre_cfg = pipe.slam_config.odometry.preprocessing

    def register_frame(frame, timestamps):
        """KissICP.register_frame (kiss_icp 1.3.0) με ΜΙΑ αλλαγή: το deskew παίρνει την κίνηση της εικόνας."""
        i = k["i"]; k["i"] += 1
        delta = odo.last_delta if i < WARMUP else deltas[i]
        if CURVE and ok[i] and i >= WARMUP:
            x = params[i][~np.isnan(params[i])]
            frame = deskew_curve(frame, timestamps, x, t_start[i], 0.1, pre_cfg.min_range, pre_cfg.max_range)
        else:
            frame = odo.preprocessor.preprocess(frame, timestamps, delta)   # ← η αλλαγή: deskew από την εικόνα
        source, frame_downsample = odo.voxelize(frame)
        sigma = odo.adaptive_threshold.get_threshold()
        initial_guess = odo.last_pose @ (delta if INIT else odo.last_delta)   # ← με "init": ίδια κίνηση και για τον ICP
        new_pose = odo.registration.align_points_to_map(
            points=source, voxel_map=odo.local_map, initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma, kernel=sigma,
        )
        if SIGMA == "adaptive":
            odo.adaptive_threshold.update_model_deviation(inv(initial_guess) @ new_pose)
        elif SIGMA == "kiss":
            odo.adaptive_threshold.update_model_deviation(inv(odo.last_pose @ odo.last_delta) @ new_pose)
        # "fixed": καμία ενημέρωση → get_threshold() επιστρέφει την αρχική τιμή
        odo.local_map.update(frame_downsample, new_pose)
        odo.last_delta = inv(odo.last_pose) @ new_pose
        odo.last_pose = new_pose
        return frame, source

    odo.register_frame = register_frame
    print(f"KissSLAM| Ι-3 DESKEW: κίνηση από την εικόνα intensity ({ok.sum()}/{len(ok)} σαρώσεις)"
          f"{' + αρχική θέση ICP' if INIT else ''}{f', προθέρμανση {WARMUP}' if WARMUP else ''}, σ {SIGMA}{', deskew με καμπύλη' if CURVE else ''} → {OUT}")
    pipe.run().print()


if __name__ == "__main__":
    main()
