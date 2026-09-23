#!/usr/bin/env python3
"""Διαγνωστικό run: KISS-SLAM με deskew από την ΠΡΑΓΜΑΤΙΚΗ κίνηση (GT) — όλα τα άλλα ίδια.

Το KISS κάνει deskew κάθε σάρωσης με τη δική του εκτίμηση της κίνησης της ΠΡΟΗΓΟΥΜΕΝΗΣ
σάρωσης (last_delta). Αν εκείνη έχει σφάλμα, η σάρωση στρεβλώνεται, ο ICP δίνει νέο σφάλμα,
που τροφοδοτεί το επόμενο deskew — πιθανός φαύλος κύκλος και πηγή του τρέμουλου (#011).

Εδώ αλλάζει ΜΟΝΟ η κίνηση που δίνεται στο deskew: inv(G[k−1])·G[k] από το GT (μετατόπιση
χρόνου −70 ms, #010). Η πρόβλεψη του ICP (last_pose·last_delta), ο χάρτης, το κατώφλι σ και
όλα τα υπόλοιπα μένουν ως έχουν. ΔΕΝ είναι μέθοδος — χρησιμοποιεί το GT· είναι έλεγχος αιτίας
και ανώτατο όριο του τι θα κέρδιζε ένα καλύτερο deskew (π.χ. από το IMU του bag).

    python scripts/run_oracle_deskew.py [config] [out_dir] [offset_ms, default -70] [bag] [gt] [run για χρονοσφραγίδες]
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/indoor_detail.yaml"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/indoor_detail_oracle_deskew"
OFFSET = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else -0.070   # ms στη γραμμή εντολών
BAG = sys.argv[4] if len(sys.argv) > 4 else "data/church_02_cut.bag"
GT = sys.argv[5] if len(sys.argv) > 5 else "gt/church_02_gt-tum.txt"
STAMPS_FROM = sys.argv[6] if len(sys.argv) > 6 else "runs/indoor_detail_base_overlapfix"   # ίδιες σαρώσεις, ίδιες χρονοσφραγίδες
os.environ["KISS_SLAM_OUT_DIR"] = str(Path(OUT).resolve())

from kiss_icp.datasets import dataset_factory  # noqa: E402

from kiss_slam.pipeline import SlamPipeline  # noqa: E402


def main():
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, _ = load_tum(find_tum(STAMPS_FROM))
    ok, G = interpolate(gs, gT, st + OFFSET)
    assert ok.all(), "το GT πρέπει να καλύπτει όλες τις σαρώσεις"
    inv = np.linalg.inv
    true_delta = [np.eye(4)] + list(inv(G[:-1]) @ G[1:])

    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG),
                         sequence=None, topic="/hesai/pandar", meta=None)
    pipe = SlamPipeline(dataset=ds, config_file=Path(CFG), use_intensity=False)
    odo = pipe.kiss_slam.odometry
    k = {"i": 0}

    def register_frame(frame, timestamps):
        """KissICP.register_frame (kiss_icp 1.3.0) με ΜΙΑ αλλαγή: το deskew παίρνει την αληθινή κίνηση."""
        i = k["i"]; k["i"] += 1
        frame = odo.preprocessor.preprocess(frame, timestamps, true_delta[i])   # ← η μόνη αλλαγή
        source, frame_downsample = odo.voxelize(frame)
        sigma = odo.adaptive_threshold.get_threshold()
        initial_guess = odo.last_pose @ odo.last_delta
        new_pose = odo.registration.align_points_to_map(
            points=source, voxel_map=odo.local_map, initial_guess=initial_guess,
            max_correspondance_distance=3 * sigma, kernel=sigma,
        )
        odo.adaptive_threshold.update_model_deviation(inv(initial_guess) @ new_pose)
        odo.local_map.update(frame_downsample, new_pose)
        odo.last_delta = inv(odo.last_pose) @ new_pose
        odo.last_pose = new_pose
        return frame, source

    odo.register_frame = register_frame
    print(f"KissSLAM| ORACLE DESKEW: deskew από GT (μετατόπιση {OFFSET*1000:+.0f} ms) → {OUT}")
    pipe.run().print()


if __name__ == "__main__":
    main()
