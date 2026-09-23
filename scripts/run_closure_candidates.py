#!/usr/bin/env python3
"""Διαγνωστικό run: καταγράφει ΟΛΟΥΣ τους υποψηφίους loop closure του ανιχνευτή κατόψεων.

Το KISS ζητά από το MapClosures μόνο τον καλύτερο υποψήφιο (get_best_closure) και τον
επαληθεύει με ICP. Εδώ ζητάμε τους K καλύτερους (get_top_k_closures), ΚΡΑΤΑΜΕ τη συμπεριφορά
ίδια (επαληθεύεται μόνο ο πρώτος, με τα περισσότερα inliers) και γράφουμε όλους σε CSV:
query, υποψήφιος, inliers. Έτσι φαίνεται αν ο ανιχνευτής προτείνει χάρτες άλλου ορόφου.

    python scripts/run_closure_candidates.py [config] [out_dir] [K]
"""
import csv
import os
import sys
from pathlib import Path

import numpy as np

CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/indoor_detail.yaml"
OUT = sys.argv[2] if len(sys.argv) > 2 else "runs/indoor_detail_closure_candidates"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 5
os.environ["KISS_SLAM_OUT_DIR"] = str(Path(OUT).resolve())

from kiss_icp.datasets import dataset_factory  # noqa: E402

from kiss_slam.pipeline import SlamPipeline  # noqa: E402


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    pipe = SlamPipeline(dataset=ds, config_file=Path(CFG), use_intensity=False)
    closer = pipe.kiss_slam.closer
    log = []

    def compute(query_id, points, local_map_graph):
        """LoopCloser.compute με get_top_k αντί για get_best — ίδια απόφαση, πλήρης καταγραφή."""
        cands = sorted(closer.detector.get_top_k_closures(query_id, points, K),
                       key=lambda c: -c.number_of_inliers)
        for rank, c in enumerate(cands):
            log.append((query_id, rank, c.source_id, c.target_id, c.number_of_inliers))
        if cands and cands[0].number_of_inliers >= closer.config.detector.inliers_threshold:
            best = cands[0]
            source = local_map_graph[best.source_id].pcd
            target = local_map_graph[query_id].pcd
            print("\nKissSLAM| Closure Detected")
            is_good, pose_constraint = closer.validate_closure(source, target, best.pose)
            if is_good:
                return [(best.source_id, query_id, pose_constraint)]
        return []

    closer.compute = compute
    pipe.run().print()
    path = Path(OUT) / "closure_candidates.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["query", "rank", "source_id", "target_id", "inliers"]); w.writerows(log)
    print(f"KissSLAM| υποψήφιοι → {path} ({len(log)} γραμμές)")


if __name__ == "__main__":
    main()
