#!/usr/bin/env python3
"""Γιατί ξεφεύγει η εκκίνηση με εξωτερικό deskew; (#030) — επανάληψη των πρώτων N σαρώσεων με διάφορες ρυθμίσεις.

Εκδοχές (όλες KISS indoor_detail, μόνο η αρχή):
  kiss        το deskew του KISS (last_delta) — αναφορά
  gt          deskew με την αληθινή κίνηση, σάρωση 0 χωρίς deskew (όπως το oracle run)
  gt0         όπως gt, ΚΑΙ η σάρωση 0 με την αληθινή κίνησή της              → υπόθεση 1 (στραβός αρχικός χάρτης)
  i3          deskew από την εικόνα, αρχική θέση ICP του KISS (όπως το run)
  i3_0        όπως i3, σάρωση 0 με την κίνηση της σάρωσης 1                    → υπόθεση 1
  i3_init     εικόνα ως deskew ΚΑΙ ως αρχική θέση του ICP (last_delta := M)    → υπόθεση 2/3 (ασυνέπεια deskew–αρχικής θέσης)
Μετρά το σωρευτικό σφάλμα προσανατολισμού έναντι GT και το τελικό σ.

    python scripts/replay_start.py <bag> <gt> <run για χρονοσφραγίδες> <motion.npz> [N=60]
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.config import load_config  # noqa: E402
from kiss_slam.slam import KissSLAM  # noqa: E402

warnings.simplefilter("ignore")
BAG, GT, RUN, MOTION = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 60
inv = np.linalg.inv
ang = lambda M: np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))


def main():
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, _ = load_tum(find_tum(RUN))
    _, G = interpolate(gs, gT, st - 0.010)                  # για τη σύγκριση (θέση στο τέλος της σάρωσης)
    _, G0 = interpolate(gs, gT, st)                         # για το deskew (τ = 0)
    _, Gm = interpolate(gs, gT, st - 0.100)                 # αρχή της σάρωσης 0
    M = np.load(MOTION)["motion"]
    img = [M[i] if not np.isnan(M[i, 0, 0]) else np.eye(4) for i in range(N)]
    true = [inv(Gm[0]) @ G0[0]] + [inv(G0[i - 1]) @ G0[i] for i in range(1, N)]
    modes = {
        "kiss": None,
        "gt": ("deskew", [np.eye(4)] + true[1:]),
        "gt0": ("deskew", true),
        "i3": ("deskew", img),
        "i3_0": ("deskew", [img[1]] + img[1:]),
        "i3_init": ("delta", img),
    }
    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG), sequence=None, topic="/hesai/pandar", meta=None)
    scans = [ds[i][:2] for i in range(N)]
    print(f"{Path(BAG).stem}: σωρευτικό σφάλμα προσανατολισμού έναντι GT (°) · σάρωση: "
          + " ".join(f"{k:4d}" for k in range(0, N, 6)) + "   | max   σ τέλος")
    for name, spec in modes.items():
        cfg = load_config(Path("configs/indoor_detail.yaml")); slam = KissSLAM(cfg); odo = slam.odometry
        if spec and spec[0] == "deskew":
            orig, k, seq = odo.preprocessor.preprocess, {"i": 0}, spec[1]
            def pre(frame, ts, delta, _o=orig, _s=seq, _k=k):
                i = _k["i"]; _k["i"] += 1
                return _o(frame, ts, _s[i])
            odo.preprocessor.preprocess = pre
        poses = []
        for i, (xyz, ts) in enumerate(scans):
            if spec and spec[0] == "delta":
                odo.last_delta = spec[1][i]                  # deskew ΚΑΙ αρχική θέση από την εικόνα
            slam.process_scan(xyz, ts); poses.append(odo.last_pose.copy())
        err = [ang(inv(inv(G[0]) @ G[i]) @ (inv(poses[0]) @ poses[i])) for i in range(N)]
        print(f"  {name:8s}" + " ".join(f"{err[k]:4.1f}" for k in range(0, N, 6))
              + f"   | {max(err):4.1f}  {odo.adaptive_threshold.get_threshold():.2f}")


if __name__ == "__main__":
    main()
