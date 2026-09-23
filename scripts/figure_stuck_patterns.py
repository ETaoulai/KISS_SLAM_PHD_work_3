#!/usr/bin/env python3
"""Σχήμα (#025): πανοράματα intensity με τις «κολλημένες» αντιστοιχίσεις (ίδιο σημείο στο πλαίσιο του σαρωτή).

Για τρεις σαρώσεις (ισόγειο, διάδρομος, όροφος): πάνω το πανόραμα intensity, κάτω η εικόνα απόστασης· σημεία SIFT που
αντιστοιχίστηκαν με την προηγούμενη σάρωση — κόκκινο: |p − q| < 5 cm (μοτίβο που ακολουθεί τον σαρωτή), πράσινο: τα άλλα.

    python scripts/figure_stuck_patterns.py
"""
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
import kiss_slam.intensity_deskew as I  # noqa: E402
from test_i3_scale_bias import read_raw  # noqa: E402

warnings.simplefilter("ignore")
SCANS = {"ισόγειο (σάρωση 320)": 320, "διάδρομος (σάρωση 450)": 450, "όροφος (σάρωση 1250)": 1250}
FIG = Path("docs/figures/i3_stuck_patterns.png")


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    need = {k for k in SCANS.values()} | {k - 1 for k in SCANS.values()}
    data = {}
    for k in range(max(need) + 1):
        scan = ds[k]
        if k in need:
            ok = ~np.isnan(scan[0]).any(axis=1) & (np.linalg.norm(scan[0], axis=1) > I.MIN_RANGE)
            s = tuple(a[ok] for a in scan)
            big, P, T, valid = I.panorama(*s)
            kps, desc = sift.detectAndCompute(big, None)
            data[k] = (big, P, T, valid, kps, desc)

    fig, axes = plt.subplots(2 * len(SCANS), 1, figsize=(15, 3.1 * len(SCANS) * 2 / 1.6))
    for i, (name, k) in enumerate(SCANS.items()):
        big1, P1, T1, v1, kp1, d1 = data[k - 1]; big2, P2, T2, v2, kp2, d2 = data[k]
        red, green = [], []
        for m, n in bf.knnMatch(d1, d2, k=2):
            if m.distance >= I.RATIO * n.distance:
                continue
            x = I.lookup(P1, T1, v1, kp1[m.queryIdx]); y = I.lookup(P2, T2, v2, kp2[m.trainIdx])
            if x is None or y is None:
                continue
            (red if np.linalg.norm(x[0] - y[0]) < 0.05 else green).append(kp2[m.trainIdx].pt)
        rng_ = np.linalg.norm(P2, axis=2); rng_[~v2] = np.nan
        ax = axes[2 * i]
        ax.imshow(big2, cmap="gray", aspect="auto", vmin=0, vmax=255)
        for pts, c in ((green, "lime"), (red, "red")):
            if pts:
                pts = np.array(pts); ax.scatter(pts[:, 0], pts[:, 1], s=14, facecolors="none", edgecolors=c, linewidths=1.2)
        ax.set_title(f"{name} — intensity · κόκκινο: «ίδιο σημείο» ({len(red)}) · πράσινο: κανονικές ({len(green)})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        ax = axes[2 * i + 1]
        ax.imshow(np.repeat(rng_, I.UP, axis=0), cmap="viridis", aspect="auto", vmin=0, vmax=15)
        if red:
            pts = np.array(red); ax.scatter(pts[:, 0], pts[:, 1], s=14, facecolors="none", edgecolors="red", linewidths=1.2)
        ax.set_title("απόσταση (0–15 m) · κάτω γραμμές = δάπεδο κοντά στον σαρωτή", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=80)
    print(f"→ {FIG}")


if __name__ == "__main__":
    main()
