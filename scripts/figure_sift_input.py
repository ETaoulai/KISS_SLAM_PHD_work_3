#!/usr/bin/env python3
"""Σχήμα: η εικόνα που βλέπει πραγματικά το SIFT, και τα σημεία-κλειδιά που βρίσκει (#034).

Το πανόραμα 64×1024 μεγεθύνεται ×8 κατακόρυφα (512×1024) και αυτό δίνεται στον ανιχνευτή. Πάνω: όλη η εικόνα με τα
σημεία-κλειδιά. Κάτω: μεγέθυνση της ίδιας περιοχής με τους δύο τρόπους κατασκευής, στην πραγματική κλίμακα pixel.

    python scripts/figure_sift_input.py [σάρωση=450]
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
K = int(sys.argv[1]) if len(sys.argv) > 1 else 450
R0, C0, NR, NC = 48, 430, 10, 90          # περιοχή μεγέθυνσης: δακτύλιοι και στήλες του πανοράματος
FIG = Path("docs/figures/sift_input.png")


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    for k in range(K + 1):
        x = ds[k]
    ok = ~np.isnan(x[0]).any(axis=1) & (np.linalg.norm(x[0], axis=1) > I.MIN_RANGE)
    scan = tuple(a[ok] for a in x)
    sift = cv2.SIFT_create()
    imgs = {}
    for mode in ("splat", "raycast"):
        I.RENDER = mode
        big = I.panorama(*scan)[0]
        imgs[mode] = (big, sift.detect(big, None))
    I.RENDER = "splat"

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.25, 1.0])

    for row, mode in enumerate(("splat", "raycast")):
        big, kps = imgs[mode]
        a = fig.add_subplot(gs[row, :])
        a.imshow(big, cmap="gray", vmin=0, vmax=255, aspect="auto")
        p = np.array([k.pt for k in kps])
        a.scatter(p[:, 0], p[:, 1], s=6, facecolors="none", edgecolors="lime", linewidths=0.5)
        a.add_patch(plt.Rectangle((C0, R0 * I.UP), NC, NR * I.UP, fill=False, ec="red", lw=1.5))
        name = "κάθε σημείο στο πλησιέστερο pixel (σημερινό)" if mode == "splat" else "ακτίνα ανά pixel (ray casting)"
        a.set_title(f"Η εικόνα που δίνεται στο SIFT — {name}: 512 × 1024 pixel, {len(kps)} σημεία-κλειδιά (πράσινα)",
                    fontsize=10)
        a.set_xticks([]); a.set_yticks([])

    for col, mode in enumerate(("splat", "raycast")):
        big, kps = imgs[mode]
        a = fig.add_subplot(gs[2, col])
        a.imshow(big[R0 * I.UP:(R0 + NR) * I.UP, C0:C0 + NC], cmap="gray", vmin=0, vmax=255,
                 interpolation="nearest", aspect="auto", extent=[C0, C0 + NC, (R0 + NR) * I.UP, R0 * I.UP])
        p = np.array([k.pt for k in kps])
        m = (p[:, 0] >= C0) & (p[:, 0] < C0 + NC) & (p[:, 1] >= R0 * I.UP) & (p[:, 1] < (R0 + NR) * I.UP)
        a.scatter(p[m, 0], p[m, 1], s=60, facecolors="none", edgecolors="lime", linewidths=1.5)
        a.set_title(f"μεγέθυνση ({'σημερινό' if mode == 'splat' else 'ray casting'}): "
                    f"{NC} στήλες × {NR} δακτύλιοι, {m.sum()} σημεία-κλειδιά", fontsize=9)
        a.set_xlabel("στήλη (γωνία περιστροφής)"); a.set_ylabel("γραμμή εικόνας")

    fig.suptitle("Τι «βλέπει» ο ανιχνευτής χαρακτηριστικών: το πανόραμα intensity μεγεθυμένο ×8 κατακόρυφα",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=100)
    print(f"→ {FIG}")


if __name__ == "__main__":
    main()
