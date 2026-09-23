#!/usr/bin/env python3
"""Έχουν οι επίπεδες επιφάνειες «σχέδιο» στο intensity; Ανά τμήμα της διαδρομής.

Μια κοινή λύση ICP + intensity (π.χ. ColoredICP) κλειδώνει την ολίσθηση πάνω σε
ένα επίπεδο μόνο αν το intensity ΜΕΤΑΒΑΛΛΕΤΑΙ πάνω στο επίπεδο. Εδώ μετράμε αυτή
τη μεταβολή, με το ακατέργαστο intensity (0–255, ίδια κλίμακα σε όλες τις
σαρώσεις — όχι η κανονικοποίηση ανά σάρωση του pipeline).

Για κάθε δείγμα σημείου σε απόσταση < R_MAX:
  - k γείτονες → PCA → κρατιέται μόνο αν η γειτονιά είναι ΕΠΙΠΕΔΗ και καλύπτει
    και τις δύο διευθύνσεις του επιπέδου (όχι μόνο μία γραμμή σάρωσης)·
  - σ_I   = τυπική απόκλιση του intensity στη γειτονιά (αντίθεση)·
  - |∇I|  = μέτρο της κλίσης του intensity ΠΑΝΩ στο επίπεδο (ελάχιστα τετράγωνα,
            όπως ορίζεται στο Colored ICP, Park et al. ICCV 2017), σε μονάδες/m.

Επιπλέον γράφει πανοραμικές εικόνες απόστασης και intensity (δακτύλιος × αζιμούθιο)
για αντιπροσωπευτικές σαρώσεις, στο docs/figures/.

    python scripts/analyze_intensity_texture.py
"""
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.datasets import dataset_factory
from scipy.spatial import cKDTree

from kiss_slam.tools.point_cloud2 import read_points

warnings.simplefilter("ignore")

BAG = Path("data/church_02_cut.bag")
FIG = Path("docs/figures/intensity_panoramas.png")
R_MAX, K, N_QUERY, STEP = 15.0, 24, 4000, 10

# Τμήματα από το προφίλ ύψους του GT (docs/experiment_log.md, #008)
SEGMENTS = {
    "ισόγειο": (300, 700),
    "ΣΚΑΛΑ πάνω": (820, 990),
    "επάνω όροφος": (1200, 1800),
    "ΣΚΑΛΑ κάτω": (2014, 2258),
}
PANORAMAS = {940: "ΣΚΑΛΑ πάνω", 2050: "ΣΚΑΛΑ κάτω", 1500: "επάνω όροφος"}


def read_raw(msg):
    """xyz, ακατέργαστο intensity (0–255), δακτύλιος."""
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "ring"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = ~np.isnan(xyz).any(axis=1)
    return xyz[ok], s["intensity"].astype(np.float64)[ok], s["ring"].astype(np.int64)[ok]


def texture(xyz, inten, rng):
    """(σ_I, |∇I|) στις επίπεδες γειτονιές μιας σάρωσης."""
    near = np.linalg.norm(xyz, axis=1) < R_MAX
    pts, I = xyz[near], inten[near]
    q = rng.choice(len(pts), min(N_QUERY, len(pts)), replace=False)
    dist, nn = cKDTree(pts).query(pts[q], k=K)
    Q = pts[nn] - pts[nn].mean(axis=1, keepdims=True)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q) / K)   # w αύξουσες
    normal = v[:, :, 0]
    flat = w[:, 0] / w.sum(axis=1) < 0.01          # επίπεδη γειτονιά
    spans_2d = w[:, 1] / w[:, 2] > 0.05             # όχι μόνο μία γραμμή σάρωσης
    compact = dist[:, -1] < 0.6                     # αρκετά πυκνή
    keep = flat & spans_2d & compact

    Qt = Q - np.einsum("nki,ni->nk", Q, normal)[:, :, None] * normal[:, None, :]
    dI = I[nn] - I[nn].mean(axis=1, keepdims=True)
    A = np.einsum("nki,nkj->nij", Qt, Qt) + 1e-6 * np.einsum("ni,nj->nij", normal, normal)
    b = np.einsum("nki,nk->ni", Qt, dI)
    grad = np.linalg.solve(A[keep], b[keep][:, :, None])[:, :, 0]
    return I[nn][keep].std(axis=1), np.linalg.norm(grad, axis=1), keep.mean()


def panorama(xyz, value, ring, n_az=1024):
    """Εικόνα δακτύλιος × αζιμούθιο· οι δακτύλιοι ταξινομούνται κατά γωνία ανύψωσης."""
    elev = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
    rings = np.unique(ring)
    order = rings[np.argsort([-elev[ring == r].mean() for r in rings])]   # πάνω δακτύλιος πρώτος
    pos = np.empty(rings.max() + 1, dtype=np.int64)
    pos[order] = np.arange(len(order))
    row = pos[ring]
    col = ((np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi) * n_az).astype(int) % n_az
    img = np.full((len(rings), n_az), np.nan)
    img[row, col] = value
    return img


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=BAG, sequence=None,
                         topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    rng = np.random.default_rng(0)
    stats = {name: [] for name in SEGMENTS}
    pano = {}
    last = max(max(b for _, b in SEGMENTS.values()), max(PANORAMAS))
    for i in range(last + 1):
        xyz, inten, ring = ds[i]
        if i in PANORAMAS:
            pano[i] = (panorama(xyz, np.linalg.norm(xyz, axis=1), ring),
                       panorama(xyz, inten, ring))
        for name, (a, b) in SEGMENTS.items():
            if a <= i < b and (i - a) % STEP == 0:
                stats[name].append(texture(xyz, inten, rng))

    print(f"Ακατέργαστο intensity (0–255), επίπεδες γειτονιές σε < {R_MAX:.0f} m, κάθε {STEP}η σάρωση\n")
    print(f"{'τμήμα':15}{'σαρώσεις':>9}{'επίπεδα':>9}{'σ_I διάμ.':>11}{'σ_I p90':>9}"
          f"{'|∇I| διάμ.':>12}{'|∇I| p90':>10}{'  «με σχέδιο»':>14}")
    for name, rows in stats.items():
        s = np.concatenate([r[0] for r in rows]); g = np.concatenate([r[1] for r in rows])
        frac = np.mean([r[2] for r in rows])
        print(f"{name:15}{len(rows):>9}{100 * frac:>8.1f}%{np.median(s):>11.1f}{np.percentile(s, 90):>9.1f}"
              f"{np.median(g):>12.1f}{np.percentile(g, 90):>10.1f}{100 * np.mean(s > 10):>13.1f}%")
    print("\n«με σχέδιο» = ποσοστό επίπεδων γειτονιών με σ_I > 10 (στην κλίμακα 0–255).")

    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2 * len(pano), 1, figsize=(12, 2.2 * 2 * len(pano)))
    for k, (i, (rimg, iimg)) in enumerate(sorted(pano.items())):
        for j, (img, lbl, cmap, lim) in enumerate(((rimg, "απόσταση (m)", "viridis", (0, 25)),
                                                   (iimg, "intensity (0–255)", "gray", (0, 255)))):
            ax = axes[2 * k + j]
            im = ax.imshow(img, aspect="auto", cmap=cmap, vmin=lim[0], vmax=lim[1], interpolation="nearest")
            ax.set_title(f"σάρωση {i} — {PANORAMAS[i]} — {lbl}", fontsize=9, loc="left")
            ax.set_yticks([]); ax.set_xticks([])
            fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    fig.tight_layout()
    fig.savefig(FIG, dpi=90)
    print(f"\nεικόνες → {FIG}")


if __name__ == "__main__":
    main()
