#!/usr/bin/env python3
"""Βαθμονόμηση του intensity ως προς απόσταση και γωνία πρόσπτωσης, από τα ίδια τα δεδομένα (#025).

Το intensity = ανακλαστικότητα του υλικού × συνάρτηση της απόστασης × συνάρτηση της γωνίας πρόσπτωσης. Σε κοντινές
επιφάνειες η γεωμετρική εξάρτηση δημιουργεί μοτίβα που ακολουθούν τον σαρωτή (#024). Με πολλές σαρώσεις, η κατανομή
των υλικών είναι περίπου ίδια σε κάθε κελί (απόσταση, γωνία)· ο διάμεσος ανά κελί εκτιμά τη γεωμετρική εξάρτηση και
η διαίρεση με αυτόν αφήνει κάτι ανάλογο της ανακλαστικότητας.

Κάθε 20ή σάρωση → πίνακας διάμεσου intensity ανά (απόσταση: 24 λογαριθμικά κελιά 1–50 m) × (|cos πρόσπτωσης|: 10
κελιά + στήλη «όλες οι γωνίες», για σημεία χωρίς κάθετη). → runs/intensity_calib.npz, docs/figures/intensity_calibration.png

    python scripts/calibrate_intensity.py
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.datasets import dataset_factory

import kiss_slam.intensity_deskew as I
from kiss_slam.tools.point_cloud2 import read_points

warnings.simplefilter("ignore")
OUT, FIG, EVERY = Path("runs/intensity_calib.npz"), Path("docs/figures/intensity_calibration.png"), 20
R_EDGES = np.geomspace(1.0, 50.0, 25)
C_EDGES = np.linspace(0.0, 1.0, 11)


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "timestamp", "ring"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = ~np.isnan(xyz).any(axis=1) & (np.linalg.norm(xyz, axis=1) > I.MIN_RANGE)
    return xyz[ok], s["timestamp"].astype(np.float64)[ok], s["intensity"].astype(np.float64)[ok], s["ring"].astype(np.int64)[ok]


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    rs, cs, vs = [], [], []
    for k in range(0, len(ds), EVERY):
        img, P, _, valid, cos = I.grid(*ds[k], with_cos=True)
        m = valid
        rs.append(np.linalg.norm(P[m], axis=1)); cs.append(cos[m]); vs.append(img[m])
    r, c, v = np.concatenate(rs), np.concatenate(cs), np.concatenate(vs)
    nr, nc = len(R_EDGES) - 1, len(C_EDGES) - 1
    table = np.full((nr, nc + 1), np.nan); count = np.zeros((nr, nc + 1), int)
    ri = np.clip(np.searchsorted(R_EDGES, r) - 1, 0, nr - 1)
    ci = np.where(np.isnan(c), nc, np.clip(np.searchsorted(C_EDGES, np.nan_to_num(c)) - 1, 0, nc - 1))
    ci_all = ci.copy()
    for i in range(nr):
        for j in range(nc + 1):
            sel = (ri == i) & ((ci_all == j) if j < nc else np.ones_like(ri, bool))
            count[i, j] = sel.sum()
            if sel.sum() >= 200:
                table[i, j] = np.median(v[sel])
    # κενά κελιά: από τη στήλη «όλες οι γωνίες» της ίδιας απόστασης, αλλιώς ο γενικός διάμεσος
    for i in range(nr):
        fill = table[i, nc] if not np.isnan(table[i, nc]) else np.nanmedian(v)
        table[i, np.isnan(table[i])] = fill
    np.savez(OUT, r_edges=R_EDGES, c_edges=C_EDGES, table=table, count=count)
    print(f"{len(v)} σημεία από {len(rs)} σαρώσεις · άγνωστη γωνία {100*np.isnan(c).mean():.0f} % → {OUT}")
    rc = np.sqrt(R_EDGES[:-1] * R_EDGES[1:])
    print("\nδιάμεσο intensity ανά απόσταση (όλες οι γωνίες):")
    print("   " + "  ".join(f"{rc[i]:4.1f}m:{np.median(v[ri == i]) if (ri == i).any() else np.nan:5.1f}" for i in range(0, nr, 2)))
    print("διάμεσο intensity ανά |cos πρόσπτωσης| (απόσταση 2–6 m):")
    near = (r > 2) & (r < 6)
    print("   " + "  ".join(f"{(C_EDGES[j]+C_EDGES[j+1])/2:.2f}:{np.median(v[near & (ci == j)]):5.1f}" for j in range(nc)))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j in range(0, nc, 2):
        ax[0].plot(rc, table[:, j], label=f"|cos| {C_EDGES[j]:.1f}–{C_EDGES[j+1]:.1f}")
    ax[0].set_xscale("log"); ax[0].set_xlabel("απόσταση (m)"); ax[0].set_ylabel("διάμεσο intensity"); ax[0].legend(fontsize=7)
    im = ax[1].imshow(table[:, :nc], aspect="auto", origin="lower", extent=[0, 1, 0, nr])
    ax[1].set_yticks(np.arange(0, nr, 4) + 0.5); ax[1].set_yticklabels([f"{x:.1f}" for x in rc[::4]])
    ax[1].set_xlabel("|cos γωνίας πρόσπτωσης|"); ax[1].set_ylabel("απόσταση (m)"); fig.colorbar(im, ax=ax[1])
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=90)
    print(f"→ {FIG}")


if __name__ == "__main__":
    main()
