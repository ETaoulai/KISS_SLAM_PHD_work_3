#!/usr/bin/env python3
"""Πού και γιατί ξεφεύγει το ύψος; (#022)

Για κάθε σάρωση: (α) πόση λάθος κατακόρυφη κίνηση προσθέτει η odometry — Δz της εκτίμησης μείον Δz του GT, μετά
από ευθυγράμμιση Umeyama όλης της τροχιάς· το αθροιστικό του είναι το σφάλμα ύψους· (β) το περιβάλλον από την
ακατέργαστη σάρωση: σημεία πάνω από τον αισθητήρα κοντά (ταβάνι), σημεία εδάφους κάτω από τον αισθητήρα, διάμεση
απόσταση· (γ) πόσο στρίβει και πόσο γέρνει η μονάδα (GT). Τυπώνει ανά παράθυρο και σώζει διάγραμμα.

    python scripts/analyze_height_error.py [run ...] [--from=0] [--to=1100]
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
RUNS = args or ["runs/indoor_detail_i3car_sp_deskew", "runs/indoor_detail_nodeskew", "runs/indoor_detail_oracle_deskew_t0"]
A = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--from=")), 0))
B = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--to=")), 1100))
FIG = Path("docs/figures/height_error_ground_floor.png")
WIN = 25


def umeyama(P, Q):
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    return R, mq - R @ mp


def read_xyz(msg):
    s = read_points(msg, field_names=["x", "y", "z"])
    x = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    return x[~np.isnan(x).any(axis=1)]


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    errs = {}
    for run in RUNS:
        off = -0.055 if "nodeskew" in run else -0.010
        st, E = load_tum(find_tum(run)); _, G = interpolate(gs, gT, st + off)
        R, t = umeyama(E[:, :3, 3], G[:, :3, 3])
        errs[run] = (E[:, :3, 3] @ R.T + t)[:, 2] - G[:, 2, 3]
    _, G = interpolate(gs, gT, st - 0.010)

    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_xyz
    env = np.full((B - A, 3), np.nan)
    for k in range(B):
        x = ds[k]
        if k < A:
            continue
        # στο πλαίσιο βαρύτητας (στροφή GT), ώστε «πάνω/κάτω» να είναι κατακόρυφα παρά την κλίση του χεριού
        xw = x @ G[k, :3, :3].T
        rh = np.linalg.norm(xw[:, :2], axis=1)
        ceiling = ((xw[:, 2] > 1.0) & (rh < 4.0)).sum()
        ground = ((xw[:, 2] < -0.8) & (rh < 6.0)).sum()
        env[k - A] = ceiling, ground, np.median(np.linalg.norm(x, axis=1))

    ks = np.arange(A, B)
    inv = np.linalg.inv
    turn = np.array([np.degrees(np.arccos(np.clip((np.trace((inv(G[k - 1]) @ G[k])[:3, :3]) - 1) / 2, -1, 1))) * 10
                     for k in ks])
    tilt = np.degrees(np.arccos(np.clip(G[ks, 2, 2], -1, 1)))

    print(f"Παράθυρα {WIN} σαρώσεων · Δ(σφάλμα ύψους) = πόσο ξέφυγε το ύψος μέσα στο παράθυρο (m)\n")
    names = [r.split("/")[-1].replace("indoor_detail_", "") for r in RUNS]
    print(f"{'σαρώσεις':>11}" + "".join(f"{n[:16]:>17}" for n in names) + f"{'ταβάνι':>8}{'έδαφος':>8}{'απόστ.':>8}{'στροφή':>8}{'κλίση':>7}")
    for a in range(A, B - WIN + 1, WIN):
        b = a + WIN; i, j = a - A, b - A
        row = "".join(f"{errs[r][b - 1] - errs[r][a]:+17.2f}" for r in RUNS)
        print(f"{a:5d}–{b:<5d}" + row + f"{np.median(env[i:j, 0]):8.0f}{np.median(env[i:j, 1]):8.0f}"
              f"{np.median(env[i:j, 2]):7.1f}m{np.median(turn[i:j]):7.0f}°/s{np.median(tilt[i:j]):5.0f}°")

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for r, n in zip(RUNS, names):
        ax[0].plot(ks, errs[r][A:B], label=n, lw=1)
    ax[0].axhline(0, c="k", lw=0.5); ax[0].set_ylabel("σφάλμα ύψους (m)"); ax[0].legend(fontsize=8)
    ax[1].plot(ks, env[:, 0], label="σημεία ταβανιού (<4 m οριζ., >1 m πάνω)")
    ax[1].plot(ks, env[:, 1], label="σημεία εδάφους (<6 m οριζ., >0.8 m κάτω)")
    ax[1].set_ylabel("σημεία"); ax[1].legend(fontsize=8)
    ax[2].plot(ks, turn, label="ρυθμός στροφής (°/s)", lw=0.8); ax[2].plot(ks, tilt, label="κλίση μονάδας (°)", lw=0.8)
    ax[2].set_xlabel("σάρωση"); ax[2].legend(fontsize=8)
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=90)
    print(f"\n→ {FIG}")


if __name__ == "__main__":
    main()
