#!/usr/bin/env python3
"""Ποια pixel του πανοράματος δείχνουν πάντα το ΙΔΙΟ πράγμα; (ιδέα Λ.Γ., #035)

Δίπλα στον σαρωτή υπάρχουν καλώδια και εξαρτήματα της διάταξης: δίνουν επιστροφές κάτω από 1 m (τις κόβει το
`MIN_RANGE`) ή μπλοκάρουν τις ακτίνες, αφήνοντας στην εικόνα σκιές σε **σταθερές θέσεις**. Ό,τι χαρακτηριστικό
πιάσει το SIFT εκεί, θα το ξαναβρεί στο ίδιο pixel και θα δείξει μηδενική κίνηση.

Εδώ, σε όλη τη διαδρομή (κάθε N-οστή σάρωση), υπολογίζεται ανά pixel:
  - πόσο συχνά ΔΕΝ έχει μέτρηση (σταθερά κενό = μπλοκαρισμένη κατεύθυνση)·
  - η διασπορά της ΑΠΟΣΤΑΣΗΣ (σταθερό αντικείμενο της διάταξης = σχεδόν μηδενική, ενώ ο κόσμος αλλάζει)·
  - η διασπορά του intensity.
Από αυτά προτείνεται μάσκα «σταθερών» pixel και μετριέται πόσες από τις «κολλημένες» αντιστοιχίσεις πέφτουν μέσα.

    python scripts/analyze_static_pixels.py [βήμα=5] [--bag=...] [--windows=400,500,1200,1300]
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
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
STEP = int(pos[0]) if pos else 5
BAG = arg("bag", "data/church_02_cut.bag")
WIN = [int(v) for v in arg("windows", "400,500,1200,1300").split(",")]
OUT, FIG = Path("runs/static_pixels.npz"), Path("docs/figures/static_pixels.png")
R_STD, INVALID = 0.15, 0.9        # μάσκα: απόσταση σχεδόν σταθερή (m) ή σχεδόν πάντα κενό


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG), sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    n = len(ds)
    acc = None
    for k in range(0, n, STEP):
        xyz, ts, inten, ring = ds[k]
        ok = ~np.isnan(xyz).any(axis=1) & (np.linalg.norm(xyz, axis=1) > I.MIN_RANGE)
        img, P, T, valid = I.grid(xyz[ok], ts[ok], inten[ok], ring[ok])
        rng = np.linalg.norm(P, axis=2)
        if acc is None:
            acc = {x: np.zeros(img.shape) for x in ("nv", "r", "r2", "i", "i2", "n")}
        acc["nv"] += ~valid
        acc["n"] += valid
        acc["r"] += np.where(valid, rng, 0); acc["r2"] += np.where(valid, rng ** 2, 0)
        acc["i"] += np.where(valid, np.nan_to_num(img), 0); acc["i2"] += np.where(valid, np.nan_to_num(img) ** 2, 0)
    m = len(range(0, n, STEP))
    cnt = np.maximum(acc["n"], 1)
    r_std = np.sqrt(np.maximum(acc["r2"] / cnt - (acc["r"] / cnt) ** 2, 0))
    i_std = np.sqrt(np.maximum(acc["i2"] / cnt - (acc["i"] / cnt) ** 2, 0))
    invalid = acc["nv"] / m
    r_mean = acc["r"] / cnt
    static = ((acc["n"] >= 0.5 * m) & (r_std < R_STD)) | (invalid > INVALID)
    np.savez(OUT, r_std=r_std, i_std=i_std, invalid=invalid, r_mean=r_mean, static=static, scans=m)
    print(f"{m} σαρώσεις (κάθε {STEP}η) από {BAG}")
    print(f"«σταθερά» pixel: {100*static.mean():.1f} % — απόσταση σχεδόν αμετάβλητη (< {R_STD} m) ή σχεδόν πάντα κενά (> {100*INVALID:.0f} %)")
    print(f"  από αυτά, με μέτρηση: {100*(static & (acc['n'] >= 0.5*m)).mean():.1f} % · μέση απόστασή τους {np.median(r_mean[static & (acc['n'] >= 0.5*m)]):.1f} m")
    print(f"  ανά γραμμή (δακτύλιο), τα 8 με τα περισσότερα σταθερά pixel: "
          + ", ".join(f"{r}:{100*static[r].mean():.0f}%" for r in np.argsort(-static.mean(1))[:8]))

    # πόσες «κολλημένες» αντιστοιχίσεις πέφτουν πάνω σε σταθερά pixel;
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    need = set()
    for a, b in zip(WIN[::2], WIN[1::2]):
        need |= set(range(a - 1, b, 2)) | set(range(a, b, 2))
    F = {}
    for k in range(max(need) + 1):
        x = ds[k]
        if k in need:
            ok = ~np.isnan(x[0]).any(axis=1) & (np.linalg.norm(x[0], axis=1) > I.MIN_RANGE)
            F[k] = I.features(*(a[ok] for a in x), sift)
    rows = []
    for a, b in zip(WIN[::2], WIN[1::2]):
        for k in range(a, b, 2):
            P1, T1, v1, kp1, d1, _ = F[k - 1]; P2, T2, v2, kp2, d2, _ = F[k]
            for mm, nn in bf.knnMatch(d1, d2, k=2):
                if mm.distance >= I.RATIO * nn.distance:
                    continue
                x1 = I.lookup(P1, T1, v1, kp1[mm.queryIdx]); y1 = I.lookup(P2, T2, v2, kp2[mm.trainIdx])
                if x1 is None or y1 is None:
                    continue
                u, v = kp2[mm.trainIdx].pt
                r, c = min(max(int(round((v + 0.5) / I.UP - 0.5)), 0), static.shape[0] - 1), int(round(u)) % I.W
                rows.append((np.linalg.norm(x1[0] - y1[0]) < 0.05, static[r, c], a))
    R = np.array(rows)
    print("\nΑντιστοιχίσεις πάνω σε «σταθερά» pixel:")
    for a in sorted(set(R[:, 2])):
        s = R[R[:, 2] == a]
        st, ins = s[:, 0] > 0.5, s[:, 1] > 0.5
        print(f"  σαρώσεις {int(a)}–…: {len(s)} αντιστοιχίσεις · «κολλημένες» {100*st.mean():.0f} % · "
              f"σε σταθερό pixel: κολλημένες {100*ins[st].mean():.0f} %, υπόλοιπες {100*ins[~st].mean():.0f} %")

    fig, ax = plt.subplots(3, 1, figsize=(14, 7))
    for a, d, t, kw in ((ax[0], invalid, "ποσοστό σαρώσεων ΧΩΡΙΣ μέτρηση σε αυτό το pixel", dict(cmap="magma", vmin=0, vmax=1)),
                        (ax[1], np.where(acc["n"] >= 0.5 * m, r_std, np.nan), "διασπορά της απόστασης (m) — σκούρο = σταθερό αντικείμενο",
                         dict(cmap="viridis", vmin=0, vmax=3)),
                        (ax[2], static.astype(float), "η προτεινόμενη μάσκα «σταθερών» pixel", dict(cmap="gray"))):
        im = a.imshow(d, aspect="auto", interpolation="nearest", **kw)
        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, fraction=0.02)
    fig.suptitle(f"Σταθερά ως προς τον σαρωτή pixel — {m} σαρώσεις, {Path(BAG).stem}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=100)
    print(f"\n→ {OUT} · {FIG}")


if __name__ == "__main__":
    main()
