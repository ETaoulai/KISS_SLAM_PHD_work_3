#!/usr/bin/env python3
"""Τι είναι καρφωμένο στον σαρωτή: χρονική στατιστική ανά pixel + φίλτρο ακμών στη διάμεση εικόνα (#036).

Σε όλη τη διαδρομή (κάθε STEP-οστή σάρωση, ray casting) κρατιούνται όλα τα πανοράματα intensity και υπολογίζονται ανά pixel:
μέση, διάμεση, τυπική απόκλιση, MAD (διάμεση απόλυτη απόκλιση), μέση απόσταση και διασπορά της, ποσοστό σαρώσεων χωρίς
μέτρηση. Μετά από εκατοντάδες σαρώσεις η σκηνή γίνεται ομαλή διαβάθμιση· ό,τι κρατά κοφτή ακμή στη ΔΙΑΜΕΣΗ εικόνα είναι
σταθερό ως προς τον σαρωτή (ιδέα Λ.Γ.): σκιές της διάταξης, σιλουέτα του χειριστή. Παράγονται τρεις μάσκες:
  narrow — σκουρότερο από τους ±20 γείτονες του δακτυλίου, σταθερά στον χρόνο (t > 20)·
  edges  — ακμές κατά το αζιμούθιο του log(διάμεσης), πάνω από το 97ο εκατοστημόριο·
  union  — η ένωσή τους.  Χρήση: `precompute_i3_motion.py --mask=runs/rig_masks_<bag>.npz --mask-key=union`.

    python scripts/analyze_rig_mask.py [βήμα=5] [--bag=data/church_02_cut.bag]
"""
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
import kiss_slam.intensity_deskew as I  # noqa: E402
from test_i3_scale_bias import read_raw  # noqa: E402

warnings.simplefilter("ignore")
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
STEP, BAG = (int(pos[0]) if pos else 5), arg("bag", "data/church_02_cut.bag")
NEIGH, T_MIN, EDGE_PCT = 41, 20.0, 97          # παράθυρο γειτόνων (στήλες), κατώφλι t, εκατοστημόριο ακμών
OUT, FIG = Path(f"runs/rig_masks_{Path(BAG).stem}.npz"), Path("docs/figures/rig_annotated.png")


def wrap_dilate(m, r):
    w = np.concatenate([m, m, m], axis=1)
    return ndi.binary_dilation(w, structure=np.ones((2 * r + 1, 2 * r + 1), bool))[:, I.W:2 * I.W]


def main():
    I.RENDER = "raycast"
    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG), sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    frames, acc = [], None
    for k in range(0, len(ds), STEP):
        xyz, ts, inten, ring = ds[k]
        ok = ~np.isnan(xyz).any(axis=1)
        img, P, T, valid = I.raycast_grid(xyz[ok], ts[ok], inten[ok], ring[ok])
        v = valid & np.isfinite(img)
        frames.append(np.where(v, img, np.nan).astype(np.float32))
        rng = np.linalg.norm(P, axis=2)
        # λόγος προς τη διάμεση των ±20 γειτόνων στον δακτύλιο (η «στενή» μάσκα, #035)
        f = np.where(v, img, np.nan); rm = np.nanmedian(f, axis=1, keepdims=True); f = np.where(v, f, rm)
        loc = ndi.median_filter(f, size=(1, NEIGH), mode="wrap"); rt = np.where(v & (loc > 5), f / loc, np.nan); g = np.isfinite(rt)
        if acc is None:
            acc = {x: np.zeros(img.shape) for x in ("n", "nv", "r", "r2", "s", "s2", "g")}
        acc["n"] += v; acc["nv"] += ~v
        acc["r"] += np.where(v, rng, 0); acc["r2"] += np.where(v, rng ** 2, 0)
        acc["s"] += np.where(g, rt, 0); acc["s2"] += np.where(g, rt ** 2, 0); acc["g"] += g
    S = np.stack(frames); N = len(S); c = np.maximum(acc["n"], 1)
    seen = acc["n"] >= 0.5 * N
    mean, med = np.nanmean(S, 0), np.nanmedian(S, 0)
    std, mad = np.sqrt(np.nanmean((S - mean) ** 2, 0)), np.nanmedian(np.abs(S - med), 0)
    r_mean, r_std = acc["r"] / c, np.sqrt(np.maximum(acc["r2"] / c - (acc["r"] / c) ** 2, 0))
    miss = acc["nv"] / N
    cg = np.maximum(acc["g"], 1); mu = acc["s"] / cg; sd = np.sqrt(np.maximum(acc["s2"] / cg - mu ** 2, 0))
    t = np.where(acc["g"] >= 0.3 * N, (1 - mu) / np.maximum(sd / np.sqrt(cg), 1e-6), np.nan)
    narrow = seen & (t > T_MIN)
    # ακμές κατά το αζιμούθιο του log(διάμεσης): μια σκίαση είναι πολλαπλασιαστική → σκαλοπάτι στον λογάριθμο
    L = np.log(np.where(seen, med, np.nan) + 8); Lf = np.where(seen, L, np.nanmedian(L, axis=1, keepdims=True))
    gx = ndi.sobel(Lf, axis=1, mode="wrap") / 8
    thr = np.nanpercentile(np.abs(gx)[seen], EDGE_PCT); edges = seen & (np.abs(gx) > thr)
    union = narrow | edges; hole = miss > 0.8
    np.savez(OUT, narrow=narrow, edges=edges, union=union, hole=hole, mean=mean, med=med, std=std, mad=mad,
             r_mean=r_mean, r_std=r_std, miss=miss, seen=seen, scans=N)
    print(f"{N} σαρώσεις (κάθε {STEP}η), ray casting, {Path(BAG).name}")
    print(f"  διάμεσοι στα pixel με μέτρηση: φωτεινότητα {np.nanmedian(med[seen]):.0f} · τυπ. απόκλιση {np.nanmedian(std[seen]):.0f} · "
          f"MAD {np.nanmedian(mad[seen]):.0f} · απόσταση {np.nanmedian(r_mean[seen]):.1f} m (διασπορά {np.nanmedian(r_std[seen]):.1f})")
    print(f"  πάντα κενό (> 80 % των σαρώσεων): {100*hole.mean():.1f} % της εικόνας — η σιλουέτα του χειριστή")
    print(f"  μάσκες: narrow {narrow.sum()} px ({100*narrow.mean():.2f} %) · edges {edges.sum()} ({100*edges.mean():.2f} %) · "
          f"union {union.sum()} ({100*union.mean():.2f} %) · κατώφλι ακμής {100*(np.exp(thr)-1):.0f} % ανά pixel")
    lab, n = ndi.label(edges, structure=np.ones((3, 3))); sizes = ndi.sum(edges, lab, range(1, n + 1))
    print("  μεγαλύτερες συνεκτικές ακμές:")
    for j in np.argsort(-sizes)[:6]:
        rr, cc = np.where(lab == j + 1)
        print(f"    {int(sizes[j]):4d} px · δακτύλιοι {rr.min():2d}–{rr.max():2d} · στήλες {cc.min():4d}–{cc.max():4d} · "
              f"φωτεινότητα {np.nanmedian(med[lab == j + 1]):.0f} · απόσταση {np.nanmedian(r_mean[lab == j + 1]):.1f} m")

    v = med[seen]; a, b = np.percentile(v, [2, 98]); img = np.clip((med - a) / (b - a), 0, 1)
    rgb = np.repeat(np.where(seen, img, 0)[:, :, None], 3, 2); rgb[~seen] = (0.45, 0.45, 0.55); rgb[edges] = (1, 0.1, 0.1)
    fig, ax = plt.subplots(3, 1, figsize=(16, 11))
    ax[0].imshow(rgb, aspect="auto", interpolation="nearest")
    ax[0].set_title("χρονικά ΔΙΑΜΕΣΗ εικόνα intensity · κόκκινο = ακμές · γκρι-μπλε = πάντα κενό (χειριστής)", fontsize=10)
    ax[1].imshow(np.where(seen, mad, np.nan), aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=40)
    ax[1].set_title("MAD της φωτεινότητας στον χρόνο (σκούρο = καρφωμένη τιμή)", fontsize=10)
    m3 = np.zeros((*seen.shape, 3)); m3[narrow & ~edges] = (0.2, 0.5, 1); m3[edges & ~narrow] = (1, 0.2, 0.2); m3[narrow & edges] = (1, 1, 1)
    ax[2].imshow(m3, aspect="auto", interpolation="nearest")
    ax[2].set_title("μάσκες: άσπρο = narrow και edges · μπλε = μόνο narrow (γείτονες ±20) · κόκκινο = μόνο edges", fontsize=10)
    for A in ax:
        A.set_yticks(range(0, 64, 8)); A.set_xticks(range(0, 1025, 128)); A.set_ylabel("δακτύλιος")
    fig.suptitle(f"Τι είναι καρφωμένο στον σαρωτή — {N} σαρώσεις, {Path(BAG).stem}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=90)
    print(f"→ {OUT} · {FIG}")


if __name__ == "__main__":
    main()
