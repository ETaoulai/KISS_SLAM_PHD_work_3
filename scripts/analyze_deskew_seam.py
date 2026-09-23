#!/usr/bin/env python3
"""Πόσο καλό είναι ένα deskew; Μέτρηση στη «ραφή» της σάρωσης, χωρίς SLAM.

Ένα περιστρεφόμενο LiDAR ξεκινά και τελειώνει τη σάρωση στο ίδιο αζιμούθιο: οι πρώτες και οι
τελευταίες επιστροφές βλέπουν τις ΙΔΙΕΣ επιφάνειες με 0.1 s διαφορά. Αν η μονάδα κινείται και το
deskew είναι λάθος, οι δύο εκδοχές της επιφάνειας δεν συμπίπτουν. Μετράμε το «άνοιγμα της ραφής»:
για κάθε σημείο της αρχής (κανονικοποιημένος χρόνος s < 0.03), απόσταση από το επίπεδο των
πλησιέστερων σημείων του τέλους (s > 0.97). Διάμεσος ανά σάρωση, σε μέτρα.

Συγκρίνονται:
  - χωρίς deskew·
  - το deskew του KISS: κίνηση της ΠΡΟΗΓΟΥΜΕΝΗΣ σάρωσης, από την εκτίμηση του baseline·
  - deskew με την αληθινή κίνηση από το GT, για διάφορες μετατοπίσεις χρόνου τ.
Η τ που κλείνει καλύτερα τη ραφή είναι η σωστή χρονική αντιστοίχιση GT–σάρωσης ΓΙΑ ΤΟ DESKEW.

    python scripts/analyze_deskew_seam.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
from kiss_icp.datasets import dataset_factory
from kiss_icp.preprocess import Preprocessor
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")
RUN = "runs/indoor_detail_base_overlapfix"
TAUS = np.arange(-0.175, 0.051, 0.025)
WINDOWS = {"στροφές στο ισόγειο (400–550)": (400, 550), "σκάλα πάνω (820–990)": (820, 990),
           "σκάλα κάτω (2014–2258)": (2014, 2258)}
STEP, MIN_RATE = 3, 12.0            # κάθε 3η σάρωση, μόνο όσες στρίβουν > 12°/s (εκεί φαίνεται το deskew)


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "timestamp"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    r = np.linalg.norm(xyz, axis=1)
    ok = ~np.isnan(xyz).any(axis=1) & (r > 1.0) & (r < 30.0)
    return xyz[ok], s["timestamp"].astype(np.float64)[ok]


def seam_gap(des, ts):
    s = (ts - ts.min()) / (ts.max() - ts.min())
    A, B = des[s < 0.03], des[s > 0.97]
    if len(A) < 50 or len(B) < 50:
        return np.nan
    d, nn = cKDTree(B).query(A, k=8)
    ok = d[:, -1] < 0.5
    Q = B[nn[ok]]; c = Q.mean(axis=1)
    w, v = np.linalg.eigh(np.einsum("nki,nkj->nij", Q - c[:, None], Q - c[:, None]))
    n = v[:, :, 0]
    return float(np.median(np.abs(np.einsum("ni,ni->n", A[ok] - c, n))))


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN))
    inv = np.linalg.inv
    ang = lambda M: np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))
    _, G0 = interpolate(gs, gT, st - 0.070)

    pick = {}
    for name, (a, b) in WINDOWS.items():
        for k in range(a, b, STEP):
            if ang(inv(G0[k - 1]) @ G0[k]) * 10 > MIN_RATE:
                pick[k] = name
    pre = Preprocessor(1e9, 0.0, True, 0)
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw

    res = {name: {"χωρίς deskew": [], "KISS (προηγ. εκτίμηση)": [], **{t: [] for t in TAUS}} for name in WINDOWS}
    for k in range(max(pick) + 1):
        xyz, ts = ds[k]
        if k not in pick:
            continue
        r = res[pick[k]]
        r["χωρίς deskew"].append(seam_gap(pre.preprocess(xyz, ts, np.eye(4)), ts))
        r["KISS (προηγ. εκτίμηση)"].append(seam_gap(pre.preprocess(xyz, ts, inv(E[k - 2]) @ E[k - 1]), ts))
        for tau in TAUS:
            _, Gt = interpolate(gs, gT, np.array([st[k - 1], st[k]]) + tau)
            r[tau].append(seam_gap(pre.preprocess(xyz, ts, inv(Gt[0]) @ Gt[1]), ts))

    for name, r in res.items():
        n = len(r["χωρίς deskew"])
        print(f"\n== {name}: {n} σαρώσεις με στροφή > {MIN_RATE:.0f}°/s — άνοιγμα ραφής (διάμεσος, cm) ==")
        print(f"   χωρίς deskew                 {100*np.nanmedian(r['χωρίς deskew']):6.2f}")
        print(f"   KISS (προηγούμενη εκτίμηση)  {100*np.nanmedian(r['KISS (προηγ. εκτίμηση)']):6.2f}")
        vals = {t: np.nanmedian(r[t]) for t in TAUS}
        best = min(vals, key=vals.get)
        for t in TAUS:
            mark = "  ← καλύτερο" if t == best else ("  ← αυτό χρησιμοποίησε το oracle run" if abs(t + 0.075) < 1e-9 else "")
            print(f"   GT, τ = {t*1000:+5.0f} ms           {100*vals[t]:6.2f}{mark}")


if __name__ == "__main__":
    main()
