#!/usr/bin/env python3
"""Από πού έρχεται η υποεκτίμηση της μετατόπισης της εικόνας (κλίμακα 0.92, #023) και ποιο φίλτρο τη διορθώνει; (#024)

Για ζεύγη σαρώσεων σε τρία παράθυρα, κάθε αντιστοίχιση SIFT χαρακτηρίζεται:
  - απόσταση του σημείου στη νέα σάρωση·
  - «ακμή βάθους»: στη γειτονιά 3×3 του pixel (σε μία από τις δύο εικόνες) οι αποστάσεις διαφέρουν > 5 %·
  - «δίπλα σε κενό»: στη γειτονιά 3×3 υπάρχει pixel χωρίς μέτρηση — άχρηστο: ισχύει για το 100 % (τα ~860 σημεία
    ανά δακτύλιο δεν γεμίζουν τις 1024 στήλες).
Για κάθε φίλτρο: ίδιος αλγόριθμος με το `ScanMotionEstimator` (RANSAC + μοντέλο car + παρεμβολή) μόνο με όσες
αντιστοιχίσεις περνούν. Μετρά την κλίμακα της μετατόπισης (προβολή στην αληθινή, διάμεσος — στόχος 1.00), το σφάλμα
στροφής / θέσης, και την επιτυχία.

    python scripts/test_i3_scale_bias.py [--bearing=0.35] [--calib] [--render=splat|raycast] [--width=1024]
    --render=raycast: η εικόνα από ακτίνα ανά pixel (#029)· --width: στήλες του πανοράματος (ο σαρωτής: 600 = 0.6°)
    --calib: βαθμονομημένο intensity (runs/intensity_calib.npz, calibrate_intensity.py, #025)
    --bearing: έλεγχος inliers κατά διεύθυνση (μοίρες) και απόσταση (5 %) αντί για 3D απόσταση (0.30 m)
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

import kiss_slam.intensity_deskew as I  # noqa: E402
from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")
_b = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--bearing=")), None)
if _b:
    I.INLIER_TEST, I.BEARING_THR = "bearing", _b
if "--calib" in sys.argv:
    I.CALIB = dict(np.load("runs/intensity_calib.npz"))
I.RENDER = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--render=")), "splat")
I.W = int(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--width=")), 1024))
WINDOWS = {"ισόγειο 300–400": (300, 400), "διάδρομος 400–500": (400, 500), "όροφος 1200–1300": (1200, 1300)}
STEP = 2


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "timestamp", "ring"])
    return (np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64), s["timestamp"].astype(np.float64),
            s["intensity"].astype(np.float64), s["ring"].astype(np.int64))


def flags(P, valid, kp):
    """(ακμή βάθους, δίπλα σε κενό) στη γειτονιά 3×3 του pixel του χαρακτηριστικού."""
    x, y = kp.pt
    r = min(max(int(round((y + 0.5) / I.UP - 0.5)), 0), P.shape[0] - 1); c = int(round(x)) % I.W
    rr = np.clip(np.arange(r - 1, r + 2), 0, P.shape[0] - 1); cc = np.arange(c - 1, c + 2) % I.W
    v = valid[np.ix_(rr, cc)]
    rng = np.linalg.norm(P[np.ix_(rr, cc)], axis=2)[v]
    edge = len(rng) > 1 and (rng.max() - rng.min()) > 0.05 * rng.min()
    return edge, not v.all()


def estimate(p, tp, q, tq, rng):
    if len(p) < I.MIN_INL:
        return None
    M0, inl = I.ransac(p, q, rng)
    if M0 is None:
        return None
    M1, _ = I.fit_time(p[inl], tp[inl], q[inl], tq[inl], M0, "car")
    return M1


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    st, _ = load_tum(find_tum("runs/indoor_detail_base_overlapfix")); _, G = interpolate(gs, gT, st)
    inv = np.linalg.inv
    need = {k for a, b in WINDOWS.values() for k in range(a, b, STEP)}
    need |= {k - 1 for k in need}
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    F = {}
    for k in range(max(need) + 1):
        scan = ds[k]
        if k in need:
            F[k] = I.features(*scan, sift)

    FILTERS = {
        "όλες (σημερινό)": lambda r, e, h: np.ones_like(r, bool),
        "χωρίς ακμές βάθους (5 %)": lambda r, e, h: ~e,
        "απόσταση > 3 m": lambda r, e, h: r > 3,
        "απόσταση > 4 m": lambda r, e, h: r > 4,
        "απόσταση > 5 m": lambda r, e, h: r > 5,
        "όχι «ίδιο σημείο» (|p−q| > 3 cm)": "stuck3",
        "όχι «ίδιο σημείο» (|p−q| > 5 cm)": "stuck5",
        "όχι «ίδιο σημείο» 5 cm, > 3 m": "stuck5far",
        "όχι κάτω από −15°": lambda r, e, h: h > -15,
        "όχι κάτω από −10°": lambda r, e, h: h > -10,
        "όχι κάτω από −10°, > 3 m": lambda r, e, h: (h > -10) & (r > 3),
        "όχι κάτω από 0°": lambda r, e, h: h > 0,
    }
    out = {w: {f: [] for f in FILTERS} for w in WINDOWS}
    share = {w: [] for w in WINDOWS}
    for w, (a, b) in WINDOWS.items():
        for k in range(a, b, STEP):
            P1, T1, v1, kp1, d1, _ = F[k - 1]; P2, T2, v2, kp2, d2, t0 = F[k]
            truth = inv(G[k - 1]) @ G[k]
            good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < I.RATIO * n.distance]
            rows = []
            for m in good:
                x = I.lookup(P1, T1, v1, kp1[m.queryIdx], True); y = I.lookup(P2, T2, v2, kp2[m.trainIdx], True)
                if x is None or y is None:
                    continue
                e1, h1 = flags(P1, v1, kp1[m.queryIdx]); e2, h2 = flags(P2, v2, kp2[m.trainIdx])
                rows.append((x[0], x[1], y[0], y[1], e1 or e2, h1 or h2))
            if not rows:
                continue
            p = np.array([r[0] for r in rows]); tp = (np.array([r[1] for r in rows]) - t0) / 0.1
            q = np.array([r[2] for r in rows]); tq = (np.array([r[3] for r in rows]) - t0) / 0.1
            edge = np.array([r[4] for r in rows]); hole = np.array([r[5] for r in rows]); rng_ = np.linalg.norm(q, axis=1)
            # 3ο όρισμα των φίλτρων: γωνία ύψους του σημείου (°) στο πλαίσιο του αισθητήρα — αρνητική = προς το δάπεδο
            elev = np.degrees(np.arctan2(q[:, 2], np.linalg.norm(q[:, :2], axis=1)))
            share[w].append((edge.mean(), (elev < -10).mean(), len(rows)))
            for f, fn in FILTERS.items():
                d = np.linalg.norm(p - q, axis=1)          # «ίδιο σημείο» στο πλαίσιο του αισθητήρα = κολλημένο μοτίβο
                sel = {"stuck3": d > 0.03, "stuck5": d > 0.05, "stuck5far": (d > 0.05) & (rng_ > 3)}[fn] \
                    if isinstance(fn, str) else fn(rng_, edge, elev)
                M = estimate(p[sel], tp[sel], q[sel], tq[sel], np.random.default_rng(0))
                if M is None:
                    out[w][f].append((np.nan, np.nan, np.nan)); continue
                D = inv(truth) @ M; tg = truth[:3, 3]
                out[w][f].append((M[:3, 3] @ tg / (tg @ tg),
                                  np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))),
                                  1000 * np.linalg.norm(D[:3, 3])))

    for w in WINDOWS:
        sh = np.array(share[w])
        print(f"\n{w}: αντιστοιχίσεις/ζεύγος {np.median(sh[:, 2]):.0f} · σε ακμή βάθους {100*sh[:, 0].mean():.0f} % · κάτω από −10° {100*sh[:, 1].mean():.0f} %")
        print(f"   {'φίλτρο':28}{'κλίμακα':>9}{'στροφή':>9}{'θέση':>9}{'επιτυχία':>10}")
        for f in FILTERS:
            R = np.array(out[w][f]); ok = ~np.isnan(R[:, 0])
            print(f"   {f:28}{np.median(R[ok, 0]):9.3f}{np.median(R[ok, 1]):8.2f}°{np.median(R[ok, 2]):6.0f} mm{100*ok.mean():9.0f}%")


if __name__ == "__main__":
    main()
