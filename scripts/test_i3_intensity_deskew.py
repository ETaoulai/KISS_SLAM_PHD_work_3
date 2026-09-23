#!/usr/bin/env python3
"""Ι-3, φθηνός έλεγχος: εκτιμά η εικόνα intensity την κίνηση ΜΕΣΑ στη σάρωση καλύτερα από τη μαντεψιά του KISS;

Το πανόραμα intensity είναι «κάμερα rolling shutter»: κάθε στήλη μετρήθηκε σε άλλη στιγμή της περιστροφής.
Για κάθε ζεύγος διαδοχικών σαρώσεων (k, k+1), από τα ΑΚΑΤΕΡΓΑΣΤΑ σημεία (χωρίς deskew):
  1. πανόραμα intensity που κρατά ανά pixel το 3D σημείο (πλαίσιο αισθητήρα τη στιγμή της μέτρησης) και τον χρόνο του·
  2. SIFT + ratio test → ζεύγη (p, t_p) στη σάρωση k και (q, t_q) στην k+1·
  3. κίνηση σταθερής ταχύτητας ξ = (ω, v) ανά περίοδο σάρωσης, με θέση αισθητήρα τη χρονική στιγμή t
     (σε περιόδους από την αρχή της k): [R(t·ω), t·v]. Ίδιο σημείο του χώρου ⇒
         R(t_p ω)·p + t_p v  =  R(t_q ω)·q + t_q v
     αρχική τιμή από RANSAC + Kabsch (χωρίς χρόνο), μετά ελάχιστα τετράγωνα με ανθεκτική απώλεια·
  4. η κίνηση μιας περιόδου [R(ω), v] συγκρίνεται με την ΑΛΗΘΙΝΗ κίνηση της σάρωσης k+1 από το GT
     (inv(G_k)·G_{k+1}, τ = 0 — αυτή που έδωσε ATE 0.101 m στο #012).

Σύγκριση με ό,τι θα έδινε κάθε άλλη πηγή για το deskew της σάρωσης k+1:
  - κανένα deskew (ταυτοτική)·
  - μαντεψιά KISS: η κίνηση της ΠΡΟΗΓΟΥΜΕΝΗΣ σάρωσης, από την εκτίμηση του baseline (inv(E_{k-1})·E_k)·
  - ICP της ίδιας σάρωσης (inv(E_k)·E_{k+1}) — ό,τι χρησιμοποιεί η επανάληψη του #017·
  - εικόνα χωρίς χρόνο (Kabsch)· εικόνα με χρόνο (Ι-3).

    python scripts/test_i3_intensity_deskew.py [--run=runs/indoor_detail_base_overlapfix]
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from kiss_icp.datasets import dataset_factory
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")

BAG, GT = Path("data/church_02_cut.bag"), "gt/church_02_gt-tum.txt"
RUN = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--run=")), "runs/indoor_detail_base_overlapfix")
SEGMENTS = {"ισόγειο, στροφές": (400, 550), "σκάλα πάνω": (820, 990), "όροφος": (1200, 1400), "σκάλα κάτω": (2014, 2258)}
STEP, PERIOD = 2, 0.1                    # κάθε 2η σάρωση · περίοδος σάρωσης (s)
W, UP, MIN_RANGE = 1024, 8, 1.0
RATIO, RANSAC_THR, RANSAC_IT, MIN_INL, FIT_THR = 0.75, 0.30, 400, 10, 0.10


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "timestamp", "ring"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = ~np.isnan(xyz).any(axis=1) & (np.linalg.norm(xyz, axis=1) > MIN_RANGE)
    return (xyz[ok], s["timestamp"].astype(np.float64)[ok],
            s["intensity"].astype(np.float64)[ok], s["ring"].astype(np.int64)[ok])


def panorama(xyz, ts, inten, ring):
    """(εικόνα για τον ανιχνευτή, 3D σημείο ανά pixel, χρόνος ανά pixel, μάσκα έγκυρων)."""
    elev = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
    rings = np.unique(ring)
    order = rings[np.argsort([-elev[ring == r].mean() for r in rings])]
    pos = np.empty(rings.max() + 1, dtype=np.int64); pos[order] = np.arange(len(order))
    row = pos[ring]
    col = ((np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi) * W).astype(int) % W
    H = len(rings)
    img = np.full((H, W), np.nan); P = np.zeros((H, W, 3)); T = np.zeros((H, W)); valid = np.zeros((H, W), bool)
    img[row, col] = inten; P[row, col] = xyz; T[row, col] = ts; valid[row, col] = True
    for r in range(H):
        v = valid[r]
        if v.sum() > 1:
            img[r, ~v] = np.interp(np.where(~v)[0], np.where(v)[0], img[r, v])
    img = np.nan_to_num(img, nan=0.0)
    big = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H * UP), interpolation=cv2.INTER_LINEAR)
    return big, P, T, valid


def lookup(P, T, valid, kp):
    x, y = kp.pt
    r = min(max(int(round((y + 0.5) / UP - 0.5)), 0), P.shape[0] - 1); c = int(round(x)) % W
    for dc in (0, -1, 1, -2, 2):
        cc = (c + dc) % W
        if valid[r, cc]:
            return P[r, cc], T[r, cc]
    return None


def kabsch(A, B):
    """T με A ≈ R·B + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = ca - R @ cb
    return M


def ransac(A, B, rng):
    best = None
    for _ in range(RANSAC_IT):
        i = rng.choice(len(A), 3, replace=False)
        M = kabsch(A[i], B[i])
        inl = np.linalg.norm(A - (B @ M[:3, :3].T + M[:3, 3]), axis=1) < RANSAC_THR
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_INL:
        return None, best
    return kabsch(A[best], B[best]), best


def residual(x, p, tp, q, tq):
    w, v = x[:3], x[3:]
    a = Rotation.from_rotvec(tp[:, None] * w).apply(p) + tp[:, None] * v
    b = Rotation.from_rotvec(tq[:, None] * w).apply(q) + tq[:, None] * v
    return (a - b).ravel()


def fit_time(p, tp, q, tq, M0):
    """Κίνηση σταθερής ταχύτητας ανά περίοδο, με τον χρόνο κάθε σημείου. Επιστρέφει (T μιας περιόδου, inliers)."""
    # Χωρίς χρόνο, M0 απεικονίζει q (πλαίσιο k+1) σε p (πλαίσιο k): ≈ κίνηση μιας περιόδου.
    x = np.concatenate([Rotation.from_matrix(M0[:3, :3]).as_rotvec(), M0[:3, 3]])
    keep = np.ones(len(p), bool)
    for _ in range(3):
        x = least_squares(residual, x, args=(p[keep], tp[keep], q[keep], tq[keep]),
                          loss="soft_l1", f_scale=0.05).x
        r = np.linalg.norm(residual(x, p, tp, q, tq).reshape(-1, 3), axis=1)
        keep = r < FIT_THR
        if keep.sum() < MIN_INL:
            return None, keep
    M = np.eye(4); M[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix(); M[:3, 3] = x[3:]
    return M, keep


def err(M, ref):
    """(σφάλμα στροφής °, σφάλμα θέσης mm) της κίνησης M έναντι της ref."""
    D = np.linalg.inv(ref) @ M
    return (np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))), 1000 * np.linalg.norm(D[:3, 3]))


def main():
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN))
    _, G = interpolate(gs, gT, st)             # τ = 0: ο σωστός χρονισμός για το deskew (#012)
    inv = np.linalg.inv

    need = {k for a, b in SEGMENTS.values() for k in range(a, b, STEP)}
    need |= {k + 1 for k in need}
    ds = dataset_factory(dataloader="rosbag", data_dir=BAG, sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    sift = cv2.SIFT_create()
    pano = {}
    for k in range(max(need) + 1):
        xyz, ts, inten, ring = ds[k]
        if k in need:
            big, P, T, valid = panorama(xyz, ts, inten, ring)
            kps, desc = sift.detectAndCompute(big, None)
            pano[k] = (P, T, valid, kps, desc)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    SRC = ["κανένα deskew", "μαντεψιά KISS", "ICP ίδιας σάρωσης", "εικόνα χωρίς χρόνο", "εικόνα με χρόνο (Ι-3)"]
    res = {name: {s: [] for s in SRC} | {"inliers": []} for name in SEGMENTS}
    for name, (a, b) in SEGMENTS.items():
        for k in range(a, b, STEP):
            P1, T1, v1, kp1, d1 = pano[k]; P2, T2, v2, kp2, d2 = pano[k + 1]
            truth = inv(G[k]) @ G[k + 1]
            cand = {"κανένα deskew": np.eye(4), "μαντεψιά KISS": inv(E[k - 1]) @ E[k],
                    "ICP ίδιας σάρωσης": inv(E[k]) @ E[k + 1], "εικόνα χωρίς χρόνο": None, "εικόνα με χρόνο (Ι-3)": None}
            n_inl = 0
            if d1 is not None and d2 is not None:
                good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance]
                pairs = [(lookup(P1, T1, v1, kp1[m.queryIdx]), lookup(P2, T2, v2, kp2[m.trainIdx])) for m in good]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                if len(pairs) >= MIN_INL:
                    p = np.array([x[0] for x, _ in pairs]); q = np.array([y[0] for _, y in pairs])
                    t0 = min(x[1] for x, _ in pairs)
                    tp = (np.array([x[1] for x, _ in pairs]) - t0) / PERIOD
                    tq = (np.array([y[1] for _, y in pairs]) - t0) / PERIOD
                    M0, inl = ransac(p, q, rng)
                    if M0 is not None:
                        cand["εικόνα χωρίς χρόνο"] = M0
                        M1, keep = fit_time(p[inl], tp[inl], q[inl], tq[inl], M0)
                        if M1 is not None:
                            cand["εικόνα με χρόνο (Ι-3)"] = M1; n_inl = int(keep.sum())
            for s, M in cand.items():
                res[name][s].append(err(M, truth) if M is not None else (np.nan, np.nan))
            res[name]["inliers"].append(n_inl)

    print(f"Σφάλμα της κίνησης που θα χρησιμοποιούσε το deskew της σάρωσης, έναντι της αληθινής (GT, τ = 0).")
    print(f"Διάμεσος · κάθε {STEP}η σάρωση · ICP/KISS από `{RUN}`\n")
    print(f"{'':26}" + "".join(f"{n:>24}" for n in SEGMENTS))
    for s in SRC:
        cells = []
        for name in SEGMENTS:
            e = np.array(res[name][s], dtype=float)
            ok = ~np.isnan(e[:, 0])
            cells.append(f"{np.median(e[ok, 0]):5.2f}° {np.median(e[ok, 1]):5.0f} mm ({100*ok.mean():3.0f}%)" if ok.any() else "—")
        print(f"{s:26}" + "".join(f"{c:>24}" for c in cells))
    print(f"{'inliers (διάμεσος)':26}" + "".join(f"{np.median(res[n]['inliers']):>24.0f}" for n in SEGMENTS))
    print("\nΠοσοστό σαρώσεων όπου η εικόνα με χρόνο είναι πιο κοντά στην αλήθεια από τη μαντεψιά του KISS (στροφή):")
    for name in SEGMENTS:
        a = np.array(res[name]["εικόνα με χρόνο (Ι-3)"], float)[:, 0]; b = np.array(res[name]["μαντεψιά KISS"], float)[:, 0]
        ok = ~np.isnan(a)
        print(f"  {name:20} {100*np.mean(a[ok] < b[ok]):4.0f}%  (από {ok.sum()} σαρώσεις)")


if __name__ == "__main__":
    main()
