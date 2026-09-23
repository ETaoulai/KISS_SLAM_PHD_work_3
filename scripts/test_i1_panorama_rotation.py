#!/usr/bin/env python3
"""Ι-1, πρώτο τεστ: υπολογίζεται η στροφή από τις εικόνες intensity καλύτερα απ' ό,τι ο ICP;

Για ζεύγη σαρώσεων (k, k+Δ) στη σκάλα και σε έναν όροφο (αναφορά):
  1. deskew ΑΚΡΙΒΩΣ όπως το KISS: στο τέλος της σάρωσης, με την κίνηση της προηγούμενης
     σάρωσης (last_delta), εδώ από την τροχιά του baseline·
  2. πανοραμική εικόνα intensity (δακτύλιος × αζιμούθιο) που κρατά για κάθε pixel και το
     3D σημείο του· ΚΡΑΤΙΟΥΝΤΑΙ και τα μακρινά σημεία (> 50 m), όπως προβλέπει η Ι-1·
  3. SIFT + ratio test → αντιστοιχίσεις → ζεύγη 3D σημείων → RANSAC + Kabsch → σχετική κίνηση·
  4. σφάλμα στροφής έναντι GT (μετατόπιση χρόνου −70 ms, βλ. #010), και σύγκριση με το σφάλμα
     του ICP (baseline) στα ΙΔΙΑ ζεύγη.

Οι εικόνες έχουν μόνο 64 γραμμές (δακτύλιοι), γι' αυτό μεγεθύνονται κατακόρυφα ×8 για τον
ανιχνευτή· κάθε σημείο-κλειδί αντιστοιχίζεται πίσω στον πλησιέστερο δακτύλιο.

    python scripts/test_i1_panorama_rotation.py [--deskew=kiss|gt] [--run=runs/<run>]
        [--deskew-offset-ms=0] [--eval-offset-ms=-10]
    Σωστός χρονισμός (#012): deskew από GT με τ = 0· οι πόζες είναι τότε στο τέλος της σάρωσης → σύγκριση στα −10 ms.

--deskew=kiss (default): ό,τι κάνει το KISS — κίνηση της ΠΡΟΗΓΟΥΜΕΝΗΣ σάρωσης, από την εκτίμηση.
--deskew=gt: η ΠΡΑΓΜΑΤΙΚΗ κίνηση της σάρωσης από το GT (έλεγχος: πόσο σφάλμα φέρνει το deskew).
Η στήλη «εικόνα↔ICP» είναι η διαφωνία των δύο μεθόδων ΜΕΤΑΞΥ ΤΟΥΣ, χωρίς GT: αν είναι πολύ
μικρότερη από το σφάλμα καθεμιάς έναντι GT, το κοινό σφάλμα βρίσκεται στο GT ή στο deskew.
"""
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import numpy as np
from kiss_icp.datasets import dataset_factory
from kiss_icp.preprocess import Preprocessor

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")

BAG, GT = Path("data/church_02_cut.bag"), "gt/church_02_gt-tum.txt"
RUN = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--run=")), "runs/indoor_detail_base_overlapfix")
_ms = lambda key, default: float(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{key}=")), default)) / 1000
OFFSET = _ms("eval-offset-ms", -70)          # χρονική μετατόπιση για τη σύγκριση ποζών με το GT
DESKEW_OFFSET = _ms("deskew-offset-ms", -70)  # χρονική μετατόπιση του GT όταν --deskew=gt (σωστό: 0, #012)
FIG = Path("docs/figures/i1_matches_stairs.png")
DESKEW = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--deskew=")), "kiss")
SEGMENTS = {"όροφος (αναφορά)": (1200, 1400), "σκάλα πάνω": (820, 990), "σκάλα κάτω": (2014, 2258)}
DELTAS, STEP = (1, 5), 2
W, UP, MIN_RANGE = 1024, 8, 1.0          # στήλες πανοράματος · κατακόρυφη μεγέθυνση · κόβει τον χειριστή
RATIO, RANSAC_THR, RANSAC_IT, MIN_INL = 0.75, 0.15, 400, 10


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "timestamp", "ring"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = ~np.isnan(xyz).any(axis=1) & (np.linalg.norm(xyz, axis=1) > MIN_RANGE)
    return (xyz[ok], s["timestamp"].astype(np.float64)[ok],
            s["intensity"].astype(np.float64)[ok], s["ring"].astype(np.int64)[ok])


def panorama(xyz_raw, ts, inten, ring, deskew_delta, pre):
    """(εικόνα uint8 για τον ανιχνευτή, 3D σημείο ανά pixel, μάσκα έγκυρων pixel)."""
    des = pre.preprocess(xyz_raw, ts, deskew_delta)           # χωρίς όριο απόστασης: γραμμή i = σημείο i
    assert len(des) == len(xyz_raw)
    elev = np.arctan2(xyz_raw[:, 2], np.linalg.norm(xyz_raw[:, :2], axis=1))
    rings = np.unique(ring)
    order = rings[np.argsort([-elev[ring == r].mean() for r in rings])]
    pos = np.empty(rings.max() + 1, dtype=np.int64); pos[order] = np.arange(len(order))
    row = pos[ring]
    col = ((np.arctan2(des[:, 1], des[:, 0]) + np.pi) / (2 * np.pi) * W).astype(int) % W
    H = len(rings)
    img = np.full((H, W), np.nan); P = np.zeros((H, W, 3)); valid = np.zeros((H, W), bool)
    img[row, col] = inten; P[row, col] = des; valid[row, col] = True
    for r in range(H):                                          # κενά pixel: οριζόντια παρεμβολή
        v = valid[r]
        if v.sum() > 1:
            img[r, ~v] = np.interp(np.where(~v)[0], np.where(v)[0], img[r, v])
    img = np.nan_to_num(img, nan=0.0)
    big = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H * UP), interpolation=cv2.INTER_LINEAR)
    return big, P, valid


def lookup(P, valid, kp):
    x, y = kp.pt
    r = int(round((y + 0.5) / UP - 0.5)); c = int(round(x)) % W
    r = min(max(r, 0), P.shape[0] - 1)
    for dc in (0, -1, 1, -2, 2):
        cc = (c + dc) % W
        if valid[r, cc]:
            return P[r, cc]
    return None


def kabsch(A, B):
    """R, t με A ≈ R·B + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = ca - R @ cb
    return T


def ransac(A, B, rng):
    best = None
    for _ in range(RANSAC_IT):
        i = rng.choice(len(A), 3, replace=False)
        T = kabsch(A[i], B[i])
        inl = np.linalg.norm(A - (B @ T[:3, :3].T + T[:3, 3]), axis=1) < RANSAC_THR
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_INL:
        return None, 0, best
    return kabsch(A[best], B[best]), int(best.sum()), best


def angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def tilt(Rw, E):
    return np.degrees(np.arccos(np.clip((Rw @ E @ Rw.T)[2, 2], -1, 1)))


def main():
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN))
    _, G = interpolate(gs, gT, st + OFFSET)
    _, Gd = interpolate(gs, gT, st + DESKEW_OFFSET)
    assert DESKEW in ("kiss", "gt"), DESKEW
    inv = np.linalg.inv

    need = set()
    for a, b in SEGMENTS.values():
        for k in range(a, b, STEP):
            for d in DELTAS:
                if k + d < b:
                    need |= {k, k + d}
    last = max(need)

    ds = dataset_factory(dataloader="rosbag", data_dir=BAG, sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    pre = Preprocessor(1e9, 0.0, True, 0)
    sift = cv2.SIFT_create()
    pano = {}
    for k in range(last + 1):
        xyz, ts, inten, ring = ds[k]
        if k not in need:
            continue
        if DESKEW == "kiss":
            delta = inv(E[k - 2]) @ E[k - 1] if k >= 2 else np.eye(4)   # last_delta του KISS
        else:
            delta = inv(Gd[k - 1]) @ Gd[k] if k >= 1 else np.eye(4)     # πραγματική κίνηση της σάρωσης
        big, P, valid = panorama(xyz, ts, inten, ring, delta, pre)
        kps, desc = sift.detectAndCompute(big, None)
        pano[k] = (big, P, valid, kps, desc)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    rng = np.random.default_rng(0)
    rows, example = [], None
    for name, (a, b) in SEGMENTS.items():
        for d in DELTAS:
            for k in range(a, b, STEP):
                if k + d >= b:
                    continue
                big1, P1, v1, kp1, d1 = pano[k]
                big2, P2, v2, kp2, d2 = pano[k + d]
                dG, dE = inv(G[k]) @ G[k + d], inv(E[k]) @ E[k + d]
                icp_err = angle((inv(dG) @ dE)[:3, :3])
                icp_tilt = tilt(G[k][:3, :3], (inv(dG) @ dE)[:3, :3])
                feat_err = feat_tilt = feat_vs_icp = np.nan; n_inl = 0
                if d1 is not None and d2 is not None and len(kp1) > 5 and len(kp2) > 5:
                    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance]
                    pairs = [(lookup(P1, v1, kp1[m.queryIdx]), lookup(P2, v2, kp2[m.trainIdx]), m) for m in good]
                    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
                    if len(pairs) >= MIN_INL:
                        A = np.array([p[0] for p in pairs]); B = np.array([p[1] for p in pairs])
                        T, n_inl, inl = ransac(A, B, rng)
                        if T is not None:
                            feat_err = angle((inv(dG) @ T)[:3, :3])
                            feat_tilt = tilt(G[k][:3, :3], (inv(dG) @ T)[:3, :3])
                            feat_vs_icp = angle((inv(dE) @ T)[:3, :3])
                            if name.startswith("σκάλα κάτω") and d == 1 and example is None and n_inl > 30:
                                example = (big1, kp1, big2, kp2, [p[2] for p, ok in zip(pairs, inl) if ok], k)
                rows.append((name, d, icp_err, icp_tilt, feat_err, feat_tilt, n_inl, feat_vs_icp))

    print(f"deskew: {DESKEW}{f' (τ = {DESKEW_OFFSET*1000:+.0f} ms)' if DESKEW == 'gt' else ''} · σύγκριση με GT στα {OFFSET*1000:+.0f} ms · ICP = baseline `{RUN}` · κάθε {STEP}η σάρωση\n")
    print(f"{'τμήμα':18}{'Δ':>3}{'ζεύγη':>7}{'επιτυχία':>10}{'inliers':>9}"
          f"{'στροφή ICP':>12}{'στροφή εικόνας':>16}{'κλίση ICP':>11}{'κλίση εικόνας':>15}{'εικόνα καλύτερη':>17}{'εικόνα↔ICP':>12}")
    print(f"{'':18}{'':>3}{'':>7}{'':>10}{'(διάμ.)':>9}{'(διάμ. °)':>12}{'(διάμ. °)':>16}{'(διάμ. °)':>11}{'(διάμ. °)':>15}{'(% ζευγών)':>17}{'(διάμ. °)':>12}")
    R = np.array(rows, dtype=object)
    for name in SEGMENTS:
        for d in DELTAS:
            m = (R[:, 0] == name) & (R[:, 1] == d)
            icp, icpt, fe, fet, ni, fvi = (R[m, j].astype(float) for j in (2, 3, 4, 5, 6, 7))
            ok = ~np.isnan(fe)
            better = np.mean(fe[ok] < icp[ok]) * 100 if ok.any() else np.nan
            print(f"{name:18}{d:>3}{m.sum():>7}{100*ok.mean():>9.0f}%{np.median(ni[ok]) if ok.any() else 0:>9.0f}"
                  f"{np.median(icp):>12.2f}{np.nanmedian(fe):>16.2f}{np.median(icpt):>11.2f}{np.nanmedian(fet):>15.2f}{better:>16.0f}%{np.nanmedian(fvi):>12.2f}")

    if example and DESKEW == "kiss":
        big1, kp1, big2, kp2, matches, k = example
        vis = cv2.drawMatches(big1, kp1, big2, kp2, matches[:80], None, matchColor=(0, 200, 0),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        vis = cv2.resize(vis, (vis.shape[1] // 2, vis.shape[0] // 2))
        FIG.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(FIG), vis)
        print(f"\nπαράδειγμα αντιστοιχίσεων (σκάλα κάτω, σαρώσεις {k}–{k+1}, inliers) → {FIG}")


if __name__ == "__main__":
    main()
