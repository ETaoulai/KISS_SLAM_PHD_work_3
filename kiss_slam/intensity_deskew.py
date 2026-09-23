"""Ι-3: the motion of a LiDAR scan measured from its intensity image (#018).

The intensity panorama of a spinning LiDAR is a rolling-shutter image: each column was
measured at a different moment of the sweep.  Matching point-like intensity features
(window corners, railings) between two consecutive raw scans, each with its own time,
constrains the motion DURING the sweep independently of the geometry, which ICP cannot:
a scan deskewed with a wrong motion still fits a smooth map (#017).

Models, with time t in scan periods from the START of the later scan (pose(0) = I):
  "cv" constant velocity:      pose(t) = [R(t w), t v]                               (6 unknowns)
  "ca" constant acceleration:  pose(t) = [R(t w + t²/2 α), t v + t²/2 a]            (12 unknowns)
  "car" acceleration in rotation only: [R(t w + t²/2 α), t v]                        (9 unknowns)
       (#020: "ca" improves rotation but its 6 extra translation unknowns hurt translation)
A feature seen as p at t_p in one scan and as q at t_q in the next satisfies
pose(t_p)·p = pose(t_q)·q.  The motion of the later scan is pose(1).  The hand-held sensor's
rotation changes between consecutive scans almost as much as it is (#019), which a single
velocity over the two scans cannot follow; "ca" lets it change.

`ScanMotionEstimator.motion(scan)` returns the motion over one period, as the 4x4 pose of
the later sensor frame in the earlier one (the quantity KISS calls `delta` and deskews
with), estimated from the previous scan and this one; None when it cannot be estimated.
"""
import cv2

cv2.setNumThreads(8)   # όχι όλους τους πυρήνες (Λ.Γ. 18/9)
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

W, UP, MIN_RANGE = 1024, 8, 1.0          # panorama columns · vertical upscaling for SIFT · drop the operator
RATIO, RANSAC_THR, RANSAC_IT, MIN_INL, FIT_THR = 0.75, 0.30, 400, 10, 0.10
EDGE_REL = 0.05                          # sub-pixel: neighbours' ranges within 5 % → same surface
# Inlier test.  "metric": 3D distance < RANSAC_THR (m) — lets through intensity patterns that move WITH the sensor
# (range / incidence falloff on near floors and walls): they match at the same pixel, look motionless, and bias the
# translation toward zero (scale 0.92, corridor 0.84; #023-#024).  "bearing": the direction of M·q must agree with p
# within BEARING_THR (deg) and the range within RANGE_REL — the error metric of an image feature.
INLIER_TEST, BEARING_THR, RANGE_REL = "metric", 0.35, 0.05
# Drop matches that are the same point in the sensor frame (|p − q| < STUCK_MIN m): a real match moves about as much as
# the sensor (~0.1 m per scan); one that does not is a pattern travelling with the sensor (#025). None = keep all.
STUCK_MIN = None
# Apply it only to the near floor (below STUCK_ELEV deg in the sensor frame and closer than STUCK_RANGE m), where the
# patterns are; elsewhere a real match can look motionless when rotation and translation cancel (#026).
STUCK_FLOOR_ONLY, STUCK_ELEV, STUCK_RANGE = False, -10.0, 5.0
# Optional ring × azimuth boolean mask of pixels fixed to the sensor (shadows of the rig, #035); a match whose
# keypoint in the later scan falls on a masked pixel is dropped.  None = off.
PIXEL_MASK = None
# Near-field bias (#039): intensity-pattern matches closer than ~5 m report only ~74 % of the true translation, those
# beyond ~12 m report 100 %.  With TRANS_MIN_RANGE set, the translation is re-estimated with the rotation frozen, from
# the inliers farther than that (metres); the rotation, and the timed curve, are untouched.  None = off.
TRANS_MIN_RANGE = None
TRANS_MIN_PAIRS = 8
# "vector": replace the translation with the far-only estimate (removes the bias, adds variance, #039).
# "magnitude": keep the DIRECTION from all inliers (well estimated) and correct only its LENGTH by the scalar the far
# inliers imply, clipped to TRANS_CLIP — same bias removal, far less variance.
TRANS_MODE, TRANS_CLIP = "vector", (0.7, 1.6)
# Simplest form: the bias is systematic, so multiply every translation by one constant instead of re-estimating it per
# scan (which adds variance, #039).  1.0 = off.
TRANS_FACTOR = 1.0
# "auto" (#039): the correction is measured FROM THE DATA and needs no GT.  Per scan the ratio of the translation the
# far inliers imply to the one all inliers imply is noisy, but its running median over TRANS_AUTO_WINDOW past scans is
# stable — the far matches' lack of bias without their variance, and it follows the scene (a corridor biases more than
# a courtyard).  Causal: a scan is corrected with the median of the scans BEFORE it.
TRANS_AUTO_WINDOW, TRANS_AUTO_MIN = 100, 20


CALIB = None   # {"r_edges", "c_edges", "table"}: median intensity per (range, cos incidence) bin (#025)


def incidence_cos(xyz, knn=10):
    """|cos| of the incidence angle per point, from PCA normals of its 3D neighbours (Open3D).

    Grid neighbours do not work: some rings have only ~250 returns per turn, so adjacent
    pixels of the 1024-column panorama are almost never both measured.
    """
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn))
    n = np.asarray(pc.normals)
    return np.abs(np.sum(n * xyz, axis=1)) / np.maximum(np.linalg.norm(xyz, axis=1), 1e-9)


def calibrated(img, P, valid, cos):
    """Intensity divided by the median intensity of its (range, cos incidence) bin — reflectance-like, ~1 on average."""
    rng_ = np.linalg.norm(P, axis=2)
    ri = np.clip(np.searchsorted(CALIB["r_edges"], rng_) - 1, 0, len(CALIB["r_edges"]) - 2)
    ci = np.clip(np.searchsorted(CALIB["c_edges"], np.nan_to_num(cos, nan=-1)) - 1, -1, len(CALIB["c_edges"]) - 2)
    # unknown incidence: the range-only column (last)
    ci = np.where(np.isnan(cos), CALIB["table"].shape[1] - 1, ci)
    out = img / np.maximum(CALIB["table"][ri, ci], 1e-3)
    return np.where(valid, out, np.nan)


def grid(xyz, ts, inten, ring, with_cos=False):
    """Ring × azimuth grid: (raw intensity, 3D point, time, valid mask[, |cos incidence|])."""
    elev = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
    rings = np.unique(ring)
    order = rings[np.argsort([-elev[ring == r].mean() for r in rings])]
    pos = np.empty(rings.max() + 1, dtype=np.int64); pos[order] = np.arange(len(order))
    row = pos[ring]
    col = ((np.arctan2(xyz[:, 1], xyz[:, 0]) + np.pi) / (2 * np.pi) * W).astype(int) % W
    H = len(rings)
    img = np.full((H, W), np.nan); P = np.zeros((H, W, 3)); T = np.zeros((H, W)); valid = np.zeros((H, W), bool)
    img[row, col] = inten; P[row, col] = xyz; T[row, col] = ts; valid[row, col] = True
    if not with_cos:
        return img, P, T, valid
    C = np.full((H, W), np.nan); C[row, col] = incidence_cos(xyz)
    return img, P, T, valid, C


# How the panorama is built (#029, idea Ι-5).
#   "splat":   each point is dropped into its nearest pixel; empty pixels of a row are filled by linear interpolation
#              of the image.  The sensor fires every 0.6 deg (dual return) while a column is 0.35 deg, so filled and
#              empty columns form a moiré that is the SAME in every turn — fixed with respect to the sensor.
#   "raycast": each pixel is a ray (its laser's elevation, the azimuth of the pixel centre) and takes range,
#              intensity and time by interpolation between the two samples of that laser on either side — the ray
#              meeting the surface through them.  No interpolation across a depth edge (the sample nearer in angle is
#              taken) or across a gap with no return (the pixel stays empty); with a dual return, the nearer surface.
RENDER, RAY_MAX_GAP, RAY_EDGE_REL, DUAL_EPS = "splat", 1.5, 0.05, 0.05     # deg, relative range, deg


def raycast_grid(xyz, ts, inten, ring):
    """Ring × azimuth grid by casting a ray per pixel: (intensity, 3D point, time, valid mask)."""
    el_all = np.arctan2(xyz[:, 2], np.linalg.norm(xyz[:, :2], axis=1))
    rings = np.unique(ring)
    order = rings[np.argsort([-el_all[ring == r].mean() for r in rings])]
    H = len(order)
    img = np.full((H, W), np.nan); P = np.zeros((H, W, 3)); T = np.zeros((H, W)); valid = np.zeros((H, W), bool)
    azc = -180.0 + (np.arange(W) + 0.5) * 360.0 / W
    for row, rg in enumerate(order):
        m = ring == rg
        if m.sum() < 2:
            continue
        az = np.degrees(np.arctan2(xyz[m, 1], xyz[m, 0])); rng_ = np.linalg.norm(xyz[m], axis=1)
        el, it, t = el_all[m], inten[m], ts[m]
        o = np.lexsort((rng_, az)); az, rng_, el, it, t = az[o], rng_[o], el[o], it[o], t[o]
        keep = np.concatenate([[True], np.diff(az) > DUAL_EPS])      # dual return: first (nearer) surface only
        az, rng_, el, it, t = az[keep], rng_[keep], el[keep], it[keep], t[keep]
        n = len(az)
        # neighbours on either side of each pixel centre, with wrap-around
        j = np.searchsorted(az, azc)
        jl, jr = (j - 1) % n, j % n
        al = np.where(j == 0, az[jl] - 360.0, az[jl]); ar = np.where(j == n, az[jr] + 360.0, az[jr])
        gap = ar - al
        ok = gap <= RAY_MAX_GAP
        w = np.clip((azc - al) / np.maximum(gap, 1e-9), 0, 1)
        edge = np.abs(rng_[jr] - rng_[jl]) > RAY_EDGE_REL * np.minimum(rng_[jl], rng_[jr])
        w = np.where(edge, np.round(w), w)                             # depth edge: the sample nearer in angle
        r_ = (1 - w) * rng_[jl] + w * rng_[jr]
        e_ = (1 - w) * el[jl] + w * el[jr]
        a = np.radians(azc)
        P[row, ok] = (r_[:, None] * np.column_stack([np.cos(e_) * np.cos(a), np.cos(e_) * np.sin(a), np.sin(e_)]))[ok]
        img[row, ok] = ((1 - w) * it[jl] + w * it[jr])[ok]
        T[row, ok] = ((1 - w) * t[jl] + w * t[jr])[ok]
        valid[row, ok] = True
    return img, P, T, valid


def panorama(xyz, ts, inten, ring):
    """(uint8 image for the detector, 3D point per pixel, time per pixel, valid mask)."""
    if RENDER == "raycast":
        img, P, T, valid = raycast_grid(xyz, ts, inten, ring)
        img = np.nan_to_num(img, nan=0.0)                              # no return: black, not interpolated
    elif CALIB is None:
        img, P, T, valid = grid(xyz, ts, inten, ring)
    else:
        img, P, T, valid, C = grid(xyz, ts, inten, ring, with_cos=True)
        img = calibrated(img, P, valid, C)
        img = np.clip(img * 80.0, 0, 255)          # median reflectance → grey 80
    H = img.shape[0]
    for r in range(H if RENDER == "splat" else 0):
        v = valid[r]
        if v.sum() > 1:
            img[r, ~v] = np.interp(np.where(~v)[0], np.where(v)[0], img[r, v])
    img = np.nan_to_num(img, nan=0.0)
    big = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (W, H * UP), interpolation=cv2.INTER_LINEAR)
    return big, P, T, valid


def lookup(P, T, valid, kp, subpixel=False):
    """3D point and time at a keypoint.

    Nearest pixel by default.  One column is 0.35 deg and one ring 0.5-1 deg, so the nearest
    pixel alone carries an error of that order.  With `subpixel`, bilinear interpolation of
    point and time over the 4 surrounding pixels when all are valid and on one surface
    (ranges within EDGE_REL); otherwise the nearest pixel.
    """
    x, y = kp.pt
    rf = (y + 0.5) / UP - 0.5
    if subpixel:
        r0, c0 = int(np.floor(rf)), int(np.floor(x))
        if 0 <= r0 < P.shape[0] - 1:
            rr = [r0, r0, r0 + 1, r0 + 1]; cc = [c0 % W, (c0 + 1) % W, c0 % W, (c0 + 1) % W]
            if valid[rr, cc].all():
                rng = np.linalg.norm(P[rr, cc], axis=1)
                if rng.max() - rng.min() < EDGE_REL * rng.min():
                    a, b = rf - r0, x - c0
                    w = np.array([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b])
                    return w @ P[rr, cc], w @ T[rr, cc]
    r = min(max(int(round(rf)), 0), P.shape[0] - 1); c = int(round(x)) % W
    for dc in (0, -1, 1, -2, 2):
        cc = (c + dc) % W
        if valid[r, cc]:
            return P[r, cc], T[r, cc]
    return None


def kabsch(A, B):
    """4x4 M with A ≈ R·B + t."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((B - cb).T @ (A - ca))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = ca - R @ cb
    return M


def inliers(A, B, M):
    Bm = B @ M[:3, :3].T + M[:3, 3]
    if INLIER_TEST == "metric":
        return np.linalg.norm(A - Bm, axis=1) < RANSAC_THR
    ra, rb = np.linalg.norm(A, axis=1), np.linalg.norm(Bm, axis=1)
    cos = np.sum(A * Bm, axis=1) / (ra * rb)
    return (np.degrees(np.arccos(np.clip(cos, -1, 1))) < BEARING_THR) & (np.abs(ra - rb) < RANGE_REL * ra)


def ransac(A, B, rng):
    best = None
    for _ in range(RANSAC_IT):
        i = rng.choice(len(A), 3, replace=False)
        M = kabsch(A[i], B[i])
        inl = inliers(A, B, M)
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < MIN_INL:
        return None, best
    return kabsch(A[best], B[best]), best


def pose_at(x, t):
    """Rotation vectors and translations of the sensor at times t (in periods) for parameters x."""
    w, v = x[:3], x[3:6]
    rot, tr = t[:, None] * w, t[:, None] * v
    if len(x) >= 9:                                        # "car" and "ca": angular acceleration
        rot = rot + 0.5 * t[:, None] ** 2 * x[6:9]
    if len(x) == 12:                                       # "ca": linear acceleration
        tr = tr + 0.5 * t[:, None] ** 2 * x[9:12]
    return rot, tr


def residual(x, p, tp, q, tq):
    rp, sp = pose_at(x, tp); rq, sq = pose_at(x, tq)
    return (Rotation.from_rotvec(rp).apply(p) + sp - Rotation.from_rotvec(rq).apply(q) - sq).ravel()


def fit_time(p, tp, q, tq, M0, model="cv"):
    """Motion over the later scan, pose(1), using each point's time → (4x4, inliers)."""
    x = np.concatenate([Rotation.from_matrix(M0[:3, :3]).as_rotvec(), M0[:3, 3]])
    keep = np.ones(len(p), bool)
    extra = {"cv": 0, "car": 3, "ca": 6}[model]
    for stage in (["cv"] if model == "cv" else ["cv", model]):
        if stage != "cv":
            x = np.concatenate([x, np.zeros(extra)])    # start from the constant-velocity solution
        for _ in range(3):
            x = least_squares(residual, x, args=(p[keep], tp[keep], q[keep], tq[keep]),
                              loss="soft_l1", f_scale=0.05).x
            r = np.linalg.norm(residual(x, p, tp, q, tq).reshape(-1, 3), axis=1)
            keep = r < FIT_THR
            if keep.sum() < MIN_INL:
                return None, keep
    rot, tr = pose_at(x, np.array([1.0]))
    M = np.eye(4); M[:3, :3] = Rotation.from_rotvec(rot[0]).as_matrix(); M[:3, 3] = tr[0]
    fit_time.last_params = x                              # the whole curve, for deskew_curve (#033)
    return M, keep


def features(xyz, ts, inten, ring, sift):
    """Panorama + SIFT of one raw scan: (P, T, valid, keypoints, descriptors)."""
    ok = ~np.isnan(xyz).any(axis=1) & (np.linalg.norm(xyz, axis=1) > MIN_RANGE)
    big, P, T, valid = panorama(xyz[ok], ts[ok], inten[ok], ring[ok])
    kps, desc = sift.detectAndCompute(big, None)
    return P, T, valid, kps, desc, ts[ok].min()


_DEFAULT = object()   # "use the module-level knob" (scripts set STUCK_* after import)


def match_motion(f1, f2, period, rng, bf, model="cv", subpixel=False,
                 stuck_min=_DEFAULT, floor_only=_DEFAULT, elev=_DEFAULT, range_=_DEFAULT):
    """Motion of the later of two consecutive scans → (rigid M0, timed M1, n inliers).

    The stuck-match filter (#025-#027) takes its knobs from the arguments; each one left at
    its default reads the module global (STUCK_MIN, STUCK_FLOOR_ONLY, STUCK_ELEV, STUCK_RANGE).
    """
    if stuck_min is _DEFAULT:
        stuck_min = STUCK_MIN
    if floor_only is _DEFAULT:
        floor_only = STUCK_FLOOR_ONLY
    if elev is _DEFAULT:
        elev = STUCK_ELEV
    if range_ is _DEFAULT:
        range_ = STUCK_RANGE
    # Reset before any early return: a failed scan must not leave the previous scan's ratio for
    # ScanMotionEstimator to append again (TRANS_MODE "auto").
    match_motion.last_ratio = None
    P1, T1, v1, kp1, d1, _ = f1; P2, T2, v2, kp2, d2, t_start = f2
    if d1 is None or d2 is None or len(kp1) < 2 or len(kp2) < 2:
        return None, None, 0
    good = [m for m, n in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * n.distance]
    if PIXEL_MASK is not None:
        def on_mask(kp):
            u, v = kp.pt
            r = min(max(int(round((v + 0.5) / UP - 0.5)), 0), PIXEL_MASK.shape[0] - 1)
            return PIXEL_MASK[r, int(round(u)) % W]
        good = [m for m in good if not on_mask(kp2[m.trainIdx])]
    pairs = [(lookup(P1, T1, v1, kp1[m.queryIdx], subpixel), lookup(P2, T2, v2, kp2[m.trainIdx], subpixel))
             for m in good]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < MIN_INL:
        return None, None, 0
    p = np.array([x[0] for x, _ in pairs]); q = np.array([y[0] for _, y in pairs])
    tp = (np.array([x[1] for x, _ in pairs]) - t_start) / period     # t = 0: start of the later scan
    tq = (np.array([y[1] for _, y in pairs]) - t_start) / period
    if stuck_min is not None:
        moved = np.linalg.norm(p - q, axis=1) >= stuck_min
        if floor_only:
            q_elev = np.degrees(np.arctan2(q[:, 2], np.linalg.norm(q[:, :2], axis=1)))
            moved |= ~((q_elev < elev) & (np.linalg.norm(q, axis=1) < range_))
        p, q, tp, tq = p[moved], q[moved], tp[moved], tq[moved]
        if len(p) < MIN_INL:
            return None, None, 0
    M0, inl = ransac(p, q, rng)
    if M0 is None:
        return None, None, 0
    M1, keep = fit_time(p[inl], tp[inl], q[inl], tq[inl], M0, model)
    if TRANS_FACTOR != 1.0:
        for M in (M0, M1):
            if M is not None:
                M[:3, 3] = TRANS_FACTOR * M[:3, 3]
    if TRANS_MIN_RANGE is not None:
        far = np.linalg.norm(q[inl], axis=1) > TRANS_MIN_RANGE
        if far.sum() >= TRANS_MIN_PAIRS:
            pf, qf = p[inl][far], q[inl][far]
            if TRANS_MODE == "auto":                            # only measure; the estimator applies the running median
                M = M1 if M1 is not None else M0
                t_all = M[:3, 3]; n2 = float(t_all @ t_all)
                if n2 > 1e-12:
                    t_far = np.median(pf - (M[:3, :3] @ qf.T).T, axis=0)
                    match_motion.last_ratio = float(t_far @ t_all) / n2
                match_motion.last_params = fit_time.last_params if M1 is not None else None
                match_motion.last_t_start = t_start
                return M0, M1, int(keep.sum()) if M1 is not None else 0
            for M in (M0, M1):                                  # p = R q + t  (same convention as the GT delta)
                if M is None:
                    continue
                t_far = np.median(pf - (M[:3, :3] @ qf.T).T, axis=0)
                if TRANS_MODE == "magnitude":
                    t_all = M[:3, 3]; n2 = float(t_all @ t_all)
                    if n2 > 1e-12:
                        M[:3, 3] = np.clip(float(t_far @ t_all) / n2, *TRANS_CLIP) * t_all
                else:
                    M[:3, 3] = t_far
    match_motion.last_params = fit_time.last_params if M1 is not None else None
    match_motion.last_t_start = t_start
    return M0, M1, int(keep.sum()) if M1 is not None else 0


def deskew_curve(xyz, ts, params, t_start, period=0.1, min_range=0.0, max_range=1e9):
    """Deskew every point with the sensor pose at ITS OWN time on the fitted curve (#033).

    KISS spreads the whole-scan motion `delta` uniformly over the sweep: p' = exp((s-1) log delta) p.
    Here p' = pose(1)^-1 · pose(tau_i) · p_i with pose(tau) = [R(tau w + tau^2/2 alpha), tau v] from
    the "car"/"ca" fit, tau_i = (t_i - t_start) / period, so the angular acceleration measured in
    the image is used per point, not only for a better total.  Output in the end-of-sweep frame,
    then the KISS range crop (strict inequalities, as Preprocessing.cpp) so the result replaces
    `Preprocessor.preprocess` one for one.  With a "cv" fit this equals the KISS deskew.
    """
    tau = (ts - t_start) / period
    rot, tr = pose_at(params, tau)
    r1, t1 = pose_at(params, np.array([1.0]))
    R1 = Rotation.from_rotvec(r1[0])
    world = Rotation.from_rotvec(rot).apply(xyz) + tr             # in the start-of-sweep frame
    out = R1.inv().apply(world - t1[0])                           # into the end-of-sweep frame
    rng_ = np.linalg.norm(out, axis=1)
    return out[(rng_ < max_range) & (rng_ > min_range)]


class ScanMotionEstimator:
    """Online: feed raw scans in order, get each scan's motion from it and the previous one."""

    def __init__(self, period=0.1, seed=0, model="cv", subpixel=False,
                 stuck_min=_DEFAULT, floor_only=_DEFAULT, elev=_DEFAULT, range_=_DEFAULT):
        """`stuck_min`, `floor_only`, `elev`, `range_`: the stuck-match filter (#025-#027);
        left at their defaults they read the module globals STUCK_* at each call."""
        self.period, self.model, self.subpixel = period, model, subpixel
        self.stuck_min, self.floor_only, self.elev, self.range_ = stuck_min, floor_only, elev, range_
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.rng = np.random.default_rng(seed)
        self.prev = None
        self.ratios = []; self.last_factor = 1.0

    def motion(self, xyz, ts, inten, ring):
        cur = features(xyz, ts, inten, ring, self.sift)
        prev, self.prev = self.prev, cur
        self.last_params, self.last_t_start = None, cur[5]
        if prev is None:
            return None, 0
        M0, M1, n = match_motion(prev, cur, self.period, self.rng, self.bf, self.model, self.subpixel,
                                 self.stuck_min, self.floor_only, self.elev, self.range_)
        if TRANS_MODE == "auto" and TRANS_MIN_RANGE is not None:
            f = (np.clip(np.median(self.ratios[-TRANS_AUTO_WINDOW:]), *TRANS_CLIP)
                 if len(self.ratios) >= TRANS_AUTO_MIN else 1.0)      # causal: only the scans before this one
            for M in (M0, M1):
                if M is not None:
                    M[:3, 3] = f * M[:3, 3]
            r = match_motion.last_ratio
            if r is not None and np.isfinite(r):
                self.ratios.append(r)
            self.last_factor = f
        self.last_params = match_motion.last_params if M1 is not None else None
        self.last_t_start = cur[5]
        return M1, n
