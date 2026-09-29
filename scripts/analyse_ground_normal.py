#!/usr/bin/env python3
"""Can a LiDAR scan measure "up"?  The ground-plane normal of raw scans against the true vertical (branch vertical_constraint).

    python scripts/analyse_ground_normal.py <sequence> <ground truth> [--frame=spires] [--topic=/hesai/pandar] [--every=10]
                                            [--max-range=15] [--cone=35] [--thr=0.05] [--cue=ground|walls]

Bodleian's height error follows its tilt error (1 deg over 85 m ~ 1.5 m).  A vertical constraint needs a per-scan measurement
of up that is (a) better than ~0.5 deg and (b) available often.  Per scan (every --every-th, the bag is read in order):
points 1 m .. --max-range, RANSAC for the largest plane whose normal lies within --cone deg of the sensor z (the unit is
held roughly upright; no GT used), inlier distance --thr m, least-squares refit on the inliers.  The normal is compared with
the true up in the sensor frame (third row of the GT rotation at the scan time, frame --frame).  Reports how often a plane
is found, its error, and the error when only large, well-supported planes are kept.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import interpolate  # noqa: E402
from evaluate_ncd import load_gt  # noqa: E402


def ground_plane(xyz, cone_deg, thr, rng, iters=200):
    """(unit normal pointing up in the sensor frame, inlier count) of the largest near-horizontal plane, or (None, 0)."""
    if len(xyz) < 50:
        return None, 0
    cos_cone = np.cos(np.radians(cone_deg))
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = xyz[rng.choice(len(xyz), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if n[2] < 0:
            n = -n
        if n[2] < cos_cone:
            continue
        d = np.abs((xyz - a) @ n)
        inl = d < thr
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None:
        return None, 0
    P = xyz[best]
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[2]
    if n[2] < 0:
        n = -n
    if c @ n > -0.3:                                          # the ground must lie at least 0.3 m below the sensor
        return None, 0
    return n, best_n


def wall_up(xyz, cone_deg, thr, rng, walls=4, min_pts=800, iters=200):
    """"Up" from walls: up to `walls` large planes whose normal is within `cone_deg` of horizontal (sequential RANSAC); up =
    the direction most perpendicular to their normals (smallest eigenvector of sum w n n^T, w = inliers), sign to sensor z.
    Needs >= 2 walls not parallel (the two largest > 20 deg apart in azimuth).  (None, 0) otherwise."""
    sin_cone = np.sin(np.radians(cone_deg))
    pts = xyz.copy()
    normals, weights = [], []
    for _ in range(walls):
        if len(pts) < min_pts:
            break
        best, best_n, best_nv = None, 0, None
        for _ in range(iters):
            a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
            n = np.cross(b - a, c - a)
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            if abs(n[2]) > sin_cone:                          # not a wall (normal too far from horizontal)
                continue
            inl = np.abs((pts - a) @ n) < thr
            if inl.sum() > best_n:
                best, best_n, best_nv = inl, int(inl.sum()), n
        if best is None or best_n < min_pts:
            break
        P = pts[best]
        _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
        normals.append(Vt[2]); weights.append(best_n)
        pts = pts[~best]
    if len(normals) < 2:
        return None, 0
    N = np.array(normals); w = np.array(weights, float)
    az = np.degrees(np.arctan2(N[:, 1], N[:, 0])) % 180
    d = abs(az[0] - az[1]); d = min(d, 180 - d)
    if d < 20:
        return None, 0
    M = (N * w[:, None]).T @ N
    ev, V = np.linalg.eigh(M)
    up = V[:, 0]
    return (up if up[2] > 0 else -up), len(normals)


def main():
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seq, gt = Path(args[0]), args[1]
    frame, topic, every = opts.get("frame", "spires"), opts.get("topic", "/hesai/pandar"), int(opts.get("every", 10))
    max_range, cone, thr = float(opts.get("max-range", 15)), float(opts.get("cone", 35)), float(opts.get("thr", 0.05))
    if opts.get("cue") == "walls":
        max_range = float(opts.get("max-range", 30))            # walls: farther, larger planes
    gt_t, gt_T, _ = load_gt(gt, frame)
    from kiss_icp.datasets.rosbag import RosbagDataset
    from kiss_slam.tools.point_cloud2 import read_point_cloud_raw
    ds = RosbagDataset(seq, topic)
    stamps = []
    ds.read_point_cloud = lambda msg: (stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9),
                                       read_point_cloud_raw(msg))[1]
    rng = np.random.default_rng(0)
    rows = []
    for k in range(len(ds)):
        xyz = ds[k][0]                                          # serial reader: every scan is read, one in `every` used
        if k % every:
            continue
        r = np.linalg.norm(xyz, axis=1)
        pts = xyz[(r > 1.0) & (r < max_range)]
        if len(pts) > 20000:
            pts = pts[rng.choice(len(pts), 20000, replace=False)]
        n, cnt = (wall_up(pts, cone, thr, rng) if opts.get("cue") == "walls" else ground_plane(pts, cone, thr, rng))
        ok, G = interpolate(gt_t, gt_T, np.array([stamps[-1]]))
        if not ok[0]:
            continue
        up = G[0, 2, :3]                                        # world z in the sensor frame
        err = np.degrees(np.arccos(np.clip(n @ up, -1, 1))) if n is not None else np.nan
        tilt = np.degrees(np.arccos(np.clip(up[2], -1, 1)))     # how far the sensor is from upright
        nw = G[0, :3, :3] @ n if n is not None else np.full(3, np.nan)   # the normal in the world frame (terrain test)
        rows.append((stamps[-1], cnt, err, tilt, *(n if n is not None else [np.nan] * 3), *up, *nw))
    a = np.array(rows)
    found = np.isfinite(a[:, 2])
    print(f"{seq.name}: {len(a)} scans checked, plane found in {100 * found.mean():.0f} %; sensor tilt from vertical median "
          f"{np.median(a[:, 3]):.1f} deg (p95 {np.percentile(a[:, 3], 95):.1f})")
    e = a[found, 2]
    print(f"  normal vs true up: median {np.median(e):.2f} deg, p75 {np.percentile(e, 75):.2f}, p90 {np.percentile(e, 90):.2f}, "
          f"p95 {np.percentile(e, 95):.2f}")
    # A constant tilt between the GT world frame and gravity (or the LiDAR mounting) is not measurement error.  The vectors
    # are all near-vertical, so a rotation about the vertical is undetermined: remove only the mean tilt difference (x, y of
    # normal - up, small angles) and re-measure what is left.
    D = a[found, 4:6] - a[found, 7:9]
    bias = D.mean(0)
    e2 = np.degrees(np.linalg.norm(D - bias, axis=1))
    print(f"  constant tilt offset {np.degrees(np.linalg.norm(bias)):.2f} deg; after removing it: median {np.median(e2):.2f}, "
          f"p75 {np.percentile(e2, 75):.2f}, p90 {np.percentile(e2, 90):.2f} deg")
    # Terrain test: is the ground level ON AVERAGE?  Normals in the world frame (via the GT orientation: this measures the
    # terrain, not the method), averaged over windows; the angle of the average to the vertical.
    t0 = a[found, 0] - a[found, 0][0]
    NW = a[found, 10:13]
    for win in (5.0, 30.0, 60.0):
        errs = []
        for s0 in np.arange(0, t0[-1] - win / 2, win):
            k = (t0 >= s0) & (t0 < s0 + win)
            if k.sum() >= 3:
                m = NW[k].mean(0)
                errs.append(np.degrees(np.arccos(np.clip(m[2] / np.linalg.norm(m), -1, 1))))
        errs = np.array(errs)
        if not len(errs):
            print(f"  ground normal averaged over {win:.0f} s: too few scans per window (use a smaller --every)")
            continue
        print(f"  ground normal averaged over {win:.0f} s ({len(errs)} windows): angle to vertical median {np.median(errs):.2f}, "
              f"p90 {np.percentile(errs, 90):.2f}, max {errs.max():.2f} deg")
    for q in (50, 75):
        big = found & (a[:, 1] >= np.percentile(a[found, 1], q))
        eb = a[big, 2]
        print(f"  planes in the top {100 - q} % by support (>= {np.percentile(a[found, 1], q):.0f} points): {100 * big.mean():.0f} % of scans, "
              f"error median {np.median(eb):.2f} deg, p90 {np.percentile(eb, 90):.2f}")


if __name__ == "__main__":
    main()
