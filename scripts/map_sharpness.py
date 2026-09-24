#!/usr/bin/env python3
"""Map sharpness against a survey prior map (#048): how well each arm's deskewed scans fit a real surface.

    python scripts/map_sharpness.py <prior map .ply> <ground truth> <frame> <run dir> [<run dir> ...]
                                    [--points=2000000] [--poses=own|gt] [--prior-voxel=<m>] [--out=<csv>]

Each run dir needs deskewed_frames.npz (run_ncd.py --save-frames=<voxel>) and its poses (TUM).  Per run:
1. the map: every deskewed scan (sensor frame) placed at the run's estimated pose;
2. into the prior map's frame: the trajectory aligned rigidly to the ground truth (Umeyama, each run at its
   best time shift, evaluate_ncd.best_offset; the Newer College GT was made by registration to this map, so
   the GT frame is the map frame), then ONE rigid point-to-plane ICP of the whole map against the prior map,
   so a global offset of the trajectory does not count, only how sharp / consistent the map is;
3. a uniform random sample of --points map points; for each, the distance to the prior surface along its
   normal (point-to-plane, nearest prior point); points farther than 0.5 m from any prior point are left out
   (parts the survey does not cover, moving people) and their share is reported.
--poses=gt: the deskewed scans are placed at the GROUND-TRUTH poses (interpolated at the run's best time shift)
instead of the run's own, and no alignment is needed: trajectory errors drop out and only the deskew is measured.
--prior-voxel: voxel-downsample the prior map (after cropping it to the run) before use, e.g. 0.02 for a 1 cm TLS map.
Reported: median and mean point-to-plane distance of the kept points, share within 5 / 10 cm, share kept.
Lower distances = a sharper map.  The prior map is a 5 cm cloud, so ~1-2 cm is the floor of the measure.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import find_tum, interpolate, load_tum, umeyama_rigid  # noqa: E402
from evaluate_ncd import best_offset, load_gt  # noqa: E402

MAX_DIST = 0.5


def load_prior(path):
    return o3d.io.read_point_cloud(str(path))


def prior_around(prior, lo, hi, voxel=None):
    """The prior map inside the box [lo, hi], optionally voxel-downsampled, with normals (estimated if the file has
    none: on the crop only — the christ-church TLS map is ~10x the New College one)."""
    crop = prior.crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi))
    if voxel:
        crop = crop.voxel_down_sample(voxel)
    if not crop.has_normals():
        crop.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(20))
    return crop


def run_map(run, gt_t, gt_T, n_points, rng, use_gt=False):
    st, est = load_tum(find_tum(run))
    f = np.load(sorted(Path(run).glob("*/deskewed_frames.npz"))[-1])    # <run>/<timestamp>/ (and the "latest" link)
    pts, off = f["points"], f["offsets"]
    if len(off) - 1 != len(est):
        raise ValueError(f"{run}: {len(off) - 1} saved scans but {len(est)} poses")
    shift = best_offset(gt_t, gt_T, st, est)
    ok, g = interpolate(gt_t, gt_T, st + shift)
    align = umeyama_rigid(est[ok][:, :3, 3], g[:, :3, 3])
    if use_gt:          # scans at the GT poses (at the run's best time shift): only the deskew is measured
        est, align = g, np.eye(4)                 # one GT pose per kept scan, in order
        pts, off = _only(pts, off, ok)
    counts = np.diff(off)
    # uniform sample over all points of all scans, then place each at its own scan's pose
    pick = np.sort(rng.choice(len(pts), size=min(n_points, len(pts)), replace=False))
    scan = np.searchsorted(off, pick, side="right") - 1
    T = align @ est[scan]
    world = np.einsum("nij,nj->ni", T[:, :3, :3], pts[pick].astype(np.float64)) + T[:, :3, 3]
    return world, shift, counts.sum()


def _only(pts, off, ok):
    """Keep the scans with ok (inside the GT span); returns points and offsets of those scans."""
    keep = np.repeat(ok, np.diff(off))
    return pts[keep], np.concatenate([[0], np.cumsum(np.diff(off)[ok])])


def score(world, prior, prior_tree, prior_n):
    src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
    reg = o3d.pipelines.registration.registration_icp(
        src, prior, 0.3, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    w = world @ reg.transformation[:3, :3].T + reg.transformation[:3, 3]
    d, i = prior_tree.query(w, distance_upper_bound=MAX_DIST, workers=-1)
    keep = np.isfinite(d)
    p2p = np.abs(np.sum((w[keep] - np.asarray(prior.points)[i[keep]]) * prior_n[i[keep]], axis=1))
    icp_shift = np.linalg.norm(reg.transformation[:3, 3])
    return dict(median_cm=100 * np.median(p2p), mean_cm=100 * p2p.mean(), within5=100 * (p2p < 0.05).mean(),
                within10=100 * (p2p < 0.10).mean(), kept=100 * keep.mean(), icp_shift_cm=100 * icp_shift)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    prior_path, gt, frame, runs = args[0], args[1], args[2], [Path(r) for r in args[3:]]
    n_points = int(opts.get("points", 2_000_000))
    gt_t, gt_T, _ = load_gt(gt, frame)
    prior = load_prior(prior_path)
    rows = []
    for run in runs:
        rng = np.random.default_rng(0)
        world, shift, n_all = run_map(run, gt_t, gt_T, n_points, rng, opts.get("poses", "own") == "gt")
        lo, hi = world.min(0) - 1.0, world.max(0) + 1.0                  # the prior map around this run only
        crop = prior_around(prior, lo, hi, float(opts["prior-voxel"]) if "prior-voxel" in opts else None)
        tree = cKDTree(np.asarray(crop.points))
        r = dict(run=run.name, arm=run.name.split("_")[0], time_shift_s=shift, map_points=n_all,
                 **score(world, crop, tree, np.asarray(crop.normals)))
        rows.append(r)
        print(f"{r['run']:18s} median {r['median_cm']:5.2f} cm  mean {r['mean_cm']:5.2f} cm  <5 cm {r['within5']:5.1f} %  "
              f"<10 cm {r['within10']:5.1f} %  kept {r['kept']:5.1f} %  (ICP moved the map {r['icp_shift_cm']:.1f} cm)", flush=True)
    if "out" in opts:
        with open(opts["out"], "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
