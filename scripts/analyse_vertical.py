#!/usr/bin/env python3
"""Where does the vertical drift come from: orientation (tilt) or translation?  (open_tasks B.1, #067)

    python scripts/analyse_vertical.py <ground truth> <run dir> [<run dir> ...] --frame=ncd2020|ncd2021|spires [--window=30]

Each run is associated with the GT as in the official evaluation (evaluate_official: poses at their own instant,
evo association within 10 ms, or the GT interpolated at the pose times) and aligned rigidly to it (Umeyama SE(3) on the
positions, as evo_ape --align), and a constant rotation between the estimate's sensor frame and the GT body frame
(extrinsic, "offset" column) is removed the same way.  Then, for consecutive poses i-1 -> i with the displacement in the sensor frame
d = R_{i-1}^T (p_i - p_{i-1}), the error of the height increment splits EXACTLY into

    dz_est - dz_gt = e_z^T (R_est - R_gt) d_est      (rotation: the sensor frame tilted, the same step goes up or down)
                   + e_z^T R_gt (d_est - d_gt)        (translation: the step itself is wrong in the sensor frame)

(R at pose i-1, GT world z is up).  Summed along the route, the two cumulative curves add up to the height error
z_est - z_gt (up to its value at the first pose).  Also: the tilt error, i.e. the angle between the estimated and the true
"up" seen in the sensor frame, and the height error accumulated per window of --window seconds (worst windows first).
"""
import sys
from pathlib import Path

import numpy as np
from evo.core import sync

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import interpolate, umeyama_rigid  # noqa: E402
from evaluate_ncd import load_gt  # noqa: E402
from evaluate_official import MIN_ASSOCIATED, T_MAX_DIFF, pose_times, to_traj  # noqa: E402


def associated(gt_t, gt_T, run, frame):
    """(t, est, gt) as the official evaluation pairs them."""
    t, T, _ = pose_times(run, frame)
    inside = (t >= gt_t[0] - T_MAX_DIFF) & (t <= gt_t[-1] + T_MAX_DIFF)
    try:
        ref_a, est_a = sync.associate_trajectories(to_traj(gt_t, gt_T), to_traj(t, T), max_diff=T_MAX_DIFF)
        if est_a.num_poses >= MIN_ASSOCIATED * inside.sum():
            return np.asarray(est_a.timestamps), np.array(est_a.poses_se3), np.array(ref_a.poses_se3)
    except sync.SyncException:
        pass
    ok, G = interpolate(gt_t, gt_T, t)
    return t[ok], T[ok], G


def body_offset(est, gt):
    """Constant rotation X with R_est ≈ R_gt · X: the sensor frame of the estimate vs the body frame of the GT (extrinsic /
    frame convention).  Chordal mean of R_gt^T R_est, projected on SO(3)."""
    M = np.einsum("nji,njk->ik", gt[:, :3, :3], est[:, :3, :3])
    U, _, Vt = np.linalg.svd(M)
    return U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def decompose(t, est, gt):
    est = umeyama_rigid(est[:, :3, 3], gt[:, :3, 3]) @ est
    # A constant body-frame rotation X is not an estimation error, but it enters both terms with opposite signs
    # (R_est = R_gt X gives d_est = X^T d_gt: rotation term e_z^T R_gt (I - X^T) d_gt, translation term its negative).
    # Remove it, as the world alignment removes the constant world-frame offset.
    X = body_offset(est, gt)
    est = est.copy()
    est[:, :3, :3] = est[:, :3, :3] @ X.T
    pe, pg = est[:, :3, 3], gt[:, :3, 3]
    Re, Rg = est[:-1, :3, :3], gt[:-1, :3, :3]
    d_est = np.einsum("nji,nj->ni", Re, pe[1:] - pe[:-1])            # R^T (p_i - p_{i-1})
    d_gt = np.einsum("nji,nj->ni", Rg, pg[1:] - pg[:-1])
    rot = np.einsum("nj,nj->n", (Re - Rg)[:, 2, :], d_est)              # e_z^T (R_est - R_gt) d_est
    tra = np.einsum("nj,nj->n", Rg[:, 2, :], d_est - d_gt)              # e_z^T R_gt (d_est - d_gt)
    z_err = pe[:, 2] - pg[:, 2]
    up_e, up_g = est[:, 2, :3], gt[:, 2, :3]                            # world z in the sensor frame = 3rd row of R
    tilt = np.degrees(np.arccos(np.clip(np.einsum("nj,nj->n", up_e, up_g), -1, 1)))
    return dict(offset_deg=float(np.degrees(np.arccos(np.clip((np.trace(X) - 1) / 2, -1, 1)))), t=t - t[0], z_err=z_err, rot=np.r_[0, np.cumsum(rot)], tra=np.r_[0, np.cumsum(tra)], tilt=tilt,
                path=np.r_[0, np.cumsum(np.linalg.norm(pg[1:] - pg[:-1], axis=1))])


def main():
    frame = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--frame=")), None)
    window = float(next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--window=")), 30))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gt_t, gt_T, frame = load_gt(args[0], frame)
    print(f"GT {args[0]} (frame {frame})\n")
    print(f"{'run':<20}{'offset':>7}{'z RMSE':>8}{'z end':>8}{'from rot':>10}{'from trans':>11}{'tilt med':>10}{'tilt p95':>10}"
          f"{'|rot| max':>11}{'|trans| max':>12}   worst {window:.0f} s windows (start s: dz rot / trans m)")
    for r in args[1:]:
        v = decompose(*associated(gt_t, gt_T, Path(r), frame))
        z0 = v["z_err"][0]
        # height error gained per window, split in the two parts
        edges = np.arange(0, v["t"][-1] + window, window)
        k = np.searchsorted(v["t"], edges).clip(0, len(v["t"]) - 1)
        gains = [(v["t"][a], v["rot"][b] - v["rot"][a], v["tra"][b] - v["tra"][a]) for a, b in zip(k[:-1], k[1:]) if b > a]
        worst = sorted(gains, key=lambda g: -abs(g[1] + g[2]))[:3]
        print(f"{Path(r).name:<20}{v['offset_deg']:7.2f}{np.sqrt((v['z_err'] ** 2).mean()):8.3f}{v['z_err'][-1] - z0:+8.2f}"
              f"{v['rot'][-1]:+10.2f}{v['tra'][-1]:+11.2f}{np.median(v['tilt']):10.2f}{np.percentile(v['tilt'], 95):10.2f}"
              f"{np.abs(v['rot']).max():11.2f}{np.abs(v['tra']).max():12.2f}   "
              + "  ".join(f"{s:.0f}: {a:+.2f}/{b:+.2f}" for s, a, b in worst))


if __name__ == "__main__":
    main()
