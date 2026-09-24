#!/usr/bin/env python3
"""Αξιολόγηση τροχιάς KISS-SLAM έναντι του Ground Truth του Oxford Spires.

Τι κάνει
--------
1. Διαβάζει το GT (`gt-tum.txt`, TUM format, απόλυτα Unix timestamps).
2. Μετατρέπει τις πόζες από το **base** frame στο **lidar** frame, με το extrinsic
   του dataset.  Το wiki δίνει
       T_base_lidar_t_xyz_q_xyzw = [0, 0, 0.124, 0, 0, 1, 0]
   ⚠️ Το `q_xyzw = (0,0,1,0)` ΔΕΝ είναι identity: είναι στροφή **180° περί τον z**.
3. Παρεμβάλλει το GT στα timestamps των δικών μας scans (SLERP για στροφή,
   γραμμικά για μετατόπιση).
4. Ευθυγραμμίζει την εκτίμηση με το GT (Umeyama, **χωρίς** scale — rigid SE(3)).
5. Υπολογίζει ATE (RMSE/mean/median/max), RPE ανά 1 s, και — ειδικά για το concept —
   σύγκριση του **κατακόρυφου (z) προφίλ**, εκεί όπου αναμένεται το drift της σκάλας.

Χρήση
-----
    python scripts/evaluate_gt.py gt/church_02_gt-tum.txt runs/base runs/int
    python scripts/evaluate_gt.py <gt.txt> <run_dir> [<run_dir> ...] [--offset-ms=-70]

--offset-ms: μετατόπιση των χρονοσφραγίδων μας πριν την παρεμβολή του GT (default 0, ώστε τα
νούμερα του log να αναπαράγονται). Το σφάλμα ανά σάρωση ελαχιστοποιείται στα −70…−75 ms· επηρεάζει
κυρίως το RPE στροφής, ελάχιστα το ATE. Βλ. docs/experiment_log.md, #010.

Κάθε `run_dir` είναι είτε ο φάκελος `slam_output/<timestamp>/` είτε ο γονέας του.
"""
import sys
import glob
import os
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

# Extrinsic του dataset: πόζα του lidar frame εκφρασμένη στο base frame.
T_BASE_LIDAR_TXYZ = np.array([0.0, 0.0, 0.124])
T_BASE_LIDAR_QXYZW = np.array([0.0, 0.0, 1.0, 0.0])   # 180° περί z — ΟΧΙ identity


def se3(t, q_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(q_xyzw).as_matrix()
    T[:3, 3] = t
    return T


def load_tum(path):
    """TUM: timestamp tx ty tz qx qy qz qw  →  (stamps, Nx4x4)."""
    d = np.loadtxt(path)
    if d.ndim == 1:
        d = d[None]
    stamps = d[:, 0]
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :3] = R.from_quat(d[:, 4:8]).as_matrix()
    T[:, :3, 3] = d[:, 1:4]
    return stamps, T


def base_to_lidar(T_world_base):
    """T_world_lidar = T_world_base @ T_base_lidar."""
    T_bl = se3(T_BASE_LIDAR_TXYZ, T_BASE_LIDAR_QXYZW)
    return T_world_base @ T_bl


def interpolate(gt_stamps, gt_T, query):
    """Παρεμβολή GT στα `query` timestamps. Επιστρέφει (mask, Nx4x4)."""
    ok = (query >= gt_stamps[0]) & (query <= gt_stamps[-1])
    q = query[ok]
    slerp = Slerp(gt_stamps, R.from_matrix(gt_T[:, :3, :3]))
    rot = slerp(q).as_matrix()
    pos = np.column_stack([np.interp(q, gt_stamps, gt_T[:, i, 3]) for i in range(3)])
    T = np.tile(np.eye(4), (len(q), 1, 1))
    T[:, :3, :3] = rot
    T[:, :3, 3] = pos
    return ok, T


def umeyama_rigid(src, dst):
    """Rigid SE(3) (χωρίς scale) που φέρνει το `src` πάνω στο `dst`. Kabsch."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, _, Vt = np.linalg.svd(S)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[2, 2] = -1
    rot = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = mu_d - rot @ mu_s
    return T


def rpe(est, gt, stamps, delta_s=1.0):
    """Relative pose error σε παράθυρο ~delta_s δευτερολέπτων."""
    dt = np.median(np.diff(stamps))
    k = max(1, int(round(delta_s / dt)))
    n = len(est) - k
    if n <= 0:
        return np.array([]), np.array([]), k
    # All windows at once (was a loop over poses; same numbers).  err = inv(inv(G_i) G_i+k) inv(E_i) E_i+k.
    d_est = np.linalg.inv(est[:n]) @ est[k:k + n]
    d_gt = np.linalg.inv(gt[:n]) @ gt[k:k + n]
    err = np.linalg.inv(d_gt) @ d_est
    trans = np.linalg.norm(err[:, :3, 3], axis=1)
    ca = np.clip((np.trace(err[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)
    return trans, np.degrees(np.arccos(ca)), k


def find_tum(run_dir):
    hits = glob.glob(os.path.join(run_dir, "**", "*_poses_tum.txt"), recursive=True)
    if not hits:
        raise SystemExit(f"Δεν βρέθηκε *_poses_tum.txt κάτω από {run_dir}")
    return sorted(hits)[-1]


def arm_of(run_dir):
    """Διαβάζει από το slam_config.yaml ποιος βραχίονας ήταν (και με ποιο mode)."""
    import yaml
    for cfg in glob.glob(os.path.join(run_dir, "**", "slam_config.yaml"), recursive=True):
        intensity = (yaml.safe_load(open(cfg)) or {}).get("intensity", {})
        if not intensity.get("enabled", False):
            return "BASELINE"
        # Runs πριν από την εισαγωγή του mode ήταν πάντα "refine".
        return f"INT-{intensity.get('mode', 'refine')}"
    return "?"


def evaluate(gt_stamps, gt_T, run_dir, offset=0.0):
    path = find_tum(run_dir)
    st, est = load_tum(path)
    ok, gt_i = interpolate(gt_stamps, gt_T, st + offset)
    est = est[ok]
    st = st[ok]
    if len(est) < 10:
        raise SystemExit(f"Πολύ λίγη επικάλυψη GT για {run_dir} ({len(est)} πόζες)")

    align = umeyama_rigid(est[:, :3, 3], gt_i[:, :3, 3])
    est_a = align @ est

    err = np.linalg.norm(est_a[:, :3, 3] - gt_i[:, :3, 3], axis=1)
    z_err = est_a[:, 2, 3] - gt_i[:, 2, 3]
    rt, rr, k = rpe(est_a, gt_i, st)

    return dict(
        name=os.path.basename(os.path.normpath(run_dir)), arm=arm_of(run_dir),
        n=len(est), frac=ok.mean(),
        ate_rmse=np.sqrt((err ** 2).mean()), ate_mean=err.mean(),
        ate_med=np.median(err), ate_max=err.max(),
        z_rmse=np.sqrt((z_err ** 2).mean()), z_final=z_err[-1], z_max=np.abs(z_err).max(),
        gt_z_span=np.ptp(gt_i[:, 2, 3]), est_z_span=np.ptp(est_a[:, 2, 3]),
        gt_z_net=gt_i[-1, 2, 3] - gt_i[0, 2, 3], est_z_net=est_a[-1, 2, 3] - est_a[0, 2, 3],
        gt_path=np.linalg.norm(np.diff(gt_i[:, :3, 3], axis=0), axis=1).sum(),
        est_path=np.linalg.norm(np.diff(est_a[:, :3, 3], axis=0), axis=1).sum(),
        rpe_t=rt.mean(), rpe_r=rr.mean(), k=k,
    )


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    offset = next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--offset-ms=")), 0.0) / 1000
    args = [a for a in sys.argv[1:] if not a.startswith("--offset-ms=")]
    gt_path, run_dirs = args[0], args[1:]

    gs, gT = load_tum(gt_path)
    gT = base_to_lidar(gT)
    print(f"GT: {len(gs)} πόζες, {gs[-1]-gs[0]:.1f}s, "
          f"{len(gs)/(gs[-1]-gs[0]):.1f} Hz  (base→lidar extrinsic εφαρμόστηκε, μετατόπιση χρόνου {offset*1000:+.0f} ms)\n")

    res = [evaluate(gs, gT, d, offset) for d in run_dirs]

    w = max(max(len(r["arm"]), len(r["name"])) for r in res) + 2
    def line(lbl, key, fmt="{:.4f}"):
        print(f"{lbl:34}" + "".join(fmt.format(r[key]).rjust(w + 8) for r in res))

    print(f"{'':34}" + "".join(r["name"].rjust(w + 8) for r in res))
    print(f"{'':34}" + "".join(r["arm"].rjust(w + 8) for r in res))
    print("-" * (34 + (w + 8) * len(res)))
    line("πόζες με GT κάλυψη", "n", "{:.0f}")
    line("ATE RMSE (m)", "ate_rmse")
    line("ATE mean (m)", "ate_mean")
    line("ATE median (m)", "ate_med")
    line("ATE max (m)", "ate_max")
    print("-" * (34 + (w + 8) * len(res)))
    line("z σφάλμα RMSE (m)", "z_rmse")
    line("z σφάλμα τελικό (m)", "z_final", "{:+.4f}")
    line("z σφάλμα max |·| (m)", "z_max")
    print("-" * (34 + (w + 8) * len(res)))
    line("z span εκτίμησης (m)", "est_z_span", "{:.3f}")
    line("z span GT (m)", "gt_z_span", "{:.3f}")
    line("z αρχή→τέλος εκτίμησης (m)", "est_z_net", "{:+.3f}")
    line("z αρχή→τέλος GT (m)", "gt_z_net", "{:+.3f}")
    print("-" * (34 + (w + 8) * len(res)))
    line("μήκος διαδρομής εκτίμησης (m)", "est_path", "{:.2f}")
    line("μήκος διαδρομής GT (m)", "gt_path", "{:.2f}")
    print("-" * (34 + (w + 8) * len(res)))
    line(f"RPE μετατόπισης @~1s (m)", "rpe_t")
    line(f"RPE στροφής @~1s (°)", "rpe_r")


if __name__ == "__main__":
    main()
