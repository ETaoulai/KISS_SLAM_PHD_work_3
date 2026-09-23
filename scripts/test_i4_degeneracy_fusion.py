#!/usr/bin/env python3
"""Εκφύλιση του ICP και συνδυασμός με την κίνηση της εικόνας intensity — έλεγχος εκτός SLAM (#023).

Για κάθε σάρωση k:
  1. deskew με την κίνηση της εικόνας (όπως στο run `indoor_detail_i3car_sp_deskew`), κάθετες επιφάνειας (PCA, Open3D)·
  2. πίνακας πληροφορίας point-to-plane της μετατόπισης: H = mean(n nᵀ) (3×3). Οι ιδιοτιμές του λένε πόσο «κλειδώνει»
     η γεωμετρία σε κάθε κατεύθυνση· μικρή ιδιοτιμή = κατεύθυνση όπου ο ICP γλιστρά (ανάλυση εκφύλισης τύπου
     Zhang & Singh, ICRA 2016 — να επιβεβαιωθεί η αναφορά)·
  3. συνδυασμός: η μετατόπιση ICP (inv(E_{k-1})·E_k του run) αναλύεται στις ιδιοκατευθύνσεις του H· σε όσες έχουν
     ιδιοτιμή < τ, η συνιστώσα αντικαθίσταται από την κίνηση της εικόνας. Η στροφή μένει του ICP.
  4. σφάλμα έναντι GT κατά μήκος / πλάγια / κατακόρυφα της διαδρομής, για ICP, εικόνα και συνδυασμό (για διάφορα τ).

    python scripts/test_i4_degeneracy_fusion.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import open3d as o3d
from kiss_icp.datasets import dataset_factory
from kiss_icp.preprocess import Preprocessor

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")
RUN, MOTION = "runs/indoor_detail_i3car_sp_deskew", "runs/i3_motion_car_sp.npz"
WINDOWS = {"ισόγειο 300–400": (300, 400), "διάδρομος 400–500": (400, 500), "ισόγειο 500–600": (500, 600),
           "σκάλα πάνω 820–990": (820, 990), "όροφος 1200–1400": (1200, 1400)}
TAUS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 1.01]     # 0 = μόνο ICP · 1.01 = όλη η μετατόπιση από την εικόνα
VOXEL, MAX_RANGE = 0.25, 50.0


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "timestamp"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    ok = ~np.isnan(xyz).any(axis=1)
    return xyz[ok], s["timestamp"].astype(np.float64)[ok]


def info_matrix(xyz):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz)).voxel_down_sample(VOXEL)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
    n = np.asarray(pc.normals)
    return n.T @ n / len(n)                                   # ιδιοτιμές αθροίζουν σε 1


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN)); _, G = interpolate(gs, gT, st - 0.010)
    M = np.load(MOTION)["motion"]
    inv = np.linalg.inv
    need = {k for a, b in WINDOWS.values() for k in range(a + 1, b)}
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    pre = Preprocessor(MAX_RANGE, 1.0, True, 0)
    H = {}
    for k in range(max(need) + 1):
        xyz, ts = ds[k]
        if k in need:
            H[k] = info_matrix(pre.preprocess(xyz, ts, M[k]))

    print("Ιδιοτιμές του πίνακα πληροφορίας μετατόπισης (διάμεσος, μικρότερη → μεγαλύτερη) και σε ποια κατεύθυνση\n"
          "πέφτει η μικρότερη (|συνημίτονο| με κατά μήκος / πλάγια / κατακόρυφα):")
    res = {}
    for name, (a, b) in WINDOWS.items():
        P = G[a:b, :2, 3]; c = P - P.mean(0); _, _, Vt = np.linalg.svd(c)
        axes = np.array([[*Vt[0], 0.0], [-Vt[0, 1], Vt[0, 0], 0.0], [0.0, 0.0, 1.0]])   # κατά μήκος, πλάγια, πάνω
        lam, weak, err = [], [], {t: [] for t in TAUS}
        e_img = []
        for k in range(a + 1, b):
            truth = (inv(G[k - 1]) @ G[k])[:3, 3]
            t_icp = (inv(E[k - 1]) @ E[k])[:3, 3]
            t_img = M[k][:3, 3]
            w, V = np.linalg.eigh(H[k])                         # στο πλαίσιο της σάρωσης k ≈ k−1
            lam.append(w)
            Rw = G[k - 1, :3, :3]                                # πλαίσιο αισθητήρα → κόσμος (για τους άξονες)
            weak.append(np.abs(axes @ (Rw @ V[:, 0])))
            for tau in TAUS:
                sel = w < tau
                fused = t_icp + V[:, sel] @ (V[:, sel].T @ (t_img - t_icp))
                err[tau].append(axes @ (Rw @ (fused - truth)))
            e_img.append(axes @ (Rw @ (t_img - truth)))
        lam = np.array(lam); weak = np.array(weak)
        print(f"  {name:20} λ = {np.median(lam[:, 0]):.3f} / {np.median(lam[:, 1]):.3f} / {np.median(lam[:, 2]):.3f}"
              f"   ασθενέστερη ≈ κατά μήκος {np.median(weak[:, 0]):.2f} · πλάγια {np.median(weak[:, 1]):.2f} · πάνω {np.median(weak[:, 2]):.2f}")
        res[name] = (err, np.array(e_img))

    rms = lambda X: "/".join(f"{1000 * np.sqrt(np.mean(np.array(X)[:, i] ** 2)):3.0f}" for i in range(3))
    print("\nΣφάλμα μετατόπισης ανά σάρωση (RMS, mm, κατά μήκος/πλάγια/κατακόρυφα):")
    print(f"{'':22}" + "".join(f"{('ICP' if t == 0 else 'εικόνα' if t > 1 else f'τ={t:.2f}'):>14}" for t in TAUS))
    for name, (err, _) in res.items():
        print(f"  {name:20}" + "".join(f"{rms(err[t]):>14}" for t in TAUS))


if __name__ == "__main__":
    main()
