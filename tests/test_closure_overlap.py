"""Ο έλεγχος επικάλυψης των loop closures δίνει λόγο στο [0, 1], για κάθε πυκνότητα χάρτη.

    python tests/test_closure_overlap.py

(a) Συνθετικά, με γνωστή απάντηση: ίδιος χάρτης → 1· μακρινός → 0.
(b) Ανεξαρτησία από την πυκνότητα: οι ίδιες επιφάνειες σε 0.25 m και σε 0.5 m → ~1.
    Εδώ ο παλιός τύπος ξεπερνούσε το 1.
(c) Πραγματικοί τοπικοί χάρτες των closures από τα runs (αν υπάρχουν), ως γράφονται (παγκόσμιο πλαίσιο):
    ο νέος τύπος δίνει λόγο στο [0, 1] και πάνω από το κατώφλι 0.4 → τα closures που έγιναν
    δεκτά είναι γνήσια. (Τα PLY είναι περιστραμμένα ως προς το πλέγμα όπου δημιουργήθηκαν,
    οπότε εδώ ο παλιός τύπος υπερμετρά ακόμη και στο indoor_fast — γι' αυτό υπάρχει το (d).)
(d) Η ακριβής συνθήκη: στο τοπικό πλαίσιο, len(points) == voxels ⇔ local_mapper.voxel_size ==
    density_map_resolution. Άρα στο indoor_fast ο νέος τύπος == ο παλιός· στο indoor_detail όχι.
"""
import glob
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from kiss_slam.loop_closer import local_maps_overlap
from kiss_slam.voxel_map import VoxelMap

RES, THRESHOLD = 0.5, 0.4
ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def old_overlap(src, tgt, pose, res):
    """Ο τύπος πριν τη διόρθωση: target μετρημένο ως len(points)."""
    u = VoxelMap(res)
    u.integrate_frame(src, pose)
    ns = u.num_voxels()
    nt = len(tgt)
    u.add_points(tgt)
    return (ns + nt - u.num_voxels()) / min(ns, nt)


def room(spacing):
    """Τέσσερις τοίχοι και πάτωμα ενός δωματίου 10×6×3 m, με δεδομένη πυκνότητα."""
    g = np.arange(0, 10 + 1e-9, spacing); h = np.arange(0, 3 + 1e-9, spacing); w = np.arange(0, 6 + 1e-9, spacing)
    walls = [np.array([[x, 0, z] for x in g for z in h]), np.array([[x, 6, z] for x in g for z in h]),
             np.array([[0, y, z] for y in w for z in h]), np.array([[10, y, z] for y in w for z in h]),
             np.array([[x, y, 0] for x in g for y in w])]
    return np.vstack(walls).astype(np.float64)


def voxelised(points, voxel):
    m = VoxelMap(voxel); m.add_points(points); return np.asarray(m.point_cloud())


I = np.eye(4)
print("(a) συνθετικά")
A = voxelised(room(0.1), RES)
check("ίδιος χάρτης, ταυτοτικός μετασχηματισμός → 1", abs(local_maps_overlap(A, A, I, RES) - 1) < 1e-9,
      f"= {local_maps_overlap(A, A, I, RES):.3f}")
far = I.copy(); far[:3, 3] = [100, 0, 0]
check("χάρτης μετατοπισμένος 100 m → 0", local_maps_overlap(A, A, far, RES) == 0,
      f"= {local_maps_overlap(A, A, far, RES):.3f}")

print("(b) ανεξαρτησία από την πυκνότητα του χάρτη")
dense, coarse = voxelised(room(0.05), 0.25), voxelised(room(0.05), 0.5)
new_v, old_v = local_maps_overlap(coarse, dense, I, RES), old_overlap(coarse, dense, I, RES)
check("ίδιες επιφάνειες, target 0.25 m, source 0.5 m → ~1", 0.9 < new_v <= 1.0,
      f"νέος = {new_v:.3f}  (παλιός = {old_v:.3f})")


def se3(v):
    T = np.eye(4); T[:3, :3] = R.from_quat(v[3:7]).as_matrix(); T[:3, 3] = v[:3]; return T


print("(c) πραγματικοί χάρτες των closures")
for run in ("indoor_fast_base", "indoor_detail_base"):
    graphs = sorted(glob.glob(f"runs/{run}/*/local_maps/local_map_graph.g2o"))
    if not graphs:
        print(f"  (παραλείπεται: δεν υπάρχει runs/{run})"); continue
    V, E = {}, []
    for line in open(graphs[-1]):
        t = line.split()
        if t and t[0].startswith("VERTEX_SE3"): V[int(t[1])] = se3(np.array(list(map(float, t[2:9]))))
        elif t and t[0].startswith("EDGE_SE3"): E.append((int(t[1]), int(t[2])))
    plys = graphs[-1].replace("local_map_graph.g2o", "plys")
    for query, ref in [(a, b) for a, b in E if abs(a - b) > 1]:
        src = o3d.io.read_point_cloud(f"{plys}/{ref:06d}.ply"); tgt = o3d.io.read_point_cloud(f"{plys}/{query:06d}.ply")
        src, tgt = np.asarray(src.points, dtype=np.float64), np.asarray(tgt.points, dtype=np.float64)
        # Το LocalMap.write() γράφει τα PLY ήδη μετασχηματισμένα με το keypose, δηλαδή σε
        # παγκόσμιες συντεταγμένες μετά τη βελτιστοποίηση → η σχετική θέση είναι ταυτοτική.
        pose = np.eye(4)
        new_v, old_v = local_maps_overlap(src, tgt, pose, RES), old_overlap(src, tgt, pose, RES)
        tag = f"{run}: closure {query}↔{ref}"
        check(f"{tag}: νέος στο [0,1], πάνω από το κατώφλι", 0 <= new_v <= 1 and new_v > THRESHOLD,
              f"νέος {new_v:.3f} · παλιός {old_v:.3f}")

print("(d) πότε ο παλιός τύπος ήταν σωστός: χάρτης αποθηκευμένος στην ίδια ανάλυση και στο ίδιο πλαίσιο")
# Στον πραγματικό κώδικα ο target είναι στο ΤΟΠΙΚΟ του πλαίσιο, στο ίδιο πλέγμα όπου δημιουργήθηκε.
# Εκεί len(points) == voxels ακριβώς όταν local_mapper.voxel_size == density_map_resolution,
# οπότε ο νέος τύπος δίνει ό,τι και ο παλιός (indoor_fast)· αλλιώς ο παλιός υπερμετρά (indoor_detail).
scan = room(0.03) + np.random.default_rng(0).normal(0, 0.01, (len(room(0.03)), 3))
for local_voxel in (0.5, 0.25):
    local_map = VoxelMap(local_voxel); local_map.integrate_frame(scan, I)
    # Ό,τι φτάνει στον loop closer: finalize_local_map → open3d_pcd_with_normals() →
    # PerVoxelPointAndNormal(), δηλαδή ΕΝΑ σημείο (ο μέσος όρος) ανά voxel — όχι το
    # point_cloud(), που κρατά πολλά σημεία ανά voxel.
    pts = local_map.open3d_pcd_with_normals().point.positions.numpy().astype(np.float64)
    at_res = VoxelMap(RES); at_res.add_points(pts)
    n_pts, n_vox = len(pts), at_res.num_voxels()
    if local_voxel == RES:
        check(f"τοπικός χάρτης {local_voxel} m: σημεία == voxels στα {RES} m → νέος == παλιός",
              n_pts == n_vox, f"({n_pts} σημεία, {n_vox} voxels)")
    else:
        check(f"τοπικός χάρτης {local_voxel} m: σημεία > voxels στα {RES} m → ο παλιός υπερμετρούσε",
              n_pts > 2 * n_vox, f"({n_pts} σημεία, {n_vox} voxels, ×{n_pts / n_vox:.1f})")

print("\nRESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
