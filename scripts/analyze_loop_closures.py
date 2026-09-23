#!/usr/bin/env python3
"""Δουλεύει σωστά η συνένωση χαρτών (loop closure) με κατόψεις σε κτήριο με ορόφους;

Για κάθε ζεύγος τοπικών χαρτών σε απόσταση ≥ 4 θέσεων (όσα μπορεί να προτείνει το MapClosures) (nodes) υπολογίζει, με τις ΣΩΣΤΕΣ θέσεις (GT):
  - επικάλυψη 3D (voxels 0.5 m, ίδιος τύπος με τον έλεγχο του loop closer) → «έπρεπε να
    ενωθούν» αν > 0.4, το κατώφλι αποδοχής του KISS·
  - επικάλυψη σε ΚΑΤΟΨΗ (κελιά 0.5 m, χωρίς ύψος) → αυτό «βλέπει» ένας ανιχνευτής κατόψεων·
  - ζεύγη με μεγάλη επικάλυψη σε κάτοψη αλλά μικρή στον χώρο = όροφοι ο ένας πάνω από τον
    άλλον → εκεί μια κάτοψη κινδυνεύει να μπερδευτεί.
Συγκρίνει με τα closures που βρήκε πραγματικά το run.

    python scripts/analyze_loop_closures.py [run_dir] [--offset-ms=-70]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.loop_closer import local_maps_overlap  # noqa: E402

OFFSET = next((float(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--offset-ms=')), -70.0) / 1000
sys.argv = [a for a in sys.argv if not a.startswith('--offset-ms=')]
RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/indoor_detail_base_overlapfix"
RES, THRESHOLD = 0.5, 0.4
# Το MapClosures δεν προτείνει ποτέ τους 3 πιο πρόσφατους χάρτες (no_of_local_maps_to_skip = 3):
# μετράμε μόνο ζεύγη που μπορεί να προταθούν. Τα κοντινότερα τα συνδέει η odometry.
MIN_GAP = 4


def floor_of(z):
    return "ισόγ." if z < 2.5 else ("σκάλα" if z < 5.0 else "όροφ.")


def bev_overlap(a, b):
    ka = set(map(tuple, np.floor(a[:, :2] / RES).astype(int)))
    kb = set(map(tuple, np.floor(b[:, :2] / RES).astype(int)))
    return len(ka & kb) / min(len(ka), len(kb))


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN)); _, G = interpolate(gs, gT, st + OFFSET)
    g2o = glob.glob(f"{RUN}/*/local_maps/local_map_graph.g2o")[0]
    K, closures = {}, set()
    for line in open(g2o):
        t = line.split()
        if t and t[0].startswith("VERTEX_SE3"):
            v = np.array(list(map(float, t[2:9]))); T = np.eye(4)
            T[:3, :3] = R.from_quat(v[3:]).as_matrix(); T[:3, 3] = v[:3]; K[int(t[1])] = T
        elif t and t[0].startswith("EDGE_SE3") and abs(int(t[1]) - int(t[2])) > 1:
            closures.add(frozenset((int(t[1]), int(t[2]))))
    ids = sorted(K)
    # Αρχή κάθε node: η σάρωση που συμπίπτει με το keypose του, ψάχνοντας μόνο ΜΕΤΑ την αρχή του προηγούμενου —
    # αλλιώς, όταν η διαδρομή επιστρέφει στην αφετηρία, ο node 0 «βρίσκει» μια σάρωση του τέλους.
    start, lo = {}, 0
    for i in sorted(K):
        start[i] = lo + int(np.argmin(np.linalg.norm(E[lo:, :3, 3] - K[i][:3, 3], axis=1)))
        lo = start[i]
    ends = {i: (start[ids[n + 1]] if n + 1 < len(ids) else len(E)) for n, i in enumerate(ids)}
    plys = g2o.replace("local_map_graph.g2o", "plys")
    # Ο τελευταίος node σβήνεται από το pipeline (erase_last_local_map) και δεν έχει PLY.
    ids = [i for i in ids if Path(f"{plys}/{i:06d}.ply").exists()]
    maps = {}
    for i in ids:
        p = np.asarray(o3d.io.read_point_cloud(f"{plys}/{i:06d}.ply").points, dtype=np.float64)
        # PLY = keypose·(τοπικά σημεία) → στις σωστές θέσεις: G(αρχή node)·keypose⁻¹·PLY
        T = G[start[i]] @ np.linalg.inv(K[i])
        maps[i] = p @ T[:3, :3].T + T[:3, 3]

    print(f"run: {RUN} · {len(ids)} τοπικοί χάρτες · closures που βρέθηκαν: "
          + ", ".join(f"{min(c)}↔{max(c)}" for c in sorted(closures, key=min)))
    print("\nnode: σαρώσεις και όροφος (από το ύψος GT στην αρχή/τέλος)")
    print("  " + "  ".join(f"{i}:{start[i]}–{ends[i]}({floor_of(G[start[i],2,3])}→{floor_of(G[ends[i]-1,2,3])})" for i in ids))

    rows = []
    for a in ids:
        for b in ids:
            if b - a >= MIN_GAP:
                o3 = local_maps_overlap(maps[b], maps[a], np.eye(4), RES)
                o2 = bev_overlap(maps[a], maps[b])
                rows.append((a, b, o3, o2))
    print(f"\n{'ζεύγος':>8}{'όροφοι':>15}{'επικάλυψη 3D':>14}{'σε κάτοψη':>11}   {'':<34}")
    shown = 0
    for a, b, o3, o2 in sorted(rows, key=lambda r: -max(r[2], r[3])):
        if max(o3, o2) < 0.25:
            continue
        found = frozenset((a, b)) in closures
        if o3 > THRESHOLD:
            verdict = "✔ βρέθηκε" if found else "✘ ΧΑΘΗΚΕ (έπρεπε να ενωθούν)"
        elif o2 > THRESHOLD and floor_of(G[start[a], 2, 3]) != floor_of(G[start[b], 2, 3]):
            verdict = "⚠ ΚΙΝΔΥΝΟΣ: ίδια κάτοψη, άλλο ύψος" + (" — ΚΑΙ ΕΝΩΘΗΚΑΝ ✘" if found else "")
        else:
            verdict = "λάθος closure ✘" if found else ""
        fl = f"{floor_of(G[start[a],2,3])}/{floor_of(G[start[b],2,3])}"
        print(f"{f'{a}↔{b}':>8}{fl:>15}{o3:>14.2f}{o2:>11.2f}   {verdict}")
        shown += 1
    should = [r for r in rows if r[2] > THRESHOLD]
    risky = [r for r in rows if r[2] <= THRESHOLD and r[3] > THRESHOLD
             and floor_of(G[start[r[0]], 2, 3]) != floor_of(G[start[r[1]], 2, 3])]
    hit = sum(frozenset((a, b)) in closures for a, b, _, _ in should)
    print(f"\nΈπρεπε να ενωθούν (3D > {THRESHOLD}, απόσταση ≥ {MIN_GAP} θέσεις): {len(should)} ζεύγη · βρέθηκαν {hit} · χάθηκαν {len(should) - hit}")
    print(f"Ίδια κάτοψη, άλλος όροφος (κάτοψη > {THRESHOLD}, 3D ≤ {THRESHOLD}): {len(risky)} ζεύγη")
    wrong = [c for c in closures if not any(frozenset((a, b)) == c for a, b, *_ in should)]
    print(f"Closures χωρίς πραγματική επικάλυψη (λάθος): {len(wrong)}" + (" — " + ", ".join(f"{min(c)}↔{max(c)}" for c in wrong) if wrong else ""))


if __name__ == "__main__":
    main()
