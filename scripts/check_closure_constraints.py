#!/usr/bin/env python3
"""Είναι σωστό κάθε closure; Σύγκριση του περιορισμού του με τη σχετική θέση/στροφή του GT (#014).

Η επικάλυψη 3D < 0.4 στο GT δεν σημαίνει λάθος ένωση — μόνο μικρή κοινή περιοχή. Το ουσιαστικό
κριτήριο: ο περιορισμός που μπαίνει στο pose graph (ακμή g2o i→j) πρέπει να συμφωνεί με το
inv(G_i)·G_j του GT, όπου G_i η θέση του αισθητήρα στην αρχή του node i. Για αναφορά τυπώνονται
και οι ακμές odometry (διαδοχικοί nodes), που έχουν μόνο το σφάλμα της odometry σε ~15 m.

    python scripts/check_closure_constraints.py <run_dir> [--offset-ms=-70] [--gt=gt/<seq>_gt-tum.txt]
"""
import glob
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

OFFSET = next((float(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--offset-ms=')), -70.0) / 1000
GT = next((a.split('=', 1)[1] for a in sys.argv if a.startswith('--gt=')), 'gt/church_02_gt-tum.txt')
sys.argv = [a for a in sys.argv if not a.startswith('--offset-ms=') and not a.startswith('--gt=')]
RUN = sys.argv[1]


def se3(v):
    T = np.eye(4); T[:3, :3] = R.from_quat(v[3:7]).as_matrix(); T[:3, 3] = v[:3]; return T


def main():
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN)); _, G = interpolate(gs, gT, st + OFFSET)
    K, edges = {}, []
    for line in open(glob.glob(f"{RUN}/*/local_maps/local_map_graph.g2o")[0]):
        t = line.split()
        if t and t[0].startswith("VERTEX_SE3"):
            K[int(t[1])] = se3(np.array(list(map(float, t[2:9]))))
        elif t and t[0].startswith("EDGE_SE3"):
            edges.append((int(t[1]), int(t[2]), se3(np.array(list(map(float, t[3:10]))))))
    # Αρχή κάθε node: η σάρωση που συμπίπτει με το keypose του, ψάχνοντας μόνο ΜΕΤΑ την αρχή του προηγούμενου —
    # αλλιώς, όταν η διαδρομή επιστρέφει στην αφετηρία, ο node 0 «βρίσκει» μια σάρωση του τέλους.
    start, lo = {}, 0
    for i in sorted(K):
        start[i] = lo + int(np.argmin(np.linalg.norm(E[lo:, :3, 3] - K[i][:3, 3], axis=1)))
        lo = start[i]
    inv = np.linalg.inv
    ang = lambda M: np.degrees(np.arccos(np.clip((np.trace(M[:3, :3]) - 1) / 2, -1, 1)))
    rows = {"odometry": [], "closure": []}
    for i, j, Z in edges:
        gt = inv(G[start[i]]) @ G[start[j]]
        err = inv(gt) @ Z
        rows["closure" if abs(i - j) > 1 else "odometry"].append((i, j, np.linalg.norm(err[:3, 3]), ang(err), err[2, 3]))
    print(RUN)
    for kind, r in rows.items():
        t = np.array([x[2] for x in r]); a = np.array([x[3] for x in r])
        print(f"  {kind:9s} ({len(r):2d}): σφάλμα θέσης διάμεσος {np.median(t):.2f} m, max {t.max():.2f} m · "
              f"στροφής διάμεσος {np.median(a):.1f}°, max {a.max():.1f}°")
    for i, j, t, a, dz in sorted(rows["closure"]):
        print(f"    {i:2d}↔{j:2d}   θέση {t:5.2f} m (z {dz:+5.2f})   στροφή {a:4.1f}°")


if __name__ == "__main__":
    main()
