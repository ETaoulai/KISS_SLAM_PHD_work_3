#!/usr/bin/env python3
"""Πού κόπηκαν οι τοπικοί χάρτες και με ποια κατακόρυφο; (#014)

Για κάθε node ενός run: σαρώσεις, οριζόντια απόσταση, μεταβολή ύψους κατά το GT, κλίση του αισθητήρα
στην αρχή του node, και — αν το run είχε `splitting_height` — η γωνία ανάμεσα στην κατακόρυφο που
εκτίμησε το SLAM (έδαφος του MapClosures, γραμμή «node N up = …» του log) και στην αληθινή (GT).

    python scripts/analyze_node_splits.py <run_dir> [<run.log>] [--offset-ms=-70]
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

OFFSET = next((float(a.split('=', 1)[1]) for a in sys.argv if a.startswith('--offset-ms=')), -70.0) / 1000
sys.argv = [a for a in sys.argv if not a.startswith('--offset-ms=')]
RUN = sys.argv[1]
LOG = sys.argv[2] if len(sys.argv) > 2 else f"{RUN}.log"


def main():
    gs, gT = load_tum("gt/church_02_gt-tum.txt"); gT = base_to_lidar(gT)
    st, E = load_tum(find_tum(RUN)); _, G = interpolate(gs, gT, st + OFFSET)
    K = {}
    for line in open(glob.glob(f"{RUN}/*/local_maps/local_map_graph.g2o")[0]):
        t = line.split()
        if t and t[0].startswith("VERTEX_SE3"):
            v = np.array(list(map(float, t[2:9]))); T = np.eye(4)
            T[:3, :3] = R.from_quat(v[3:]).as_matrix(); T[:3, 3] = v[:3]; K[int(t[1])] = T
    ups = {}
    if Path(LOG).exists():
        for m in re.finditer(r"node (\d+) up = \[([^\]]+)\]", open(LOG, errors="ignore").read()):
            ups[int(m.group(1))] = np.array(list(map(float, m.group(2).split(","))))
    ids = sorted(K)
    start = [int(np.argmin(np.linalg.norm(E[:, :3, 3] - K[i][:3, 3], axis=1))) for i in ids]
    start.append(len(E))
    ez = np.array([0, 0, 1.0])
    print(f"{RUN}: {len(ids)} nodes")
    print(" node  σαρώσεις    xy (m)  Δz GT (m)  κλίση (°)  σφάλμα κατακορύφου (°)")
    errs = []
    for n, i in enumerate(ids):
        a, b = start[n], start[n + 1] - 1
        if b <= a:
            continue
        rel = np.linalg.inv(E[a]) @ E[b]
        up_gt = G[a][:3, :3].T @ ez                      # κατακόρυφος στο πλαίσιο αισθητήρα
        tilt = np.degrees(np.arccos(abs(up_gt[2])))
        err = ""
        if i in ups:
            e = np.degrees(np.arccos(np.clip(ups[i] @ up_gt, -1, 1))); errs.append(e); err = f"{e:6.1f}"
        print(f" {i:3d}   {a:4d}–{b:4d}   {np.linalg.norm(rel[:2, 3]):6.1f}   {G[b][2, 3] - G[a][2, 3]:+7.2f}   {tilt:7.1f}   {err}")
    if errs:
        print(f"σφάλμα εκτιμώμενης κατακορύφου: διάμεσος {np.median(errs):.1f}°, max {np.max(errs):.1f}°")


if __name__ == "__main__":
    main()
