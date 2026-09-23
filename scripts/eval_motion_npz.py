#!/usr/bin/env python3
"""Σφάλμα κίνησης ενός αποθηκευμένου npz έναντι GT, χωρίς να ξανατρέξει ο εκτιμητής (#039).

    python scripts/eval_motion_npz.py runs/<x>.npz gt/<seq>_gt-tum.txt runs/<ένα run της ακολουθίας>
"""
import sys, warnings; warnings.simplefilter("ignore")
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

NPZ, GT, RUN = sys.argv[1], sys.argv[2], sys.argv[3]
M = np.load(NPZ)["motion"]
gs, gT = load_tum(GT); gT = base_to_lidar(gT)
st, _ = load_tum(find_tum(RUN)); cov, G = interpolate(gs, gT, st)
N = min(len(M), len(st), len(G))
ok = (~np.isnan(M[:, 0, 0]))[:N] & np.concatenate([[False], cov[1:N] & cov[:N - 1]])
inv = np.linalg.inv
rot = lambda D: np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
k = np.where(ok)[0]
e = np.array([rot(inv(inv(G[i - 1]) @ G[i]) @ M[i]) for i in k])
et = np.array([1000 * np.linalg.norm((inv(inv(G[i - 1]) @ G[i]) @ M[i])[:3, 3]) for i in k])
tg = np.array([(inv(G[i - 1]) @ G[i])[:3, 3] for i in k]); ti = np.array([M[i][:3, 3] for i in k])
s = np.median(np.sum(ti * tg, 1) / np.maximum(np.sum(tg * tg, 1), 1e-9))
print(f"{Path(NPZ).name}: κλίμακα {s:.3f} · στροφή διάμεσος {np.median(e):.2f}°, 90ό {np.percentile(e,90):.2f}° · "
      f"θέση διάμεσος {np.median(et):.0f} mm · {len(k)} σαρώσεις")
