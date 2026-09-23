"""Κάθε σημείο πρέπει να παίρνει το ΔΙΚΟ του intensity.

Τρέχει πάνω σε πραγματικά scans του church_02_cut.bag:
    python tests/test_intensity_alignment.py

(A) deskew OFF: το preprocessor δεν μετακινεί σημεία, οπότε για κάθε σημείο του
    source βρίσκεται το ΑΚΡΙΒΕΣ αρχικό του σημείο → γνωστό σωστό intensity.
(B) deskew ON με ρεαλιστική κίνηση: η αλήθεια βγαίνει από το ίδιο deskew χωρίς
    όριο απόστασης (γραμμή i = αρχικό σημείο i). Εδώ η μάσκα με βάση την
    απόσταση του ΑΡΧΙΚΟΥ σημείου δεν αρκεί, γιατί το deskew μετακινεί σημεία
    πάνω από το όριο των 50 m.
"""
import sys, warnings
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
warnings.simplefilter("ignore")

from kiss_icp.datasets import dataset_factory
from kiss_icp.preprocess import Preprocessor
from kiss_icp.voxelization import voxel_down_sample
from kiss_slam.tools.point_cloud2 import read_point_cloud
from kiss_slam.slam import _interpolate_intensity, _align_intensity_to_preprocessed

MAX_R, MIN_R, VOXEL = 50.0, 0.0, 0.25
SCANS = {0, 150, 300, 450, 600, 750, 900, 1500}

ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                     sequence=None, topic="/hesai/pandar", meta=None)
ds.read_point_cloud = read_point_cloud
pre_off = Preprocessor(MAX_R, MIN_R, False, 0)
pre_on  = Preprocessor(MAX_R, MIN_R, True, 0)
motion = np.eye(4); motion[:3, :3] = R.from_euler("z", 2.5, degrees=True).as_matrix(); motion[:3, 3] = [0.18, 0.02, 0.01]

def source_of(frame):
    return voxel_down_sample(voxel_down_sample(frame, VOXEL * 0.5), VOXEL * 1.5)

def check(pre, motion, xyz, ts, inten):
    """Επιστρέφει (%σωστά ΠΑΛΙΟ, %σωστά ΝΕΟ, fallback;) με ακριβή αλήθεια."""
    des = pre.preprocess(xyz, ts, motion)
    des_all = unbounded(pre).preprocess(xyz, ts, motion)      # όπως στο slam.py
    src = source_of(des)
    d, row = KDTree(des_all).query(src, k=1)
    assert d.max() < 1e-9, "κάθε σημείο του source πρέπει να είναι ακριβώς μια γραμμή του des_all"
    truth = inten[row]                                        # row i του des_all = raw σημείο i
    old = _interpolate_intensity(des, inten, src)
    aligned, fb = _align_intensity_to_preprocessed(des_all, inten, des, MAX_R, MIN_R)
    new = _interpolate_intensity(des, aligned, src)
    return (np.mean(np.isclose(old, truth)) * 100, np.mean(np.isclose(new, truth)) * 100, fb)

_unb = {}
def unbounded(pre):
    key = id(pre)
    if key not in _unb:
        _unb[key] = Preprocessor(1e9, 0.0, pre is pre_on, 0)
    return _unb[key]

ok = True
print(f"{'scan':>5}{'>50m':>6}{'| deskew OFF: ΠΑΛΙΟ':>21}{'ΝΕΟ':>8}{'| deskew ON: ΠΑΛΙΟ':>20}{'ΝΕΟ':>8}{'fallback':>10}")
for idx in range(max(SCANS) + 1):
    xyz, ts, inten = ds[idx]
    if idx not in SCANS:
        continue
    o1, n1, f1 = check(pre_off, np.eye(4), xyz, ts, inten)
    o2, n2, f2 = check(pre_on, motion, xyz, ts, inten)
    n_far = int((np.linalg.norm(xyz, axis=1) > MAX_R).sum())
    print(f"{idx:>5}{n_far:>6}{o1:>19.1f}%{n1:>7.1f}%{o2:>18.1f}%{n2:>7.1f}%{('ΝΑΙ ⚠' if (f1 or f2) else 'όχι'):>10}")
    ok &= n1 == 100.0 and n2 == 100.0 and not f1 and not f2

print("\nRESULT:", "PASS — κάθε σημείο παίρνει το δικό του intensity, με και χωρίς deskew, χωρίς fallback"
      if ok else "FAIL")
sys.exit(0 if ok else 1)
