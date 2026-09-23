#!/usr/bin/env python3
"""Ι-3 (#019): κίνηση κάθε σάρωσης από την εικόνα intensity, για όλο το bag, αιτιακά.

Η σάρωση k παίρνει την κίνηση που εκτιμάται από το ζεύγος (k−1, k) — μόνο παρελθόν και παρόν, όπως θα γινόταν
online. Δεν χρησιμοποιεί GT· το GT διαβάζεται μόνο για την αναφορά σφάλματος στο τέλος.
Αποθηκεύει runs/i3_motion.npz: motion (N×4×4, NaN όπου απέτυχε), inliers (N).

    python scripts/precompute_i3_motion.py [--model=cv|ca|car] [--subpixel] [--bag=data/x.bag --gt=gt/x.txt --run=runs/<baseline>]
    --bag/--gt/--run: άλλη ακολουθία (#028)· --run = ένα run της ίδιας ακολουθίας, μόνο για τις χρονοσφραγίδες της αναφοράς.
    Με --bag το αρχείο εξόδου παίρνει πρόθεμα το όνομα του bag.
    cv: σταθερή ταχύτητα στις δύο σαρώσεις (#019) → runs/i3_motion.npz
    ca: σταθερή επιτάχυνση — η ταχύτητα αλλάζει γραμμικά (#020) → runs/i3_motion_ca.npz
    car: επιτάχυνση μόνο στη στροφή (#021) · --subpixel: παρεμβολή σημείου/χρόνου μέσα στο pixel (#021, κατάληξη _sp)
    --ransac=M --fit=M: κατώφλια απόρριψης (default 0.30 / 0.10 m)· κατάληξη _r<M>_f<M> (#023)
    --stuck=M: απόρριψη αντιστοιχίσεων με |p − q| < M (το «ίδιο σημείο» στο πλαίσιο του αισθητήρα)· κατάληξη _st<M> (#026)
    --floor-only: το --stuck μόνο στο κοντινό δάπεδο (< −10°, < 5 m)· κατάληξη _floor (#027)
    --render=raycast: η εικόνα με ακτίνα ανά pixel αντί για «κάθε σημείο στο πλησιέστερο pixel»· κατάληξη _ray (#034)
    --mask=<npz> --mask-key=narrow|edges|union: εξαίρεση χαρακτηριστικών σε pixel καρφωμένα στον σαρωτή (#036)
    --seed=N: σπόρος του RANSAC· άλλος σπόρος = ανεξάρτητο run, για μέτρηση διασποράς· κατάληξη _seed<N> (#037)
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from kiss_icp.datasets import dataset_factory
from tqdm import trange

sys.path.insert(0, str(Path(__file__).parent))
from evaluate_gt import base_to_lidar, find_tum, interpolate, load_tum  # noqa: E402

from kiss_slam.intensity_deskew import ScanMotionEstimator  # noqa: E402
from kiss_slam.tools.point_cloud2 import read_points  # noqa: E402

warnings.simplefilter("ignore")
MODEL = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "cv")
SUBPIXEL = "--subpixel" in sys.argv
_arg = lambda key: next((float(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith(f"--{key}=")), None)
RANSAC, FIT, STUCK = _arg("ransac"), _arg("fit"), _arg("stuck")
FLOOR = "--floor-only" in sys.argv
RENDER = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--render=")), "splat")
_s = lambda key, default: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{key}=")), default)
MASK, MASK_KEY = _s("mask", None), _s("mask-key", "mask")     # npz με μάσκα pixel καρφωμένων στον σαρωτή (#035)
TRANS = _arg("trans-min")                                     # ελάχιστη απόσταση αντιστοίχισης για τη ΜΕΤΑΤΟΠΙΣΗ (#039)
SEED = int(_s("seed", "0"))                                   # σπόρος του RANSAC· άλλος σπόρος = άλλο run για μέτρηση διασποράς (#037)
BAG, GT, STAMPS = _s("bag", "data/church_02_cut.bag"), _s("gt", "gt/church_02_gt-tum.txt"), _s("run", "runs/indoor_detail_base_overlapfix")
PREFIX = "" if BAG == "data/church_02_cut.bag" else Path(BAG).stem + "_"
OUT = Path(f"runs/{PREFIX}i3_motion" + ("" if MODEL == "cv" else f"_{MODEL}") + ("_sp" if SUBPIXEL else "")
           + (f"_r{RANSAC:g}_f{FIT:g}" if RANSAC else "") + (f"_st{STUCK:g}" if STUCK else "") + ("_floor" if FLOOR else "") + ("_ray" if RENDER == "raycast" else "") + (f"_mask-{MASK_KEY}" if MASK else "") + (f"_tr{TRANS:g}" if TRANS else "") + ("_mag" if TRANS and _s("trans-mode","vector")=="magnitude" else "") + (f"_f{_s('trans-factor','1.0')}" if _s("trans-factor","1.0")!="1.0" else "") + (f"_seed{SEED}" if SEED else "") + ".npz")


def read_raw(msg):
    s = read_points(msg, field_names=["x", "y", "z", "intensity", "timestamp", "ring"])
    xyz = np.column_stack([s["x"], s["y"], s["z"]]).astype(np.float64)
    return xyz, s["timestamp"].astype(np.float64), s["intensity"].astype(np.float64), s["ring"].astype(np.int64)


def main():
    ds = dataset_factory(dataloader="rosbag", data_dir=Path(BAG),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    import kiss_slam.intensity_deskew as I
    if RANSAC:
        I.RANSAC_THR, I.FIT_THR = RANSAC, FIT
    I.STUCK_MIN, I.STUCK_FLOOR_ONLY = STUCK, FLOOR
    I.RENDER = RENDER
    I.TRANS_MIN_RANGE = TRANS
    I.TRANS_MODE = _s("trans-mode", "vector")
    I.TRANS_FACTOR = float(_s("trans-factor", "1.0"))
    if MASK:
        _z = np.load(MASK); I.PIXEL_MASK = _z[MASK_KEY].astype(bool)
        print(f"μάσκα pixel «{MASK_KEY}» από {MASK}: {int(I.PIXEL_MASK.sum())} pixel ({100*I.PIXEL_MASK.mean():.2f} %)")
    est = ScanMotionEstimator(model=MODEL, subpixel=SUBPIXEL, seed=SEED)
    N = len(ds)
    motion = np.full((N, 4, 4), np.nan); inliers = np.zeros(N, int)
    params = np.full((N, 12), np.nan); t_start = np.full(N, np.nan)      # η καμπύλη κίνησης, για deskew_curve (#033)
    t = time.time()
    for k in trange(N, unit=" scans"):
        M, n = est.motion(*ds[k])
        if M is not None:
            motion[k] = M; inliers[k] = n
            x = est.last_params; params[k, :len(x)] = x
        t_start[k] = est.last_t_start
    ms = 1000 * (time.time() - t) / N
    np.savez(OUT, motion=motion, inliers=inliers, params=params, t_start=t_start)
    ok = ~np.isnan(motion[:, 0, 0])
    if I.TRANS_MODE == "auto":
        print(f"τρέχων συντελεστής μετατόπισης: τελικός {est.last_factor:.3f} · "
              f"διάμεσος λόγος {np.median(est.ratios):.3f} από {len(est.ratios)} σαρώσεις")
    print(f"→ {OUT}: επιτυχία {ok[1:].mean()*100:.1f} % ({(~ok[1:]).sum()} αποτυχίες), inliers διάμεσος "
          f"{np.median(inliers[ok]):.0f}, ~{ms:.0f} ms/σάρωση (μαζί με την ανάγνωση)")

    # Αναφορά σφάλματος έναντι της αληθινής κίνησης (GT, τ = 0) — μόνο για έλεγχο.
    gs, gT = load_tum(GT); gT = base_to_lidar(gT)
    st, _ = load_tum(find_tum(STAMPS))
    cov, G = interpolate(gs, gT, st)
    N = min(N, len(st), len(G))                                     # η τροχιά αναφοράς μπορεί να έχει λιγότερες πόζες
    ok = ok[:N] & np.concatenate([[False], cov[1:N] & cov[:N - 1]])  # μόνο σαρώσεις με GT και στις δύο άκρες
    inv = np.linalg.inv
    rot = lambda D: np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
    e = np.array([rot(inv(inv(G[k - 1]) @ G[k]) @ motion[k]) for k in range(1, N) if ok[k]])
    z = np.array([rot(inv(G[k - 1]) @ G[k]) for k in range(1, N)])
    et = np.array([1000 * np.linalg.norm((inv(inv(G[k - 1]) @ G[k]) @ motion[k])[:3, 3]) for k in range(1, N) if ok[k]])
    tg = np.array([(inv(G[k - 1]) @ G[k])[:3, 3] for k in range(1, N) if ok[k]])
    ti = np.array([motion[k][:3, 3] for k in range(1, N) if ok[k]])
    scale = np.median(np.sum(ti * tg, 1) / np.maximum(np.sum(tg * tg, 1), 1e-9))
    print(f"[{MODEL}{' +subpixel' if SUBPIXEL else ''}{f' ransac {RANSAC} fit {FIT}' if RANSAC else ''}{f' stuck {STUCK}' if STUCK else ''}{' floor-only' if FLOOR else ''}{' raycast' if RENDER == 'raycast' else ''}] κλίμακα μετατόπισης {scale:.3f} · σφάλμα έναντι GT: στροφή διάμεσος {np.median(e):.2f}°, 90ό εκατ. {np.percentile(e, 90):.2f}° · "
          f"θέση διάμεσος {np.median(et):.0f} mm · "
          f"(μέγεθος της κίνησης: διάμεσος {np.median(z):.2f}°)")


if __name__ == "__main__":
    main()
