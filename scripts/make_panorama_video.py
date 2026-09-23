#!/usr/bin/env python3
"""Βίντεο των πανοραμάτων intensity, με τις αντιστοιχίσεις SIFT σημειωμένες (#034).

Σε κάθε καρέ σχεδιάζονται τα χαρακτηριστικά που αντιστοιχίστηκαν με την ΠΡΟΗΓΟΥΜΕΝΗ σάρωση:
  πράσινο = κανονική αντιστοίχιση (το σημείο μετακινήθηκε ως προς τον σαρωτή)
  κόκκινο = «κολλημένη» (|p − q| < 5 cm: εμφανίζεται ακίνητη ως προς τον σαρωτή)
Ανοίγοντας το βίντεο καρέ-καρέ φαίνεται τι κινείται μαζί με τη σκηνή και τι μένει στην ίδια θέση της εικόνας.

    python scripts/make_panorama_video.py [πρώτη=400] [πλήθος=100] [--render=splat|raycast] [--fps=5]
                                          [--zoom=R0,C0,NR,NC] [--out=αρχείο.mp4] [--codec=avc1] [--png]
    --codec=avc1 (H.264, παίζει σε QuickTime· mp4v = MPEG-4 Part 2, συχνά δεν ανοίγει)· --png: και σειρά εικόνων
"""
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
from kiss_icp.datasets import dataset_factory

sys.path.insert(0, str(Path(__file__).parent))
import kiss_slam.intensity_deskew as I  # noqa: E402
from test_i3_scale_bias import read_raw  # noqa: E402

warnings.simplefilter("ignore")
arg = lambda k, d: next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(f"--{k}=")), d)
pos = [a for a in sys.argv[1:] if not a.startswith("--")]
FIRST = int(pos[0]) if pos else 400
N = int(pos[1]) if len(pos) > 1 else 100
I.RENDER = arg("render", "splat")
FPS = float(arg("fps", 5))
ZOOM = arg("zoom", None)
STUCK = 0.05
CODEC = arg("codec", "avc1")
PNG = "--png" in sys.argv
NOMATCH = "--no-matches" in sys.argv      # χωρίς SIFT: πολύ πιο γρήγορο, καθαρή εικόνα
RANGE = "--with-range" in sys.argv        # και η εικόνα ΑΠΟΣΤΑΣΗΣ από κάτω
ZUP = int(arg("zup", 4))                  # μεγέθυνση της περιοχής
MASKF = arg("mask", None)                 # npz με μάσκα «καρφωμένων» pixel → αχνό κόκκινο υπόβαθρο
# Χρώματα αντιστοιχίσεων: πράσινο = κανονική· ΚΟΚΚΙΝΟ = «κολλημένη» ΣΤΟ ΚΟΝΤΙΝΟ ΔΑΠΕΔΟ (αυτές που πετά το φίλτρο)·
# κίτρινο = κολλημένη αλλού (τις κρατάμε σήμερα).


def main():
    zoom = tuple(int(v) for v in ZOOM.split(",")) if ZOOM else None
    out = Path(arg("out", f"runs/videos/panorama_{FIRST}_{FIRST+N}_{I.RENDER}{'_zoom' if zoom else ''}.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = dataset_factory(dataloader="rosbag", data_dir=Path("data/church_02_cut.bag"),
                         sequence=None, topic="/hesai/pandar", meta=None)
    ds.read_point_cloud = read_raw
    sift, bf = cv2.SIFT_create(), cv2.BFMatcher(cv2.NORM_L2)
    mask = None
    if MASKF:
        z = np.load(MASKF); mask = z["mask"] if "mask" in z else ((z["seen"] & (z["t"] > 20)))
    writer, prev = None, None
    for k in range(FIRST + N):
        scan = ds[k]
        if k < FIRST - 1:
            continue
        ok = ~np.isnan(scan[0]).any(axis=1) & (np.linalg.norm(scan[0], axis=1) > I.MIN_RANGE)
        big, P, T, valid = I.panorama(*(a[ok] for a in scan))
        kps, desc = (None, None) if NOMATCH else sift.detectAndCompute(big, None)
        cur = (big, P, T, valid, kps, desc)
        if prev is not None:
            frame = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
            if mask is not None:
                big_mask = np.repeat(mask, I.UP, axis=0)
                frame[big_mask] = (0.5 * frame[big_mask] + np.array([0, 0, 128])).astype(np.uint8)
            if RANGE:
                rng = np.linalg.norm(P, axis=2); rng[~valid] = np.nan
                rgb = cv2.applyColorMap(np.clip(np.nan_to_num(rng, nan=0) / 12 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
                rgb[~valid] = (255, 255, 255)
                rgb = cv2.resize(rgb, (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST)
            P1, T1, v1, kp1, d1 = prev[1], prev[2], prev[3], prev[4], prev[5]
            n_stuck = n_ok = 0
            for m, nn in ([] if NOMATCH else bf.knnMatch(d1, desc, k=2)):
                if m.distance >= I.RATIO * nn.distance:
                    continue
                x = I.lookup(P1, T1, v1, kp1[m.queryIdx]); y = I.lookup(P, T, valid, kps[m.trainIdx])
                if x is None or y is None:
                    continue
                q = y[0]
                elev = np.degrees(np.arctan2(q[2], np.linalg.norm(q[:2])))
                stuck = np.linalg.norm(x[0] - y[0]) < STUCK
                floor = stuck and elev < -10 and np.linalg.norm(q) < 5.0     # ό,τι πετά το φίλτρο δαπέδου
                n_stuck += floor; n_ok += not floor
                u, v = kps[m.trainIdx].pt
                col = (0, 0, 255) if floor else ((0, 215, 255) if stuck else (0, 220, 0))
                cv2.circle(frame, (int(u), int(v)), 9 if floor else 7, col, 3 if floor else 2)
            if zoom:
                r0, c0, nr, nc = zoom
                sl = (slice(r0 * I.UP, (r0 + nr) * I.UP), slice(c0, c0 + nc))
                frame = frame[sl]
                if RANGE:
                    rgb = rgb[sl]
                frame = cv2.resize(frame, (frame.shape[1] * ZUP, frame.shape[0] * ZUP), interpolation=cv2.INTER_NEAREST)
                if RANGE:
                    rgb = cv2.resize(rgb, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            if RANGE:
                frame = np.vstack([frame, np.full((6, frame.shape[1], 3), 60, np.uint8), rgb])
            cv2.putText(frame, f"scan {k}" + ("" if NOMATCH else f"   ok {n_ok}   FLOOR-STUCK {n_stuck}")
                        + ("   [intensity / range 0-12 m]" if RANGE else ""), (10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*CODEC), FPS, (w, h))
                if not writer.isOpened():
                    raise SystemExit(f"ο codec {CODEC} δεν είναι διαθέσιμος")
            writer.write(frame)
            if PNG:
                d = out.with_suffix(""); d.mkdir(exist_ok=True); cv2.imwrite(str(d / f"{k:05d}.png"), frame)
        prev = cur
    writer.release()
    print(f"→ {out}  ({out.stat().st_size/1e6:.1f} MB, codec {CODEC})"
          + (f" · εικόνες: {out.with_suffix('')}/" if PNG else ""))


if __name__ == "__main__":
    main()
