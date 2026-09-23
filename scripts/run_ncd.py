#!/usr/bin/env python3
"""One arm of KISS-SLAM on a Newer College 2020 sequence (.pcd scans, kiss_slam/tools/ncd_pcd.py).

    python scripts/run_ncd.py <arm> <sequence dir> <out dir> [n_scans] [--config=<yaml>] [--seed=N]

arm:  kiss  upstream KISS-SLAM (image_deskew off)
      sift  image-motion deskew (i3), SIFT features on the intensity panorama
      surf  image-motion deskew (i3), SURF features (OpenCV with OPENCV_ENABLE_NONFREE)
All arms read the same scans with the same config (default: the KISS-SLAM defaults, the setting of
the KISS-SLAM paper); only image_deskew.enabled / detector differ.  --seed: RANSAC seed of the
image-motion estimator, for measuring run-to-run spread (#037).
"""
import os
import sys
from pathlib import Path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
arm, seq, out = args[0], Path(args[1]), Path(args[2])
n_scans = int(args[3]) if len(args) > 3 else -1
os.environ["KISS_SLAM_OUT_DIR"] = str(out)          # read when the config is built

from kiss_slam.pipeline import SlamPipeline          # noqa: E402
from kiss_slam.tools.ncd_pcd import NewerCollege2020Pcd  # noqa: E402

if arm not in ("kiss", "sift", "surf"):
    sys.exit(f"arm must be kiss, sift or surf, not {arm!r}")
seed = int(opts.get("seed", 0))
if seed:
    import kiss_slam.intensity_deskew as I
    _init = I.ScanMotionEstimator.__init__
    I.ScanMotionEstimator.__init__ = lambda self, *a, **k: _init(self, *a, **{**k, "seed": seed})

SlamPipeline(
    dataset=NewerCollege2020Pcd(seq),
    config_file=Path(opts["config"]) if "config" in opts else None,
    n_scans=n_scans,
    image_deskew=arm != "kiss",
    image_detector=None if arm == "kiss" else arm,
).run().print()
