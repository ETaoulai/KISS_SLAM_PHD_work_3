#!/usr/bin/env python3
"""One arm of KISS-SLAM on a Newer College 2020 sequence (.pcd scans, kiss_slam/tools/ncd_pcd.py).

    python scripts/run_ncd.py <arm> <sequence dir> <out dir> [n_scans] [--config=<yaml>] [--seed=N] [--parallel]

arm:  kiss  upstream KISS-SLAM (image_deskew off)
      sift  image-motion deskew (i3), SIFT features on the intensity panorama
      surf  image-motion deskew (i3), SURF features (OpenCV with OPENCV_ENABLE_NONFREE)
All arms read the same scans with the same config (default: the KISS-SLAM defaults, the setting of
the KISS-SLAM paper); only image_deskew.enabled / detector differ.  --seed: RANSAC seed of the
image-motion estimator, for measuring run-to-run spread (#037).  --parallel: the image motion in a
worker process, overlapping the ICP (image_deskew.parallel); same trajectory, less time per scan.
"""
import os
import sys
from pathlib import Path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    arm, seq, out = args[0], Path(args[1]), Path(args[2])
    n_scans = int(args[3]) if len(args) > 3 else -1
    if arm not in ("kiss", "sift", "surf"):
        sys.exit(f"arm must be kiss, sift or surf, not {arm!r}")
    os.environ["KISS_SLAM_OUT_DIR"] = str(out)          # read when the config is built

    import kiss_slam.pipeline as pipeline
    from kiss_slam.tools.ncd_pcd import NewerCollege2020Pcd

    # --seed / --parallel as config fields, so they also reach the image-motion worker process.
    load_config = pipeline.load_config

    def load_with_overrides(path):
        config = load_config(path)
        config.image_deskew.seed = int(opts.get("seed", 0))
        config.image_deskew.parallel = "--parallel" in sys.argv
        return config

    pipeline.load_config = load_with_overrides
    pipeline.SlamPipeline(
        dataset=NewerCollege2020Pcd(seq),
        config_file=Path(opts["config"]) if "config" in opts else None,
        n_scans=n_scans,
        image_deskew=arm != "kiss",
        image_detector=None if arm == "kiss" else arm,
    ).run().print()


if __name__ == "__main__":      # required: image_deskew.parallel starts its worker with "spawn"
    main()
