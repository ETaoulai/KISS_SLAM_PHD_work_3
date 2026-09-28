#!/usr/bin/env python3
"""The scans where the intensity-image motion fails, as images for inspection (request M.T. 28/9).

    python scripts/dump_failed_matches.py <sequence> <out dir> [--topic=/os_cloud_node/points] [--intensity-scale=0.249]
                                          [--seed=0] [--detector=surf] [--config=<yaml>] [--n=<scans>]

<sequence> as for scripts/run_ncd.py (a bag, a folder of split bags, or a Newer College 2020 dir).  Only the image-motion
estimator runs (no ICP): it reads the scans through the same SlamPipeline reader as run_ncd.py and uses the same settings
(image_deskew defaults: SURF, model "car", sub-pixel, near-floor filter; same seed), so it fails on exactly the scans the
runs fail on (the motion depends only on the raw scans and the RANSAC seed).  For every failed scan k, in <out dir>:
  scan<k>_a_previous.png   intensity panorama of scan k-1 (8x up-sampled vertically, as the detector sees it)
  scan<k>_b_current.png    intensity panorama of scan k
  scan<k>_c_matches.png    the matches that passed the ratio test (at most 300, best first)
  rejected.csv             scan, time, reason, matches after the checks, ratio-test matches, keypoints of both scans
The same images during a SLAM run: run_ncd.py --save-failed.  Outputs on ext4 only (#046).
"""
import os
import sys
from pathlib import Path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
    seq, out = Path(args[0]), Path(args[1])
    os.environ["KISS_SLAM_OUT_DIR"] = str(out / "slam_output_unused")
    import kiss_slam.pipeline as pipeline

    is_bag = seq.suffix == ".bag" or (seq.is_dir() and any(seq.glob("*.bag")))
    if is_bag:
        from kiss_icp.datasets.rosbag import RosbagDataset
        dataset = RosbagDataset(seq, opts.get("topic", "/os_cloud_node/points"))
    else:
        from kiss_slam.tools.ncd_pcd import NewerCollege2020Pcd
        dataset = NewerCollege2020Pcd(seq)
    load_config = pipeline.load_config

    def load_with_overrides(path):
        config = load_config(path)
        config.image_deskew.seed = int(opts.get("seed", 0))
        config.image_deskew.intensity_scale = float(opts.get("intensity-scale", 255.0 / 1024.0))
        config.image_deskew.save_rejected_dir = str(out)
        config.image_deskew.parallel = False
        return config

    pipeline.load_config = load_with_overrides
    p = pipeline.SlamPipeline(dataset=dataset, config_file=Path(opts["config"]) if "config" in opts else None,
                              n_scans=int(opts.get("n", -1)), image_deskew=True, image_detector=opts.get("detector", "surf"))
    est = p.kiss_slam._image_motion_est
    from tqdm import trange
    failed = 0
    for idx in trange(p._first, p._last, unit=" scans", dynamic_ncols=True):
        M, _ = est.motion(*p._next(idx))
        failed += M is None and idx > p._first
    print(f"{failed} of {p._last - p._first - 1} scan pairs failed -> {out}")


if __name__ == "__main__":
    main()
