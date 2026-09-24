#!/usr/bin/env python3
"""One arm of KISS-SLAM on a Newer College sequence (Ouster).

    python scripts/run_ncd.py <arm> <sequence> <out dir> [n_scans] [--config=<yaml>] [--seed=N] [--parallel]
                              [--topic=/os_cloud_node/points] [--intensity-scale=0.249] [--diag]
                              [--parts=full|translation|rotation] [--rot-smooth=k]

sequence: a 2020 sequence dir with raw_format/ouster_scan/*.pcd (kiss_slam/tools/ncd_pcd.py), or a
          .bag file, or a folder whose *.bag are ONE split sequence (read in time order; 2021 bags).
arm:  kiss  upstream KISS-SLAM (image_deskew off)
      sift  image-motion deskew (i3), SIFT features on the intensity panorama
      surf  image-motion deskew (i3), SURF features (OpenCV with OPENCV_ENABLE_NONFREE)
All arms read the same scans with the same config (default: the KISS-SLAM defaults, the setting of
the KISS-SLAM paper); only image_deskew.enabled / detector differ.  --seed: RANSAC seed of the
image-motion estimator, for measuring run-to-run spread (#037).  --parallel: the image motion in a
worker process, overlapping the ICP (image_deskew.parallel); same trajectory, less time per scan.
--intensity-scale: image_deskew.intensity_scale, default 255/1024 for the Ouster signal (#041).
--parts / --rot-smooth: ablation of the image motion (image_deskew.use_parts / rotation_smoothing, #046).
--diag: per-scan ICP diagnostics (a KD-tree over the local map per scan; off by default here, it
does not change the trajectory).  For bags the written timestamps are the scans' header stamps
(the loader's own are the bag record times), so the evaluation matches them to the ground truth.
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

    is_bag = seq.suffix == ".bag" or (seq.is_dir() and any(seq.glob("*.bag")))
    if is_bag:
        from kiss_icp.datasets.rosbag import RosbagDataset
        dataset = RosbagDataset(seq, opts.get("topic", "/os_cloud_node/points"))
    else:
        from kiss_slam.tools.ncd_pcd import NewerCollege2020Pcd
        dataset = NewerCollege2020Pcd(seq)

    # Options as config fields, so they also reach the image-motion worker process.
    load_config = pipeline.load_config

    def load_with_overrides(path):
        config = load_config(path)
        config.image_deskew.seed = int(opts.get("seed", 0))
        config.image_deskew.parallel = "--parallel" in sys.argv
        config.image_deskew.intensity_scale = float(opts.get("intensity-scale", 255.0 / 1024.0))
        config.diagnostics.icp_metrics = "--diag" in sys.argv
        config.image_deskew.use_parts = opts.get("parts", "full")
        config.image_deskew.rotation_smoothing = int(opts.get("rot-smooth", 1))
        return config

    pipeline.load_config = load_with_overrides
    slam_pipeline = pipeline.SlamPipeline(
        dataset=dataset,
        config_file=Path(opts["config"]) if "config" in opts else None,
        n_scans=n_scans,
        image_deskew=arm != "kiss",
        image_detector=None if arm == "kiss" else arm,
    )
    if is_bag:
        # After SlamPipeline installed its reader: record each scan's header stamp on the way through.
        stamps, read = [], dataset.read_point_cloud

        def read_and_stamp(msg):
            stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
            return read(msg)

        dataset.read_point_cloud = read_and_stamp
        dataset.get_frames_timestamps = lambda: stamps
    slam_pipeline.run().print()


if __name__ == "__main__":      # required: image_deskew.parallel starts its worker with "spawn"
    main()
    # Everything is written: leave without the interpreter's shutdown.  (Added for a run of #044 that looked like a
    # hang at exit — main thread gone, pool threads waiting on a futex; the real cause was a kernel BUG in the ntfs3
    # driver while writing to the NTFS data disk, #046.  Kept: it does no harm.)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
