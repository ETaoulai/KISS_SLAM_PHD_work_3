# MIT License
#
# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import csv
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from kiss_icp.pipeline import OdometryPipeline
from tqdm import tqdm, trange

from kiss_slam.config import load_config
from kiss_slam.occupancy_mapper import OccupancyGridMapper
from kiss_slam.slam import KissSLAM
from kiss_slam.tools.visualizer import RegistrationVisualizer, StubVisualizer

# CSV column order
_ICP_CSV_FIELDS = [
    "frame_idx",
    "icp_rms_error",
    "icp_inlier_count",
    "icp_inlier_ratio",
    "n_source_pts",
    "n_map_pts",
    "adaptive_sigma",
]


class SlamPipeline(OdometryPipeline):
    def __init__(
        self,
        dataset,
        config_file: Optional[Path] = None,
        visualize: bool = False,
        n_scans: int = -1,
        jump: int = 0,
        refuse_scans: bool = False,
    ):
        super().__init__(dataset=dataset, config=None, n_scans=n_scans, jump=jump)
        self.slam_config = load_config(config_file)
        self.config = self.slam_config.kiss_icp_config()
        self.visualize = visualize
        self.kiss_slam = KissSLAM(self.slam_config)
        self.visualizer = RegistrationVisualizer() if self.visualize else StubVisualizer()
        self.refuse_scans = refuse_scans

    def run(self):
        self._run_pipeline()
        self._run_evaluation()
        self._evaluate_closures()
        self._create_output_dir()
        self._write_result_poses()
        self._write_gt_poses()
        self._write_cfg()
        self._write_log()
        self._write_graph()
        self._write_closures()
        self._write_local_maps()
        self._write_icp_metrics()       # ← new
        self._plot_icp_metrics()        # ← new
        self._global_mapping()
        return self.results

    # ─────────────────────────────────────────────────────────────────────────
    # ICP diagnostics: CSV + plots
    # ─────────────────────────────────────────────────────────────────────────

    def _write_icp_metrics(self):
        """Write per-frame ICP metrics to <results_dir>/icp_metrics.csv."""
        csv_path = os.path.join(self.results_dir, "icp_metrics.csv")
        metrics = self.kiss_slam.icp_metrics_log
        if not metrics:
            print("KissSLAM| No ICP metrics to write.")
            return

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_ICP_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(metrics)

        print(f"KissSLAM| ICP metrics saved → {csv_path}  ({len(metrics)} frames)")

    def _plot_icp_metrics(self):
        """Save diagnostic plots of ICP errors and inlier point counts."""
        metrics = self.kiss_slam.icp_metrics_log
        if not metrics:
            return

        frames = [m["frame_idx"] for m in metrics]
        rms = [m["icp_rms_error"] for m in metrics]
        inlier_count = [m["icp_inlier_count"] for m in metrics]
        inlier_ratio = [m["icp_inlier_ratio"] for m in metrics]
        sigma = [m["adaptive_sigma"] for m in metrics]

        # ── Figure 1: RMS error + adaptive sigma ─────────────────────────────
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("ICP Diagnostics – KISS-SLAM", fontsize=13, fontweight="bold")

        ax = axes[0]
        ax.plot(frames, rms, color="#e74c3c", linewidth=0.8, label="ICP RMS error (m)")
        ax.set_ylabel("RMS error (m)")
        ax.set_title("Point-to-map RMS error (inliers only)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)
        # Highlight NaN frames (no map yet / no inliers) in grey
        nan_frames = [f for f, r in zip(frames, rms) if np.isnan(r)]
        if nan_frames:
            ax.axvspan(nan_frames[0] - 0.5, nan_frames[-1] + 0.5,
                       alpha=0.15, color="grey", label="no map / no inliers")

        ax = axes[1]
        ax.plot(frames, inlier_count, color="#2980b9", linewidth=0.8, label="Inlier points")
        ax.set_ylabel("Points")
        ax.set_title("Inlier point count per frame")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

        ax = axes[2]
        ax.plot(frames, sigma, color="#27ae60", linewidth=0.8, label="Adaptive σ")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("σ (m)")
        ax.set_title("KISS-ICP adaptive threshold σ")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.4)

        plt.tight_layout()
        plot_path = os.path.join(self.results_dir, "icp_metrics.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"KissSLAM| ICP metrics plot  → {plot_path}")

        # ── Figure 2: Inlier ratio ────────────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(frames, inlier_ratio, color="#8e44ad", linewidth=0.8)
        ax2.set_xlabel("Frame index")
        ax2.set_ylabel("Ratio")
        ax2.set_title("ICP inlier ratio (fraction of source points matched)")
        ax2.set_ylim(0, 1.05)
        # Draw a warning threshold line at 0.3 — below this SLAM often drifts
        ax2.axhline(0.3, color="#e74c3c", linestyle="--", linewidth=1,
                    label="Drift-risk threshold (0.30)")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.4)
        plt.tight_layout()
        ratio_path = os.path.join(self.results_dir, "icp_inlier_ratio.png")
        plt.savefig(ratio_path, dpi=150)
        plt.close(fig2)
        print(f"KissSLAM| Inlier ratio plot  → {ratio_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # Existing methods – unchanged below
    # ─────────────────────────────────────────────────────────────────────────

    def _run_pipeline(self):
        for idx in trange(self._first, self._last, unit=" frames", dynamic_ncols=True):
            scan, timestamps = self._next(idx)
            start_time = time.perf_counter_ns()
            self.kiss_slam.process_scan(scan, timestamps)
            self.times[idx - self._first] = time.perf_counter_ns() - start_time
            self.visualizer.update(self.kiss_slam)
        self.kiss_slam.generate_new_node()
        self.kiss_slam.local_map_graph.erase_last_local_map()
        self.poses, self.pose_graph = self.kiss_slam.fine_grained_optimization()
        self.poses = np.array(self.poses)

    def _global_mapping(self):
        if self.refuse_scans:
            from kiss_icp.preprocess import get_preprocessor

            if hasattr(self._dataset, "reset"):
                self._dataset.reset()
            ref_ground_alignment = self.kiss_slam.closer.detector.get_ground_alignment_from_id(0)
            deskewing_deltas = np.vstack(
                (
                    np.eye(4)[None],
                    np.eye(4)[None],
                    np.linalg.inv(self.poses[:-2]) @ self.poses[1:-1],
                )
            )
            preprocessor = get_preprocessor(self.config)
            occupancy_grid_mapper = OccupancyGridMapper(self.slam_config.occupancy_mapper)
            print("KissSLAM| Computing Occupancy Grid")
            for idx in trange(self._first, self._last, unit=" frames", dynamic_ncols=True):
                scan, timestamps = self._next(idx)
                deskewed_scan = preprocessor.preprocess(scan, timestamps, deskewing_deltas[idx])
                occupancy_grid_mapper.integrate_frame(
                    deskewed_scan, ref_ground_alignment @ self.poses[idx - self._first]
                )
            occupancy_grid_mapper.compute_3d_occupancy_information()
            occupancy_grid_mapper.compute_2d_occupancy_information()
            occupancy_dir = os.path.join(self.results_dir, "occupancy_grid")
            os.makedirs(occupancy_dir, exist_ok=True)
            occupancy_grid_mapper.write_3d_occupancy_grid(occupancy_dir)
            occupancy_2d_map_dir = os.path.join(occupancy_dir, "map2d")
            os.makedirs(occupancy_2d_map_dir, exist_ok=True)
            occupancy_grid_mapper.write_2d_occupancy_grid(occupancy_2d_map_dir)

    def _write_local_maps(self):
        local_maps_dir = os.path.join(self.results_dir, "local_maps")
        os.makedirs(local_maps_dir, exist_ok=True)
        self.kiss_slam.optimizer.write_graph(os.path.join(local_maps_dir, "local_map_graph.g2o"))
        plys_dir = os.path.join(local_maps_dir, "plys")
        os.makedirs(plys_dir, exist_ok=True)
        print("KissSLAM| Writing Local Maps on Disk")
        for local_map in tqdm(self.kiss_slam.local_map_graph.local_maps()):
            filename = os.path.join(plys_dir, "{:06d}.ply".format(local_map.id))
            local_map.write(filename)

    def _evaluate_closures(self):
        self.results.append(
            desc="Number of closures found", units="closures", value=len(self.kiss_slam.closures)
        )

    def _write_closures(self):
        locations = [pose[:3, -1] for pose in self.poses]
        loc_x = [loc[0] for loc in locations]
        loc_y = [loc[1] for loc in locations]
        plt.scatter(loc_x, loc_y, s=0.1, color="black")
        key_poses = self.kiss_slam.get_keyposes()
        for closure in self.kiss_slam.closures:
            i, j = closure
            plt.plot(
                [key_poses[i][0, -1], key_poses[j][0, -1]],
                [key_poses[i][1, -1], key_poses[j][1, -1]],
                color="red",
                linewidth=1,
                markersize=1,
            )
        plt.savefig(os.path.join(self.results_dir, "trajectory.png"), dpi=2000)
        plt.close()

    def _write_graph(self):
        self.pose_graph.write_graph(os.path.join(self.results_dir, "trajectory.g2o"))

    def _next(self, idx):
        dataframe = self._dataset[idx]
        frame, timestamps = dataframe
        return frame, timestamps
