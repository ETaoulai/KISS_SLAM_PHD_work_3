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
import numpy as np
import open3d as o3d
from map_closures.map_closures import MapClosures

from kiss_slam.config import LoopCloserConfig
from kiss_slam.local_map_graph import LocalMapGraph
from kiss_slam.voxel_map import VoxelMap

# lambda_geometric controls the geometry vs. photometry trade-off in ColoredICP.
#   lambda_geometric = 1.0  → pure geometry  (same as PointToPlane)
#   lambda_geometric = 0.0  → pure photometry
#   lambda_geometric = 0.968 → Open3D default (96.8 % geometry, 3.2 % colour)
#
# For narrow staircases where geometry is degenerate, pulling this toward 0.90
# gives more weight to the intensity channel, which distinguishes step edges
# from treads even when XYZ constraints are nearly rank-deficient.
#
# The value now comes from `config.intensity.lambda_geometric` so that it is
# recorded per run; this constant is only the fallback default.
_LAMBDA_GEOMETRIC = 0.90


def local_maps_overlap(source_pts, target_pts, pose, voxel_size):
    """Fraction of the smaller local map shared with the other, counted in voxels.

    intersection / min(|source|, |target|), all three counted as occupied voxels
    of size `voxel_size` (the density-map resolution), so the result is in [0, 1].

    The upstream code counted the target as `len(target_pts)`, which equals its
    voxel count only when the local maps are stored at exactly `voxel_size`.
    With `local_mapper.voxel_size` = 0.25 and `density_map_resolution` = 0.5 the
    target counted ~4x too many, the "overlap" came out 4.3-5.5, and the 0.4
    threshold could never reject a closure.
    """
    source_map = VoxelMap(voxel_size)
    source_map.integrate_frame(source_pts, pose)
    target_map = VoxelMap(voxel_size)
    target_map.add_points(target_pts)
    union_map = VoxelMap(voxel_size)
    union_map.integrate_frame(source_pts, pose)
    union_map.add_points(target_pts)

    num_source_voxels = source_map.num_voxels()
    num_target_voxels = target_map.num_voxels()
    intersection = num_source_voxels + num_target_voxels - union_map.num_voxels()
    return intersection / min(num_source_voxels, num_target_voxels)


class LoopCloser:
    def __init__(self, config: LoopCloserConfig, lambda_geometric: float = _LAMBDA_GEOMETRIC):
        self.config = config
        self.detector = MapClosures(config.detector)
        self.local_map_voxel_size = config.detector.density_map_resolution
        self.icp_threshold = np.sqrt(3) * self.local_map_voxel_size
        self.overlap_threshold = config.overlap_threshold

        # PointToPlane is the fallback when no intensity is available
        self._p2plane = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

        # ColoredICP is used when both point clouds carry intensity (as colors)
        self._colored = o3d.t.pipelines.registration.TransformationEstimationForColoredICP(
            lambda_geometric=lambda_geometric
        )

        self.termination_criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_rmse=1e-4
        )

    def compute(self, query_id, points, local_map_graph: LocalMapGraph):
        """Accepted closures of `query_id`, as a list of (ref_id, query_id, pose_constraint).

        top_k = 1 is upstream: only the best candidate is verified.  With top_k > 1 every
        candidate above the inliers threshold is verified and every accepted one is kept:
        a query map can revisit several earlier maps (church_02, #013: 8<->13 and 9<->13).
        """
        if self.config.top_k <= 1:
            candidates = [self.detector.get_best_closure(query_id, points)]
        else:
            candidates = sorted(
                self.detector.get_top_k_closures(query_id, points, self.config.top_k),
                key=lambda c: -c.number_of_inliers,
            )
        accepted = []
        for closure in candidates:
            if closure.number_of_inliers < self.config.detector.inliers_threshold:
                break
            ref_id = closure.source_id
            source = local_map_graph[ref_id].pcd
            target = local_map_graph[query_id].pcd
            print(f"\nKissSLAM| Closure Detected ({ref_id}<->{query_id}, {closure.number_of_inliers} inliers)")
            is_good, pose_constraint = self.validate_closure(source, target, closure.pose)
            if is_good and self.config.max_height_disagreement is not None:
                is_good = self.height_is_consistent(ref_id, query_id, pose_constraint, local_map_graph)
            if is_good:
                accepted.append((ref_id, query_id, pose_constraint))
        return accepted

    def height_is_consistent(self, ref_id, query_id, pose_constraint, local_map_graph):
        """Does the closure put the reference map at the height the odometry puts it?

        The pose graph reads `pose_constraint` as the reference node's pose in the query
        node's frame, the same quantity the odometry gives as inv(K_query) @ K_ref.  Both
        translations are projected on the query map's vertical (third row of its MapClosures
        ground alignment), since the sensor is held tilted and the node-frame z is not height.
        """
        up = self.detector.get_ground_alignment_from_id(query_id)[2, :3]
        up = up / np.linalg.norm(up)
        odometry = np.linalg.inv(local_map_graph[query_id].keypose) @ local_map_graph[ref_id].keypose
        dz_closure = up @ pose_constraint[:3, -1]
        dz_odometry = up @ odometry[:3, -1]
        disagreement = abs(dz_closure - dz_odometry)
        ok = disagreement <= self.config.max_height_disagreement
        print(f"KissSLAM| Height check: closure {dz_closure:+.2f} m, odometry {dz_odometry:+.2f} m "
              f"-> {'ok' if ok else 'Closure rejected for height disagreement'} ({disagreement:.2f} m)")
        return ok

    def validate_closure(self, source, target, initial_guess):
        """Validate a loop closure candidate using ICP.

        Uses ColoredICP if both local maps carry intensity information,
        otherwise falls back to PointToPlane.  ColoredICP is much more
        robust on repetitive-geometry environments (staircases, corridors)
        because the photometric channel disambiguates structurally similar
        but visually different locations (e.g. step 3 vs step 7).
        """
        use_colored = (
            "colors" in source.point
            and "colors" in target.point
        )

        if use_colored:
            estimation = self._colored
            print("KissSLAM| Using ColoredICP for closure validation (intensity available)")
        else:
            estimation = self._p2plane
            print("KissSLAM| Using PointToPlane ICP for closure validation (no intensity)")

        registration_result = o3d.t.pipelines.registration.icp(
            source,
            target,
            self.icp_threshold,
            initial_guess,
            estimation,
            self.termination_criteria,
        )

        source_pts = source.point.positions.numpy().astype(np.float64)
        target_pts = target.point.positions.numpy().astype(np.float64)
        pose = registration_result.transformation.numpy()
        overlap = local_maps_overlap(source_pts, target_pts, pose, self.local_map_voxel_size)

        closure_is_accepted = overlap > self.overlap_threshold
        print(f"KissSLAM| LocalMaps Overlap: {overlap:.3f}")
        if closure_is_accepted:
            print("KissSLAM| Closure Accepted")
        else:
            print("KissSLAM| Closure rejected for low overlap.")
        return closure_is_accepted, pose
