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
from copy import deepcopy as copy
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from kiss_slam.voxel_map import VoxelMap


class LocalMap:
    def __init__(self, id: np.uint64, keypose: np.ndarray):
        self.id = id
        self.keypose = keypose
        self.local_trajectory = [np.eye(4)]
        self.pcd = None          # o3d.t.geometry.PointCloud (with normals, optionally colors)

    @property
    def endpose(self):
        return self.keypose @ self.local_trajectory[-1]

    def write(self, filename):
        local_map_pcd = copy(self.pcd)
        local_map_pcd.transform(self.keypose)
        o3d.t.io.write_point_cloud(filename, local_map_pcd)

    @property
    def has_intensity(self) -> bool:
        """True if this local map has colour (= intensity) information."""
        return self.pcd is not None and "colors" in self.pcd.point


class LocalMapGraph:
    def __init__(self):
        self.graph = dict()
        local_map0 = LocalMap(id=0, keypose=np.eye(4))
        local_map0.local_trajectory.clear()
        self.graph[0] = local_map0

    def __getitem__(self, key):
        return self.graph[key]

    def local_maps(self):
        for local_map in self.graph.values():
            yield local_map

    def keyposes(self):
        for local_map in self.graph.values():
            yield local_map.keypose

    @property
    def last_id(self):
        return next(reversed(self.graph))

    @property
    def last_local_map(self):
        return self.graph[self.last_id]

    @property
    def last_keypose(self):
        return self.last_local_map.keypose

    def erase_local_map(self, key: np.uint64):
        self.graph.pop(key)

    def erase_last_local_map(self):
        self.erase_local_map(self.last_id)

    def finalize_local_map(
        self,
        voxel_grid: VoxelMap,
        intensity_lookup_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """Finalise the current local map node and open a new one.

        Parameters
        ----------
        voxel_grid         : the current mapping VoxelMap
        intensity_lookup_fn: optional callable  points (N,3) → intensity (N,), points in
                             this local map's own frame (the frame the voxel grid and the
                             odometry poses of this node use).  When provided, intensity is
                             stored as Open3D colours so that ColoredICP can use it in loop
                             closure.
        """
        local_map = self.last_local_map

        # Build Open3D pcd with normals (same as original)
        pcd = voxel_grid.open3d_pcd_with_normals()

        # Optionally attach intensity as colours
        if intensity_lookup_fn is not None:
            try:
                # Local-map frame, the frame the intensity was accumulated in.  Moving the
                # points to the global frame with the keypose (as before) made every lookup
                # miss for every node but the first, whose keypose is the identity.
                pts_local = pcd.point.positions.numpy().astype(np.float64)
                intensity = intensity_lookup_fn(pts_local).astype(np.float32)
                intens_rgb = np.repeat(intensity.reshape(-1, 1), 3, axis=1)
                pcd.point.colors = o3d.core.Tensor(
                    intens_rgb, dtype=o3d.core.Dtype.Float32
                )
            except Exception as e:
                # Non-fatal: if anything goes wrong we just skip intensity
                print(f"KissSLAM| [warn] intensity attachment failed: {e}")

        local_map.pcd = pcd

        proto_id = local_map.id + 1
        proto_keypose = local_map.endpose
        new_local_map = LocalMap(proto_id, np.copy(proto_keypose))
        self.graph[new_local_map.id] = new_local_map
