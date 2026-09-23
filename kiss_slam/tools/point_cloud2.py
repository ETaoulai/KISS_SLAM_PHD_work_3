# Copyright 2008 Willow Garage, Inc.
# (see original kiss_icp/tools/point_cloud2.py for full licence header)
#
# Extended by kiss-slam intensity branch:
#   read_point_cloud() now also returns a (N,) float32 intensity array.
#   If the bag topic has no intensity/reflectivity/ring-amplitude field the
#   array is None, and all intensity-aware code degrades gracefully.

import sys
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

# Fields tried in order when looking for intensity in a PointCloud2 message.
_INTENSITY_FIELD_NAMES = [
    "intensity",
    "reflectivity",
    "ring_intensity",
    "amplitude",
    "i",
]

_DATATYPES = {
    "int8":    np.dtype(np.int8),
    "uint8":   np.dtype(np.uint8),
    "int16":   np.dtype(np.int16),
    "uint16":  np.dtype(np.uint16),
    "int32":   np.dtype(np.int32),
    "uint32":  np.dtype(np.uint32),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

DUMMY_FIELD_PREFIX = "unnamed_field"


def read_point_cloud(
    msg,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Read XYZ, per-point timestamps, and intensity from a PointCloud2 message.

    Returns
    -------
    points     : (N, 3) float64 – XYZ positions (NaN rows removed)
    timestamps : (N,) float64  – per-point timestamps (empty array if absent)
    intensity  : (N,) float32  – normalised [0, 1] intensity, or None if the
                                  message carries no intensity-like field
    """
    field_names = ["x", "y", "z"]
    t_field = None
    i_field = None

    msg_field_names = {f.name for f in msg.fields}

    for field in msg.fields:
        if field.name in ["t", "timestamp", "time"] and t_field is None:
            t_field = field.name
            field_names.append(t_field)
        if field.name in _INTENSITY_FIELD_NAMES and i_field is None:
            i_field = field.name
            field_names.append(i_field)

    points_structured = read_points(msg, field_names=field_names)

    points = np.column_stack(
        [points_structured["x"], points_structured["y"], points_structured["z"]]
    )

    # Remove NaN rows (keep track of valid mask for intensity)
    valid = ~np.any(np.isnan(points), axis=1)
    points = points[valid]

    if t_field:
        timestamps = points_structured[t_field].astype(np.float64)[valid]
    else:
        timestamps = np.array([])

    if i_field:
        raw = points_structured[i_field].astype(np.float32)[valid]
        # Normalise to [0, 1] robustly
        lo, hi = np.percentile(raw, [1, 99])
        if hi > lo:
            intensity = np.clip((raw - lo) / (hi - lo), 0.0, 1.0)
        else:
            intensity = np.zeros_like(raw)
    else:
        intensity = None

    return points.astype(np.float64), timestamps, intensity


_RING_FIELD_NAMES = ["ring", "channel", "laser_id", "line"]


def read_point_cloud_raw(
    msg,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Read XYZ, timestamps, RAW intensity and ring from a PointCloud2 message.

    For the image-motion deskew (`image_deskew.enabled`, `kiss_slam.intensity_deskew`):
    the intensity panorama needs the sensor's own intensity scale (Hesai: 0-255, turned
    into a uint8 image as is) and the laser/ring index of every point.  Same input as
    `scripts/precompute_i3_motion.py::read_raw`, so the online estimate reproduces the
    precomputed one.

    Returns
    -------
    points     : (N, 3) float64 – XYZ positions (NaN rows removed)
    timestamps : (N,) float64  – per-point timestamps (empty array if absent)
    intensity  : (N,) float64  – intensity as stored in the message (NOT normalised),
                                  or None if the message carries no intensity-like field
    ring       : (N,) int64    – laser/ring index, or None if absent
    """
    field_names = ["x", "y", "z"]
    t_field = i_field = r_field = None
    for field in msg.fields:
        if field.name in ["t", "timestamp", "time"] and t_field is None:
            t_field = field.name
            field_names.append(t_field)
        if field.name in _INTENSITY_FIELD_NAMES and i_field is None:
            i_field = field.name
            field_names.append(i_field)
        if field.name in _RING_FIELD_NAMES and r_field is None:
            r_field = field.name
            field_names.append(r_field)

    s = read_points(msg, field_names=field_names)
    points = np.column_stack([s["x"], s["y"], s["z"]])
    valid = ~np.any(np.isnan(points), axis=1)
    points = points[valid]
    timestamps = s[t_field].astype(np.float64)[valid] if t_field else np.array([])
    intensity = s[i_field].astype(np.float64)[valid] if i_field else None
    ring = s[r_field].astype(np.int64)[valid] if r_field else None
    return points.astype(np.float64), timestamps, intensity, ring


# ── unchanged helpers from kiss_icp ──────────────────────────────────────────

def read_points(
    cloud,
    field_names: Optional[List[str]] = None,
    uvs: Optional[Iterable] = None,
    reshape_organized_cloud: bool = False,
) -> np.ndarray:
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )
    if field_names is not None:
        assert all(fn in points.dtype.names for fn in field_names), \
            "Requested field is not in the fields of the PointCloud!"
        points = points[list(field_names)]
    if bool(sys.byteorder != "little") != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)
    if uvs is not None:
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        points = points[uvs]
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)
    return points


def get_datatype_name(field) -> str:
    for attr_name, attr_value in vars(field).items():
        if attr_name.lower() in _DATATYPES and attr_value == field.datatype:
            return attr_name.lower()
    raise ValueError(f"Unknown datatype code {field.datatype} for field {vars(field)}")


def dtype_from_fields(fields: Iterable, point_step: Optional[int] = None) -> np.dtype:
    field_names, field_offsets, field_datatypes = [], [], []
    for i, field in enumerate(fields):
        datatype = _DATATYPES[get_datatype_name(field)]
        name = field.name if field.name else f"{DUMMY_FIELD_PREFIX}_{i}"
        assert field.count > 0
        for a in range(field.count):
            subfield_name = f"{name}_{a}" if field.count > 1 else name
            assert subfield_name not in field_names, "Duplicate field names!"
            field_names.append(subfield_name)
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)
    dtype_dict = {"names": field_names, "formats": field_datatypes, "offsets": field_offsets}
    if point_step is not None:
        dtype_dict["itemsize"] = point_step
    return np.dtype(dtype_dict)
