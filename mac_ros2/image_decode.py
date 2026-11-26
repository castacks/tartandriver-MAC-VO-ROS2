"""ROS image payloads -> numpy arrays, as `cv2.imread(IMREAD_UNCHANGED)` returns the extracted PNG."""

from __future__ import annotations

import cv2
import numpy as np

# encoding -> (dtype, channels); channel order is kept as sent, like cv_bridge passthrough
_ENCODINGS: dict[str, tuple[type[np.generic], int]] = {
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "8UC3": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
    "8UC4": (np.uint8, 4),
}


def decode_image(encoding: str, height: int, width: int, step: int, data: bytes, is_bigendian: bool) -> np.ndarray:
    """`sensor_msgs/Image` fields -> HxW or HxWx3 array; alpha is dropped as the extraction does."""
    if encoding not in _ENCODINGS:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    dtype, channels = _ENCODINGS[encoding]
    itemsize = np.dtype(dtype).itemsize
    row_bytes = width * channels * itemsize
    if step < row_bytes or len(data) < step * height:
        raise ValueError(f"{encoding} {width}x{height} image has step {step} and {len(data)} bytes")
    rows = np.frombuffer(data, dtype=np.uint8, count=step * height).reshape(height, step)[:, :row_bytes]
    order = ">" if is_bigendian else "<"
    image = rows.copy().view(np.dtype(dtype).newbyteorder(order)).astype(dtype, copy=False)
    image = image.reshape(height, width, channels) if channels > 1 else image.reshape(height, width)
    return np.ascontiguousarray(image[..., :3]) if channels == 4 else image


def decode_compressed(data: bytes) -> np.ndarray:
    """`sensor_msgs/CompressedImage` payload -> array via `cv2.imdecode(IMREAD_UNCHANGED)`; alpha is dropped."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError("failed to decode compressed image")
    return np.ascontiguousarray(image[..., :3]) if image.ndim == 3 and image.shape[2] == 4 else image
