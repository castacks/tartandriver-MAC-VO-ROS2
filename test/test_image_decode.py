"""ROS image payload decoding matches what the offline loader reads from the extracted PNG."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from mac_ros2.image_decode import decode_compressed, decode_image

RNG = np.random.default_rng(0)


def png_round_trip(image: np.ndarray, tmp_path) -> np.ndarray:
    """What `cv2.imread(IMREAD_UNCHANGED)` returns for the array the extraction writes."""
    path = str(tmp_path / "frame.png")
    cv2.imwrite(path, image)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


@pytest.mark.parametrize("encoding,dtype,channels", [
    ("mono8", np.uint8, 1), ("8UC1", np.uint8, 1), ("mono16", np.uint16, 1), ("16UC1", np.uint16, 1),
    ("bgr8", np.uint8, 3), ("rgb8", np.uint8, 3), ("8UC3", np.uint8, 3),
])
def test_raw_image_matches_extracted_png(encoding, dtype, channels, tmp_path):
    shape = (5, 7) if channels == 1 else (5, 7, channels)
    image = RNG.integers(0, np.iinfo(dtype).max, size=shape, dtype=dtype)
    decoded = decode_image(encoding, 5, 7, 7 * channels * image.itemsize, image.tobytes(), False)
    assert decoded.dtype == dtype and decoded.shape == shape
    np.testing.assert_array_equal(decoded, png_round_trip(image, tmp_path))


@pytest.mark.parametrize("encoding", ["bgra8", "rgba8"])
def test_alpha_is_dropped(encoding):
    image = RNG.integers(0, 255, size=(3, 4, 4), dtype=np.uint8)
    decoded = decode_image(encoding, 3, 4, 16, image.tobytes(), False)
    np.testing.assert_array_equal(decoded, image[..., :3])


def test_row_padding_is_skipped():
    image = RNG.integers(0, 65535, size=(3, 4), dtype=np.uint16)
    padded = np.zeros((3, 6), dtype=np.uint16)
    padded[:, :4] = image
    np.testing.assert_array_equal(decode_image("mono16", 3, 4, 12, padded.tobytes(), False), image)


def test_big_endian_mono16():
    image = RNG.integers(0, 65535, size=(3, 4), dtype=np.uint16)
    decoded = decode_image("mono16", 3, 4, 8, image.astype(">u2").tobytes(), True)
    assert decoded.dtype == np.uint16 and decoded.dtype.byteorder in ("=", "<", "|")
    np.testing.assert_array_equal(decoded, image)


@pytest.mark.parametrize("encoding", ["bayer_rggb8", "32FC1", "yuv422"])
def test_unsupported_encodings_raise(encoding):
    with pytest.raises(ValueError, match="unsupported"):
        decode_image(encoding, 1, 1, 4, bytes(4), False)


def test_short_payload_raises():
    with pytest.raises(ValueError, match="bytes"):
        decode_image("mono8", 4, 4, 4, bytes(8), False)


@pytest.mark.parametrize("image", [
    RNG.integers(0, 65535, size=(6, 5), dtype=np.uint16),
    RNG.integers(0, 255, size=(6, 5), dtype=np.uint8),
    RNG.integers(0, 255, size=(6, 5, 3), dtype=np.uint8),
])
def test_compressed_png_matches_extracted_png(image, tmp_path):
    ok, payload = cv2.imencode(".png", image)
    assert ok
    decoded = decode_compressed(payload.tobytes())
    assert decoded.dtype == image.dtype
    np.testing.assert_array_equal(decoded, png_round_trip(image, tmp_path))


def test_compressed_garbage_raises():
    with pytest.raises(ValueError, match="decode"):
        decode_compressed(b"not an image")
