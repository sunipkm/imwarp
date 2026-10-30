"""imwarp against SciPy and scikit-image."""
import numpy as np
import pytest

import imwarp as pw

scipy_ndimage = pytest.importorskip("scipy.ndimage")
skimage_transform = pytest.importorskip("skimage.transform")

MODES = ["constant", "nearest", "mirror", "reflect", "wrap"]


@pytest.mark.parametrize("order", [0, 1, 3])
@pytest.mark.parametrize("mode", MODES)
def test_map_coordinates_matches_scipy(order, mode):
    rng = np.random.default_rng(order)
    image = rng.random((17, 23))
    coords = rng.uniform(-6, 28, size=(2, 5, 7))
    ours = pw.map_coordinates(image, coords, order=order, mode=mode, cval=-3.0)
    theirs = scipy_ndimage.map_coordinates(image, coords, order=order, mode=mode, cval=-3.0)
    assert ours.shape == (5, 7)
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def _inverse():
    theta = np.deg2rad(12.0)
    return np.array([[np.cos(theta), -np.sin(theta), 3.0], [np.sin(theta), np.cos(theta), -2.0], [0, 0, 1.0]])


@pytest.mark.parametrize("order", [0, 1])
def test_warp_matches_skimage(order):
    image = np.random.default_rng(7).random((40, 50))
    inverse = _inverse()
    ours = pw.warp(image, inverse, order=order, mode="edge")
    theirs = skimage_transform.warp(image, inverse, order=order, mode="edge", preserve_range=True)
    np.testing.assert_allclose(ours, theirs, rtol=1e-7, atol=1e-7)


def test_cubic_warp_is_the_b_spline_scipy_uses():
    # scikit-image's own 2-D kernel for order 3 is not a B-spline, so SciPy is the reference.
    image = np.random.default_rng(7).random((40, 50))
    inverse = _inverse()
    # skimage's (x, y) matrix as scipy's (row, column) matrix and offset.
    matrix = inverse[:2, :2][::-1, ::-1]
    offset = inverse[:2, 2][::-1]
    ours = pw.warp(image, inverse, order=3, mode="nearest")
    theirs = scipy_ndimage.affine_transform(image, matrix, offset=offset, order=3, mode="nearest")
    np.testing.assert_allclose(ours, theirs, rtol=1e-8, atol=1e-8)


def test_arguments_are_checked():
    image = np.zeros((4, 4))
    with pytest.raises(ValueError, match="shape"):
        pw.map_coordinates(image, np.zeros((3, 2)))
    with pytest.raises(ValueError, match="order"):
        pw.map_coordinates(image, np.zeros((2, 2)), order=2)
    with pytest.raises(ValueError, match="boundary mode"):
        pw.map_coordinates(image, np.zeros((2, 2)), mode="bogus")
    with pytest.raises(ValueError, match="3x3"):
        pw.warp(image, np.eye(2))
