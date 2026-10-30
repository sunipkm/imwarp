# imwarp

2-D image interpolation and warping with the results of `scipy.ndimage.map_coordinates` and
`skimage.transform.warp`, in Rust, with Python bindings that need neither SciPy nor
scikit-image.

| | |
|---|---|
| `crates/imwarp` | the Rust library ([crates.io](https://crates.io/crates/imwarp)) |
| `crates/imwarp-py` | the PyO3 binding (`imwarp._native`), not published on its own |
| `python/` | the `imwarp` Python package ([PyPI](https://pypi.org/project/imwarp/)) |
| `tools/make_goldens.py` | regenerates the SciPy reference values the tests compare with |

```python
import numpy as np
import imwarp

image = np.random.default_rng(0).random((64, 64))
rows, cols = np.mgrid[0:64, 0:64] + 0.5
out = imwarp.map_coordinates(image, np.stack([rows, cols]), order=3, mode="reflect")
shifted = imwarp.warp(image, [[1, 0, 2], [0, 1, 0], [0, 0, 1]])
```

Orders 0, 1 and 3 are implemented (SciPy's 2, 4 and 5 are not implemented). The cubic order uses the same
B-spline prefilter as SciPy; `scikit-image` bicubic kernel (used by `skimage.transform.warp`
with a transform object) is not a B-spline and is not reproduced.
