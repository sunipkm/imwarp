# imwarp

2-D image interpolation and warping in pure Rust, with the results of
`scipy.ndimage.map_coordinates` and `skimage.transform.warp`.

- `map_coordinates`: sample an image at fractional pixel positions. Orders 0 (nearest), 1
  (bilinear) and 3 (cubic B-spline with prefilter); the boundary modes `constant`, `nearest`
  (`edge`), `mirror` (`symmetric`), `reflect` and `wrap`.
- `warp`: warp an image with a 3x3 homography, as `skimage.transform.warp` does with a matrix.

The behavior was ported from SciPy's `ni_interpolation.c` and `ni_splines.c` and is checked
against SciPy's output (`tools/make_goldens.py`) to 1e-11. Rows are processed in parallel
with rayon.

```rust
use imwarp::{BoundaryMode, Interpolation, Order, map_coordinates};
use ndarray::array;

let image = array![[0.0, 1.0], [2.0, 3.0]];
let rows = array![[0.5]];
let cols = array![[0.5]];
let options = Interpolation::new(Order::Linear, BoundaryMode::Nearest, 0.0);
let out = map_coordinates(image.view(), rows.view(), cols.view(), &options)?;
assert_eq!(out[[0, 0]], 1.5);
# Ok::<(), imwarp::WarpError>(())
```

Python bindings: the `imwarp` package on PyPI (`pip install imwarp`).

MIT licensed.
