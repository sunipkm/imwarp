//! `imwarp._native`: `scipy.ndimage.map_coordinates` and `skimage.transform.warp` in Rust.

use imwarp::{BoundaryMode, Interpolation, Order};
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn err(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn options(order: u8, mode: &str, cval: f64, prefilter: bool) -> PyResult<Interpolation> {
    let order = Order::from_u8(order).map_err(err)?;
    let mode: BoundaryMode = mode.parse().map_err(err)?;
    Ok(Interpolation::new(order, mode, cval).with_prefilter(prefilter))
}

/// Sample `image` at `coordinates`, an array of shape `(2, ...)` holding the row and column
/// of every point (SciPy's convention). The result has the shape `coordinates.shape[1:]`.
#[pyfunction]
#[pyo3(signature = (image, coordinates, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
fn map_coordinates<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    coordinates: PyReadonlyArrayDyn<'py, f64>,
    order: u8,
    mode: &str,
    cval: f64,
    prefilter: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let options = options(order, mode, cval, prefilter)?;
    let coordinates = coordinates.as_array();
    let shape = coordinates.shape().to_vec();
    if shape.first() != Some(&2) {
        return Err(err(format!(
            "coordinates must have shape (2, ...): the row and column of each point, not {shape:?}"
        )));
    }
    let n: usize = shape[1..].iter().product();
    let flat = coordinates
        .to_owned()
        .into_shape_with_order((2, 1, n))
        .map_err(err)?;
    let rows = flat.index_axis(Axis(0), 0).to_owned();
    let cols = flat.index_axis(Axis(0), 1).to_owned();
    let image = image.as_array();
    let out = py
        .detach(|| imwarp::map_coordinates(image, rows.view(), cols.view(), &options))
        .map_err(err)?;
    let out: ArrayD<f64> = out.into_shape_with_order(IxDyn(&shape[1..])).map_err(err)?;
    Ok(out.into_pyarray(py))
}

/// Warp `image` with `inverse_map`, a 3x3 homography taking `(column, row, 1)` of an output
/// pixel to the input (scikit-image's convention), into `output_shape` (the image's shape by
/// default).
#[pyfunction]
#[pyo3(signature = (image, inverse_map, output_shape = None, order = 1, mode = "constant", cval = 0.0, prefilter = true))]
#[allow(clippy::too_many_arguments)]
fn warp<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    inverse_map: PyReadonlyArray2<'py, f64>,
    output_shape: Option<(usize, usize)>,
    order: u8,
    mode: &str,
    cval: f64,
    prefilter: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let options = options(order, mode, cval, prefilter)?;
    let matrix = inverse_map.as_array();
    if matrix.dim() != (3, 3) {
        return Err(err(format!(
            "inverse_map must be 3x3, not {:?}",
            matrix.dim()
        )));
    }
    let mut m = [[0.0; 3]; 3];
    for ((i, j), v) in matrix.indexed_iter() {
        m[i][j] = *v;
    }
    let image = image.as_array();
    let shape = output_shape.unwrap_or_else(|| image.dim());
    let out: Array2<f64> = py
        .detach(|| imwarp::warp(image, &m, shape, &options))
        .map_err(err)?;
    Ok(out.into_pyarray(py))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(map_coordinates, m)?)?;
    m.add_function(wrap_pyfunction!(warp, m)?)?;
    Ok(())
}
