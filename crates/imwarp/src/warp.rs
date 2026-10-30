//! Warping an image with a homography, as `skimage.transform.warp` does with a matrix.

use ndarray::{Array2, ArrayView2};

use crate::error::{Result, WarpError};
use crate::interpolate::{Interpolation, map_coordinates};

/// Warp `image` into an array of `output_shape` `(rows, columns)`.
///
/// `inverse` is the 3x3 homography from output to input coordinates in scikit-image's
/// convention: it takes `(column, row, 1)` of an output pixel to `(column, row, w)` in the
/// input, which is divided by `w`.
pub fn warp(
    image: ArrayView2<'_, f64>,
    inverse: &[[f64; 3]; 3],
    output_shape: (usize, usize),
    options: &Interpolation,
) -> Result<Array2<f64>> {
    if inverse.iter().flatten().any(|v| !v.is_finite()) {
        return Err(WarpError::InvalidInput(
            "the inverse map holds a value that is not finite".into(),
        ));
    }
    let (ny, nx) = output_shape;
    let mut rows = Array2::<f64>::zeros(output_shape);
    let mut cols = Array2::<f64>::zeros(output_shape);
    for r in 0..ny {
        for c in 0..nx {
            let (x, y) = (c as f64, r as f64);
            let w = inverse[2][0] * x + inverse[2][1] * y + inverse[2][2];
            let src_x = (inverse[0][0] * x + inverse[0][1] * y + inverse[0][2]) / w;
            let src_y = (inverse[1][0] * x + inverse[1][1] * y + inverse[1][2]) / w;
            if let Some(v) = rows.get_mut((r, c)) {
                *v = src_y;
            }
            if let Some(v) = cols.get_mut((r, c)) {
                *v = src_x;
            }
        }
    }
    map_coordinates(image, rows.view(), cols.view(), options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BoundaryMode, Order};

    #[test]
    fn the_identity_returns_the_image_and_a_shift_moves_it() {
        let image = Array2::from_shape_fn((4, 5), |(r, c)| (r * 5 + c) as f64);
        let options = Interpolation::new(Order::Linear, BoundaryMode::Constant, 0.0);
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let same = warp(image.view(), &identity, (4, 5), &options).unwrap();
        assert_eq!(same, image);
        // Output (col, row) samples the input one column to the right.
        let shift = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let moved = warp(image.view(), &shift, (4, 5), &options).unwrap();
        assert_eq!(moved[[2, 1]], image[[2, 2]]);
        assert_eq!(moved[[2, 4]], 0.0);
        assert!(warp(image.view(), &[[f64::NAN; 3]; 3], (4, 5), &options).is_err());
    }
}
