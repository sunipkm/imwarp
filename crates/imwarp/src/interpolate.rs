//! Evaluating the interpolant at arbitrary coordinates.

use ndarray::{Array2, ArrayView2, Axis};
use rayon::prelude::*;

use crate::boundary::BoundaryMode;
use crate::error::{Result, WarpError};
use crate::spline::spline_filter;

/// Interpolation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Order {
    /// Nearest neighbour.
    Nearest,
    /// Bilinear.
    #[default]
    Linear,
    /// Bicubic B-spline (with prefilter).
    Cubic,
}

impl Order {
    /// From SciPy's numeric order. Orders 2, 4 and 5 are not implemented.
    pub fn from_u8(order: u8) -> Result<Self> {
        match order {
            0 => Ok(Self::Nearest),
            1 => Ok(Self::Linear),
            3 => Ok(Self::Cubic),
            other => Err(WarpError::UnsupportedOrder(other)),
        }
    }

    const fn degree(self) -> usize {
        match self {
            Self::Nearest => 0,
            Self::Linear => 1,
            Self::Cubic => 3,
        }
    }
}

/// How to interpolate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interpolation {
    /// Interpolation order.
    pub order: Order,
    /// Behaviour beyond the edge of the image.
    pub mode: BoundaryMode,
    /// Value used outside the image in [`BoundaryMode::Constant`].
    pub cval: f64,
    /// Convert samples to spline coefficients first. Only meaningful for cubic order;
    /// defaults to `true` for it, like SciPy.
    pub prefilter: bool,
}

impl Interpolation {
    /// Interpolate with `order`, treating the outside as `mode`, filling with `cval` for
    /// [`BoundaryMode::Constant`]. Cubic order prefilters by default, like SciPy.
    #[must_use]
    pub fn new(order: Order, mode: BoundaryMode, cval: f64) -> Self {
        Self {
            order,
            mode,
            cval,
            prefilter: order == Order::Cubic,
        }
    }

    /// Turn the spline prefilter on or off (only cubic order uses it).
    #[must_use]
    pub fn with_prefilter(mut self, prefilter: bool) -> Self {
        self.prefilter = prefilter;
        self
    }
}

impl Default for Interpolation {
    /// Linear, constant boundary, fill 0.
    fn default() -> Self {
        Self::new(Order::Linear, BoundaryMode::Constant, 0.0)
    }
}

/// Interpolation footprint along one axis: up to four taps.
struct Taps {
    index: [usize; 4],
    weight: [f64; 4],
    len: usize,
}

/// Number of samples edge-padded before cubic prefiltering in `nearest` mode.
const NEAREST_PAD: usize = 12;

/// Sample `image` at fractional pixel positions `(rows[i, j], cols[i, j])`.
///
/// Positions are pixel indices (centre of the first pixel is `0.0`). The result has the
/// shape of `rows` and `cols`, which must match.
pub fn map_coordinates(
    image: ArrayView2<'_, f64>,
    rows: ArrayView2<'_, f64>,
    cols: ArrayView2<'_, f64>,
    options: &Interpolation,
) -> Result<Array2<f64>> {
    if rows.dim() != cols.dim() {
        return Err(WarpError::InvalidInput(
            "row and column coordinate arrays differ in shape".into(),
        ));
    }
    if image.is_empty() {
        return Err(WarpError::InvalidInput(
            "cannot interpolate an empty image".into(),
        ));
    }
    let order = options.order;
    let mode = options.mode;

    let filtering = order == Order::Cubic && options.prefilter;
    let pad = if filtering && mode == BoundaryMode::Nearest {
        NEAREST_PAD
    } else {
        0
    };
    let mut coefficients = if pad > 0 {
        edge_padded(&image, pad)
    } else {
        image.to_owned()
    };
    if filtering {
        spline_filter(&mut coefficients, mode.filter_extension());
    }
    let (ny, nx) = coefficients.dim();
    let (out_ny, out_nx) = rows.dim();

    let mut out = Array2::<f64>::zeros((out_ny, out_nx));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(rows.axis_iter(Axis(0)).into_par_iter())
        .zip(cols.axis_iter(Axis(0)).into_par_iter())
        .for_each(|((mut out_row, row_coords), col_coords)| {
            for ((slot, &r), &c) in out_row.iter_mut().zip(row_coords).zip(col_coords) {
                *slot = match (taps(r, ny, order, mode, pad), taps(c, nx, order, mode, pad)) {
                    (Some(ty), Some(tx)) => {
                        let mut total = 0.0;
                        for a in 0..ty.len {
                            for b in 0..tx.len {
                                // Coefficient times the weight of each axis, in that order.
                                total += coefficients[[ty.index[a], tx.index[b]]]
                                    * ty.weight[a]
                                    * tx.weight[b];
                            }
                        }
                        total
                    }
                    _ => options.cval,
                };
            }
        });
    Ok(out)
}

fn edge_padded(image: &ArrayView2<'_, f64>, pad: usize) -> Array2<f64> {
    let (ny, nx) = image.dim();
    Array2::from_shape_fn((ny + 2 * pad, nx + 2 * pad), |(j, i)| {
        image[[
            j.saturating_sub(pad).min(ny - 1),
            i.saturating_sub(pad).min(nx - 1),
        ]]
    })
}

/// Footprint for coordinate `coord` along an axis of length `len`, or `None` when the
/// point is outside the image (and so takes the fill value).
fn taps(coord: f64, len: usize, order: Order, mode: BoundaryMode, pad: usize) -> Option<Taps> {
    if coord.is_nan() {
        return None;
    }
    let mut cc = coord + pad as f64;
    if mode != BoundaryMode::Nearest {
        cc = mode.map_coordinate(cc, len);
    }
    if cc <= -1.0 && mode != BoundaryMode::Nearest {
        return None;
    }
    let degree = order.degree();
    let start = if degree % 2 == 1 {
        cc.floor()
    } else {
        (cc + 0.5).floor()
    } as i64
        - (degree / 2) as i64;

    let weight = weights(cc, order);
    let spline_mode = mode.spline_mode();
    let mut taps = Taps {
        index: [0; 4],
        weight,
        len: degree + 1,
    };
    for (ll, slot) in taps.index.iter_mut().enumerate().take(degree + 1) {
        let idx = start + ll as i64;
        *slot = if (0..len as i64).contains(&idx) {
            idx as usize
        } else {
            spline_mode.map_coordinate(idx as f64, len).max(0.0) as usize
        };
    }
    Some(taps)
}

/// B-spline weights of the taps for coordinate `cc`.
fn weights(cc: f64, order: Order) -> [f64; 4] {
    let mut w = [0.0; 4];
    match order {
        Order::Nearest => w[0] = 1.0,
        Order::Linear => {
            let x = cc - cc.floor();
            w[0] = 1.0 - x;
            w[1] = 1.0 - w[0];
        }
        Order::Cubic => {
            let x = cc - cc.floor();
            let (y, z) = (x, 1.0 - x);
            w[1] = (y * y * (y - 2.0) * 3.0 + 4.0) / 6.0;
            w[2] = (z * z * (z - 2.0) * 3.0 + 4.0) / 6.0;
            w[0] = z * z * z / 6.0;
            w[3] = 1.0 - w[0] - w[1] - w[2];
        }
    }
    w
}
