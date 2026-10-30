#![doc = include_str!("../README.md")]

mod boundary;
mod error;
mod interpolate;
mod spline;
mod warp;

pub use boundary::BoundaryMode;
pub use error::{Result, WarpError};
pub use interpolate::{Interpolation, Order, map_coordinates};
pub use warp::warp;
