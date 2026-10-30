//! Errors of the warping crate.

use thiserror::Error;

/// Everything that can go wrong while interpolating an image.
#[derive(Debug, Error)]
pub enum WarpError {
    /// An argument violates a documented precondition.
    #[error("{0}")]
    InvalidInput(String),

    /// The requested interpolation order is not implemented.
    #[error("interpolation order {0} is not supported (use 0, 1 or 3)")]
    UnsupportedOrder(u8),
}

/// A result with a [`WarpError`].
pub type Result<T, E = WarpError> = std::result::Result<T, E>;
