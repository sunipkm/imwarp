from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

__version__: str

Mode = Literal["constant", "nearest", "mirror", "reflect", "wrap", "edge", "symmetric"]

def map_coordinates(
    image: ArrayLike,
    coordinates: ArrayLike,
    order: int = 1,
    mode: Mode = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Sample a 2-D ``image`` at ``coordinates`` of shape ``(2, ...)`` (row, column), like
    ``scipy.ndimage.map_coordinates``. Orders 0, 1 and 3."""

def warp(
    image: ArrayLike,
    inverse_map: ArrayLike,
    output_shape: tuple[int, int] | None = None,
    order: int = 1,
    mode: Mode = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> np.ndarray:
    """Warp a 2-D ``image`` with a 3x3 ``inverse_map`` (output ``(column, row, 1)`` to input),
    like ``skimage.transform.warp``."""
