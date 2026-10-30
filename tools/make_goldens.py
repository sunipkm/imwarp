"""Golden values for the `imwarp` tests: SciPy results on small random arrays.

Writes crates/imwarp/tests/data/scipy.json. Every case holds ``data`` (ny, nx), ``coords``
(2, N) with points inside and well outside the array, and SciPy's ``map_coordinates`` output for
each interpolation order (0, 1, 3) and boundary mode (``prefilter=order > 1``, ``cval=-7.5``),
plus a NaN fill value, and the unfiltered orders 1 and 3.

    pip install numpy scipy
    python tools/make_goldens.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

OUT = Path(__file__).resolve().parents[1] / "crates" / "imwarp" / "tests" / "data" / "scipy.json"
MODES = ("constant", "nearest", "mirror", "reflect", "wrap")
SHAPES = ((7, 9), (4, 6), (2, 3), (1, 5), (6, 1))


def flat(a: np.ndarray) -> list:
    """Row-major values; NaN is written as null."""
    return [None if np.isnan(v) else float(v) for v in np.asarray(a, dtype=float).ravel()]


def main() -> None:
    rng = np.random.default_rng(2024)
    cases = []
    for shape in SHAPES:
        ny, nx = shape
        data = rng.normal(size=shape) * 10
        ys = np.concatenate([np.linspace(-3.7, ny + 2.5, 23), [0.0, ny - 1.0, ny - 1.0, -0.5, ny - 0.5, 0.25]])
        xs = np.concatenate([np.linspace(-4.1, nx + 3.7, 19), [0.0, nx - 1.0, 0.0, nx - 0.5, -0.5, 0.75]])
        coords = np.stack([np.repeat(ys, len(xs)), np.tile(xs, len(ys))])
        out = {}
        for order in (0, 1, 3):
            for mode in MODES:
                out[f"o{order}_{mode}"] = flat(
                    map_coordinates(data, coords, order=order, mode=mode, cval=-7.5, prefilter=order > 1)
                )
        out["o1_constant_nan"] = flat(map_coordinates(data, coords, order=1, mode="constant", cval=np.nan))
        out["o1_noprefilter_constant"] = flat(
            map_coordinates(data, coords, order=1, mode="constant", cval=0.0, prefilter=False)
        )
        out["o3_noprefilter_mirror"] = flat(
            map_coordinates(data, coords, order=3, mode="mirror", cval=0.0, prefilter=False)
        )
        cases.append({"shape": list(shape), "data": flat(data), "coords": flat(coords), "expected": out})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"cases": cases}, separators=(",", ":")))
    print(f"{OUT.name}: {len(cases)} cases, {OUT.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
