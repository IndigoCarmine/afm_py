"""Image-processing filters for AFM height maps.

Every filter takes a 2-D NumPy array (nanometers) and returns a new 2-D array
of the same shape.  Filters are also registered by name so they can be driven
from the command line or chained into a :class:`Pipeline`, e.g.::

    afm-convert scan.spm -o out -f png --filter flatten:2 --filter smooth:1.5
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Union

import numpy as np

from .parser import _core  # native filter core (may be None)

__all__ = [
    "flatten_polynomial",
    "gaussian_smooth",
    "plane_subtract",
    "rolling_ball_subtract",
    "rolling_ball_background",
    "subtract_min",
    "subtract_mean",
    "invert",
    "clip",
    "Pipeline",
    "get_filter",
    "parse_filter_spec",
    "apply_filters",
    "FILTERS",
    "list_filters",
]


# --------------------------------------------------------------------------- #
# Individual filters
# --------------------------------------------------------------------------- #
def flatten_polynomial(data: np.ndarray, order: int = 1) -> np.ndarray:
    """Subtract a per-line least-squares polynomial baseline (orders 0-3).

    Each row is fit independently against x normalized to ``[0, 1]`` and the
    fitted baseline is subtracted -- the standard Nanoscope line-flatten.
    """
    if not 0 <= order <= 3:
        raise ValueError("Polynomial order must be 0-3")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("flatten_polynomial expects a 2-D array")
    n_lines, n_samples = data.shape
    if n_lines == 0 or n_samples == 0:
        return data.copy()

    x = np.linspace(0.0, 1.0, n_samples) if n_samples > 1 else np.zeros(1)
    # Shared Vandermonde matrix (same x for every row): columns 1, x, x^2, ...
    vander = np.vander(x, order + 1, increasing=True)
    # Solve V @ coeffs = data.T in one least-squares call (coeffs per line).
    coeffs, *_ = np.linalg.lstsq(vander, data.T, rcond=None)
    baseline = (vander @ coeffs).T
    return data - baseline


def gaussian_smooth(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur with edge-clamped borders (radius = ceil(3*sigma))."""
    data = np.asarray(data, dtype=np.float64)
    if sigma <= 0:
        return data.copy()
    if data.ndim != 2:
        raise ValueError("gaussian_smooth expects a 2-D array")

    radius = int(math.ceil(3.0 * sigma))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()

    height, width = data.shape

    # Horizontal pass.
    cols = np.arange(width)
    tmp = np.zeros_like(data)
    for k, weight in zip(offsets, kernel):
        idx = np.clip(cols + k, 0, width - 1)
        tmp += data[:, idx] * weight

    # Vertical pass.
    rows = np.arange(height)
    out = np.zeros_like(data)
    for k, weight in zip(offsets, kernel):
        idx = np.clip(rows + k, 0, height - 1)
        out += tmp[idx, :] * weight
    return out


def plane_subtract(data: np.ndarray) -> np.ndarray:
    """Subtract the global best-fit plane ``z = a + b*x + c*y`` from the image."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("plane_subtract expects a 2-D array")
    h, w = data.shape
    ys, xs = np.mgrid[0:h, 0:w]
    a = np.column_stack([np.ones(data.size), xs.ravel(), ys.ravel()])
    coeffs, *_ = np.linalg.lstsq(a, data.ravel(), rcond=None)
    plane = (a @ coeffs).reshape(data.shape)
    return data - plane


# --------------------------------------------------------------------------- #
# Rolling-ball background subtraction (Sternberg)
# --------------------------------------------------------------------------- #
def rolling_ball_subtract(data: np.ndarray, radius: int = 30) -> np.ndarray:
    """Sternberg rolling-ball background subtraction (``radius`` in pixels).

    A ball of the given radius is rolled underneath the surface (grayscale
    opening on a 0..100-normalized field, with an ImageJ-style shrink for large
    radii) and the resulting background is subtracted.  Broad background is
    removed while features narrower than the ball are preserved.

    Uses the native Rust core (``afm_files._core``) when available and falls
    back to an equivalent NumPy implementation otherwise.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("rolling_ball_subtract expects a 2-D array")
    radius = int(radius)
    if radius <= 0:
        return data.copy()

    h, w = data.shape
    if _core is not None:
        buf = np.ascontiguousarray(data, dtype="<f8").tobytes()
        out = _core.rolling_ball_subtract(buf, w, h, radius)
        return np.frombuffer(out, dtype="<f8").reshape(h, w).copy()
    return data - _rolling_ball_background_py(data, radius)


def rolling_ball_background(data: np.ndarray, radius: int = 30) -> np.ndarray:
    """Return just the estimated rolling-ball background (``data`` minus this is
    :func:`rolling_ball_subtract`)."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("rolling_ball_background expects a 2-D array")
    if int(radius) <= 0:
        return np.zeros_like(data)
    return data - rolling_ball_subtract(data, radius)


def _rolling_ball_background_py(data: np.ndarray, radius: int) -> np.ndarray:
    """Pure-NumPy port of the Rust rolling-ball background estimate."""
    h, w = data.shape
    dmin = float(data.min())
    dmax = float(data.max())
    rng = max(dmax - dmin, float(np.finfo(np.float64).eps))
    work = 100.0
    norm = (data - dmin) / rng * work

    if radius <= 10:
        shrink = 1
    elif radius <= 30:
        shrink = 2
    elif radius <= 100:
        shrink = 4
    else:
        shrink = 8
    small_r = max(radius // shrink, 1)

    sw, sh, small = _shrink_min(norm, shrink)
    small_bg = _ball_opening(small, small_r)
    bg_norm = _enlarge(small_bg, sw, sh, w, h)
    return bg_norm / work * rng + dmin


def _shrink_min(data: np.ndarray, k: int):
    h, w = data.shape
    if k <= 1:
        return w, h, data.copy()
    sh = -(-h // k)
    sw = -(-w // k)
    padded = np.full((sh * k, sw * k), np.inf)
    padded[:h, :w] = data
    small = padded.reshape(sh, k, sw, k).min(axis=(1, 3))
    return sw, sh, small


def _build_ball(r: int):
    r2 = float(r * r)
    dj, di = np.mgrid[-r : r + 1, -r : r + 1]
    d2 = (di * di + dj * dj).astype(np.float64)
    mask = d2 <= r2
    heights = np.sqrt(np.clip(r2 - d2, 0.0, None))
    return di[mask].astype(int), dj[mask].astype(int), heights[mask]


def _ball_morph(g, di_arr, dj_arr, bh_arr, dilate: bool):
    h, w = g.shape
    acc = np.full((h, w), -np.inf if dilate else np.inf)
    rows = np.arange(h)
    cols = np.arange(w)
    for di, dj, bh in zip(di_arr, dj_arr, bh_arr):
        ys = np.clip(rows + dj, 0, h - 1)
        xs = np.clip(cols + di, 0, w - 1)
        shifted = g[np.ix_(ys, xs)]
        if dilate:
            np.maximum(acc, shifted + bh, out=acc)
        else:
            np.minimum(acc, shifted - bh, out=acc)
    return acc


def _ball_opening(g: np.ndarray, r: int) -> np.ndarray:
    di, dj, bh = _build_ball(r)
    eroded = _ball_morph(g, di, dj, bh, dilate=False)
    return _ball_morph(eroded, di, dj, bh, dilate=True)


def _enlarge(small: np.ndarray, sw: int, sh: int, w: int, h: int) -> np.ndarray:
    if sw == w and sh == h:
        return small.copy()
    sx = (sw - 1) / (w - 1) if w > 1 else 0.0
    sy = (sh - 1) / (h - 1) if h > 1 else 0.0
    gy = np.arange(h) * sy
    y0 = np.floor(gy).astype(int)
    y1 = np.minimum(y0 + 1, sh - 1)
    ty = (gy - y0)[:, None]
    gx = np.arange(w) * sx
    x0 = np.floor(gx).astype(int)
    x1 = np.minimum(x0 + 1, sw - 1)
    tx = (gx - x0)[None, :]
    v00 = small[np.ix_(y0, x0)]
    v01 = small[np.ix_(y0, x1)]
    v10 = small[np.ix_(y1, x0)]
    v11 = small[np.ix_(y1, x1)]
    top = v00 + (v01 - v00) * tx
    bot = v10 + (v11 - v10) * tx
    return top + (bot - top) * ty


def subtract_min(data: np.ndarray) -> np.ndarray:
    """Shift the image so its minimum becomes zero."""
    data = np.asarray(data, dtype=np.float64)
    return data - data.min()


def subtract_mean(data: np.ndarray) -> np.ndarray:
    """Shift the image so its mean becomes zero."""
    data = np.asarray(data, dtype=np.float64)
    return data - data.mean()


def invert(data: np.ndarray) -> np.ndarray:
    """Flip the height axis (negate)."""
    return -np.asarray(data, dtype=np.float64)


def clip(data: np.ndarray, zmin: float = None, zmax: float = None) -> np.ndarray:
    """Clip heights to ``[zmin, zmax]`` (either bound may be omitted)."""
    return np.clip(np.asarray(data, dtype=np.float64), zmin, zmax)


# --------------------------------------------------------------------------- #
# Registry + pipeline
# --------------------------------------------------------------------------- #
# name -> (function, parameter-caster).  The caster turns positional CLI string
# arguments into typed kwargs.
FILTERS: dict = {
    "flatten": (flatten_polynomial, lambda *a: {"order": int(a[0])} if a else {}),
    "smooth": (gaussian_smooth, lambda *a: {"sigma": float(a[0])} if a else {}),
    "plane": (plane_subtract, lambda *a: {}),
    "rollball": (rolling_ball_subtract, lambda *a: {"radius": int(a[0])} if a and a[0] != "" else {}),
    "offset": (subtract_min, lambda *a: {}),
    "zeromean": (subtract_mean, lambda *a: {}),
    "invert": (invert, lambda *a: {}),
    "clip": (
        clip,
        lambda *a: {
            "zmin": float(a[0]) if len(a) > 0 and a[0] != "" else None,
            "zmax": float(a[1]) if len(a) > 1 and a[1] != "" else None,
        },
    ),
}


def list_filters() -> list:
    """Names of all registered filters."""
    return sorted(FILTERS)


def get_filter(name: str) -> Callable:
    """Look up a filter function by registry name."""
    try:
        return FILTERS[name][0]
    except KeyError:
        raise KeyError(
            f"Unknown filter {name!r}. Available: {', '.join(list_filters())}"
        )


def parse_filter_spec(spec: str):
    """Parse a ``"name:arg1,arg2"`` string into ``(function, kwargs)``.

    Examples: ``"flatten:2"``, ``"smooth:1.5"``, ``"plane"``, ``"clip:0,50"``.
    """
    name, _, rest = spec.partition(":")
    name = name.strip()
    if name not in FILTERS:
        raise KeyError(
            f"Unknown filter {name!r}. Available: {', '.join(list_filters())}"
        )
    func, caster = FILTERS[name]
    args = [a.strip() for a in rest.split(",")] if rest else []
    return func, caster(*args)


class Pipeline:
    """An ordered chain of filters applied left-to-right.

    Steps may be filter names (``"plane"``), spec strings (``"flatten:2"``),
    ``(function, kwargs)`` tuples, or plain callables.
    """

    def __init__(self, steps: Iterable = ()):
        self.steps = [self._coerce(s) for s in steps]

    @staticmethod
    def _coerce(step):
        if isinstance(step, str):
            return parse_filter_spec(step)
        if isinstance(step, tuple) and len(step) == 2 and callable(step[0]):
            return step
        if callable(step):
            return (step, {})
        raise TypeError(f"Cannot interpret pipeline step: {step!r}")

    def then(self, step) -> "Pipeline":
        """Return a new pipeline with ``step`` appended."""
        return Pipeline([*self.steps, step])

    def __call__(self, data: np.ndarray) -> np.ndarray:
        for func, kwargs in self.steps:
            data = func(data, **kwargs)
        return data

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        names = [getattr(f, "__name__", repr(f)) for f, _ in self.steps]
        return f"Pipeline({names})"


def apply_filters(data: np.ndarray, specs: Union[Iterable, None]) -> np.ndarray:
    """Apply an iterable of filter specs/callables to ``data``."""
    if not specs:
        return np.asarray(data, dtype=np.float64)
    return Pipeline(specs)(data)
