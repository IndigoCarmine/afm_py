"""Colormaps for rendering AFM height maps to RGB images.

``afmhot``, ``gray`` and ``viridis`` are vectorized NumPy ports of the
colormaps used by the upstream Kintuba AFM Viewer, so PNG output matches the
desktop app.  Any other name is looked up in matplotlib if it is installed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["normalize", "apply_colormap", "available_colormaps"]

# Viridis control points (same 8 stops as the Rust viewer).
_VIRIDIS_STOPS = np.array(
    [
        [0.267, 0.004, 0.329],
        [0.283, 0.141, 0.458],
        [0.254, 0.265, 0.530],
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.566, 0.551],
        [0.285, 0.706, 0.429],
        [0.993, 0.906, 0.144],
    ]
)


def _sigmoid(t, sigma, z):
    return 1.0 / (1.0 + np.exp(-sigma * (t - z)))


def _afmhot(t: np.ndarray) -> np.ndarray:
    t = t * 0.8 + 0.2  # shift/scale for more vibrant colors (matches upstream)
    r = np.clip(_sigmoid(t, 7.0, 0.6) * 1.05, 0.0, 1.0)
    g = np.clip(_sigmoid(t, 10.0, 0.7) * 1.1 - 0.05, 0.0, 1.0)
    b = np.clip(_sigmoid(t, 11.0, 0.7) * 1.6 - 0.55, 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def _gray(t: np.ndarray) -> np.ndarray:
    return np.stack([t, t, t], axis=-1)


def _viridis(t: np.ndarray) -> np.ndarray:
    n = len(_VIRIDIS_STOPS)
    scaled = t * (n - 1)
    lo = np.clip(np.floor(scaled).astype(int), 0, n - 2)
    frac = (scaled - lo)[..., None]
    c0 = _VIRIDIS_STOPS[lo]
    c1 = _VIRIDIS_STOPS[lo + 1]
    return c0 + frac * (c1 - c0)


_BUILTIN = {
    "afmhot": _afmhot,
    "afm hot": _afmhot,
    "afm_hot": _afmhot,
    "hot": _afmhot,
    "gray": _gray,
    "grey": _gray,
    "viridis": _viridis,
}


def available_colormaps() -> list:
    """Names of the built-in colormaps (matplotlib names also work if installed)."""
    return ["afmhot", "gray", "viridis"]


def normalize(data: np.ndarray, zmin=None, zmax=None) -> np.ndarray:
    """Scale ``data`` to ``[0, 1]`` given optional z-limits (defaults to min/max)."""
    data = np.asarray(data, dtype=np.float64)
    if zmin is None:
        zmin = float(data.min())
    if zmax is None:
        zmax = float(data.max())
    rng = max(zmax - zmin, np.finfo(np.float64).eps)
    return np.clip((data - zmin) / rng, 0.0, 1.0)


def apply_colormap(norm: np.ndarray, name: str = "afmhot") -> np.ndarray:
    """Map a normalized ``[0, 1]`` array to a ``uint8`` RGB image ``(H, W, 3)``."""
    norm = np.clip(np.asarray(norm, dtype=np.float64), 0.0, 1.0)
    key = name.strip().lower()
    if key in _BUILTIN:
        rgb = _BUILTIN[key](norm)
    else:
        rgb = _matplotlib_colormap(norm, name)
    return (rgb * 255.0).astype(np.uint8)


def _matplotlib_colormap(norm: np.ndarray, name: str) -> np.ndarray:
    try:
        import matplotlib
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ValueError(
            f"Unknown colormap {name!r}. Built-in options are "
            f"{available_colormaps()}; install matplotlib for more."
        ) from exc
    try:
        cmap = matplotlib.colormaps[name]  # registry API (matplotlib >= 3.5)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Unknown colormap {name!r}") from exc
    return cmap(norm)[..., :3]
