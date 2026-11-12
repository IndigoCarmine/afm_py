"""File-conversion filters: turn Nanoscope SPM files into open formats.

Each ``write_*`` function saves a 2-D array; :func:`convert` reads an SPM file,
optionally runs processing filters, and dispatches on the output extension.

Supported output formats:

===========  ===================================================
Extension    Content
===========  ===================================================
``.csv``     Comma-separated height grid (nm)
``.tsv``     Tab-separated height grid (nm)
``.txt``     Whitespace-separated height grid (nm)
``.npy``     NumPy binary array (nm)
``.png``     Colormapped image (needs Pillow)
``.tif``     Colormapped image (needs Pillow)
``.xyz``     ``x y z`` point list in nm
``.json``    Header metadata only
===========  ===================================================
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np

from . import colormaps
from .filters import apply_filters
from .parser import SpmImage, read_spm

__all__ = [
    "write_csv",
    "write_npy",
    "write_png",
    "write_xyz",
    "write_metadata",
    "convert",
    "batch_convert",
    "SUPPORTED_FORMATS",
]

# extension (without dot) -> format key
_ALIASES = {
    "csv": "csv",
    "tsv": "tsv",
    "txt": "txt",
    "dat": "txt",
    "npy": "npy",
    "png": "png",
    "tif": "png",
    "tiff": "png",
    "xyz": "xyz",
    "json": "json",
}

SUPPORTED_FORMATS = sorted(set(_ALIASES.values()))


def _as_data(obj: Union[SpmImage, np.ndarray]) -> np.ndarray:
    return obj.data if isinstance(obj, SpmImage) else np.asarray(obj)


# --------------------------------------------------------------------------- #
# Individual writers
# --------------------------------------------------------------------------- #
def write_csv(obj, path, delimiter: str = ",", fmt: str = "%.6g") -> Path:
    """Write the height grid as a delimited text file (nm)."""
    path = Path(path)
    np.savetxt(path, _as_data(obj), delimiter=delimiter, fmt=fmt)
    return path


def write_npy(obj, path) -> Path:
    """Write the height grid as a ``.npy`` binary array (nm)."""
    path = Path(path)
    np.save(path, _as_data(obj))
    # np.save appends .npy if missing; normalize the returned path.
    return path if path.suffix == ".npy" else path.with_suffix(path.suffix + ".npy")


def write_xyz(obj, path, fmt: str = "%.6g") -> Path:
    """Write an ``x y z`` point list in nm (uses pixel size when available)."""
    path = Path(path)
    data = _as_data(obj)
    h, w = data.shape
    if isinstance(obj, SpmImage):
        px, py = obj.pixel_size_nm, obj.pixel_size_y_nm
    else:
        px = py = 1.0
    ys, xs = np.mgrid[0:h, 0:w]
    cols = np.column_stack([xs.ravel() * px, ys.ravel() * py, data.ravel()])
    # Headerless 3-column "x y z" (nm), the usual interchange form for AFM data.
    np.savetxt(path, cols, fmt=fmt)
    return path


def write_png(
    obj,
    path,
    colormap: str = "afmhot",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> Path:
    """Render the height map to a colormapped image (PNG/TIFF). Needs Pillow."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Writing images requires Pillow. Install it with "
            "`pip install afm-py[png]` (or `pip install pillow`)."
        ) from exc
    path = Path(path)
    data = _as_data(obj)
    norm = colormaps.normalize(data, zmin, zmax)
    rgb = colormaps.apply_colormap(norm, colormap)
    Image.fromarray(rgb).save(path)  # mode inferred from uint8 HxWx3 array
    return path


def write_metadata(obj, path) -> Path:
    """Write header metadata as JSON (only meaningful for :class:`SpmImage`)."""
    path = Path(path)
    if isinstance(obj, SpmImage):
        payload = {
            "channel_name": obj.channel_name,
            "channel_index": obj.channel_index,
            "scan_size_nm": obj.scan_size_nm,
            "scan_size_y_nm": obj.scan_size_xy_nm[1],
            "samps_per_line": obj.samps_per_line,
            "number_of_lines": obj.number_of_lines,
            "pixel_size_nm": obj.pixel_size_nm,
            "pixel_size_y_nm": obj.pixel_size_y_nm,
            "nm_per_lsb": obj.nm_per_lsb,
            "available_channels": [
                {"index": c.index, "name": c.name, "direction": c.direction}
                for c in obj.available_channels
            ],
            "metadata": obj.metadata,
        }
    else:
        payload = {"shape": list(np.asarray(obj).shape)}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def _resolve_format(path: Path, fmt: Optional[str]) -> str:
    key = (fmt or path.suffix.lstrip(".")).lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unsupported output format {key!r}. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    return _ALIASES[key]


def _write(obj, path: Path, kind: str, colormap: str, zmin, zmax) -> Path:
    if kind == "csv":
        return write_csv(obj, path, delimiter=",")
    if kind == "tsv":
        return write_csv(obj, path, delimiter="\t")
    if kind == "txt":
        return write_csv(obj, path, delimiter=" ")
    if kind == "npy":
        return write_npy(obj, path)
    if kind == "png":
        return write_png(obj, path, colormap=colormap, zmin=zmin, zmax=zmax)
    if kind == "xyz":
        return write_xyz(obj, path)
    if kind == "json":
        return write_metadata(obj, path)
    raise ValueError(f"Unhandled format {kind!r}")  # pragma: no cover


def convert(
    src,
    dst,
    *,
    fmt: Optional[str] = None,
    channel: Union[int, str, None] = None,
    filters: Optional[Iterable] = None,
    colormap: str = "afmhot",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> Path:
    """Read ``src`` (an SPM path or :class:`SpmImage`), filter it, and write ``dst``.

    Parameters
    ----------
    src:
        Path to a Nanoscope file, or an already-loaded :class:`SpmImage`.
    dst:
        Output path.  The format is inferred from its extension unless ``fmt``
        is given.
    fmt:
        Explicit output format (e.g. ``"png"``), overriding the extension.
    channel:
        Channel index or name substring to load (SPM inputs only).
    filters:
        Iterable of filter specs/callables applied before writing, e.g.
        ``["flatten:2", "smooth:1.0"]``.
    colormap, zmin, zmax:
        Rendering options for image output.
    """
    dst = Path(dst)
    kind = _resolve_format(dst, fmt)

    if isinstance(src, SpmImage):
        img = src
    else:
        img = read_spm(src, channel=channel)

    if filters:
        img.data = apply_filters(img.data, filters)

    return _write(img, dst, kind, colormap, zmin, zmax)


def batch_convert(
    sources: Iterable,
    out_dir,
    fmt: str,
    *,
    channel: Union[int, str, None] = None,
    filters: Optional[Iterable] = None,
    colormap: str = "afmhot",
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
) -> list:
    """Convert many SPM files into ``out_dir``, one output per input.

    Each input ``foo.spm`` is written to ``out_dir/foo.<fmt>``.  Returns the
    list of written paths.  This stops on the first error; callers wanting to
    tolerate bad files should loop over :func:`convert` themselves.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = _resolve_format(Path(f"x.{fmt}"), fmt)  # validate early
    written = []
    for src in sources:
        src = Path(src)
        dst = out_dir / f"{src.stem}.{fmt}"
        written.append(
            convert(
                src,
                dst,
                fmt=ext,
                channel=channel,
                filters=filters,
                colormap=colormap,
                zmin=zmin,
                zmax=zmax,
            )
        )
    return written
