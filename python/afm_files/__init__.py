"""afm_files -- read Bruker Nanoscope SPM/AFM files and convert them.

Quick start
-----------
>>> from afm_files import read_spm
>>> img = read_spm("scan.spm", flatten=2)
>>> img.data.shape
(512, 512)
>>> img.summary()

Convert to another format::

>>> from afm_files import convert
>>> convert("scan.spm", "scan.png", filters=["flatten:2"], colormap="afmhot")

Or from the shell::

    afm-convert scan.spm -o out -f png --flatten 2
"""

from __future__ import annotations

from . import colormaps, convert as _convert_mod, filters
from .convert import (
    batch_convert,
    convert,
    write_csv,
    write_npy,
    write_png,
    write_xyz,
)
from .filters import (
    Pipeline,
    apply_filters,
    flatten_polynomial,
    gaussian_smooth,
    plane_subtract,
    rolling_ball_subtract,
)
from .parser import HAS_NATIVE, ChannelInfo, SpmImage, list_channels, read_spm

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "HAS_NATIVE",
    # parsing
    "read_spm",
    "list_channels",
    "SpmImage",
    "ChannelInfo",
    # filters
    "filters",
    "flatten_polynomial",
    "gaussian_smooth",
    "plane_subtract",
    "rolling_ball_subtract",
    "apply_filters",
    "Pipeline",
    # conversion
    "convert",
    "batch_convert",
    "write_csv",
    "write_npy",
    "write_png",
    "write_xyz",
    # colormaps
    "colormaps",
]
