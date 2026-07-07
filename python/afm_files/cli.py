"""Command-line interface: ``afm-convert``.

Convert one or more Bruker Nanoscope SPM files to CSV / NPY / PNG / XYZ /
JSON, with optional processing filters.

Examples
--------
List channels in a file::

    afm-convert scan.spm --list-channels

Convert a folder of scans to flattened PNGs::

    afm-convert *.spm -o pngs -f png --flatten 2 --colormap afmhot

Chain arbitrary filters::

    afm-convert scan.spm -o scan.csv --filter plane --filter smooth:1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .convert import SUPPORTED_FORMATS, convert
from .filters import list_filters
from .parser import list_channels, read_spm


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="afm-convert",
        description="Convert Bruker Nanoscope SPM/AFM files to open formats.",
    )
    p.add_argument("inputs", nargs="+", help="SPM file(s) to read")
    p.add_argument(
        "-o", "--output",
        help="Output file (single input) or output directory (with -f)",
    )
    p.add_argument(
        "-f", "--format",
        choices=SUPPORTED_FORMATS,
        help="Output format. Required when writing many files to a directory.",
    )
    p.add_argument(
        "-c", "--channel",
        help="Channel index (int) or name substring, e.g. 'Height'. Default: auto.",
    )

    # Convenience filter shortcuts (applied in this order, before --filter).
    p.add_argument("--flatten", type=int, metavar="ORDER",
                   help="Per-line polynomial flatten of the given order (0-3)")
    p.add_argument("--smooth", type=float, metavar="SIGMA",
                   help="Gaussian smoothing with the given sigma (pixels)")
    p.add_argument("--filter", action="append", default=[], dest="filters",
                   metavar="SPEC",
                   help="Extra filter, e.g. 'plane', 'flatten:2', 'clip:0,50'. "
                        "Repeatable. Filters known: " + ", ".join(list_filters()))

    # Image options.
    p.add_argument("--colormap", default="afmhot",
                   help="Colormap for image output (afmhot/gray/viridis or a "
                        "matplotlib name). Default: afmhot")
    p.add_argument("--zmin", type=float, help="Lower z-limit for image scaling (nm)")
    p.add_argument("--zmax", type=float, help="Upper z-limit for image scaling (nm)")

    # Info modes.
    p.add_argument("--list-channels", action="store_true",
                   help="List channels in each input and exit")
    p.add_argument("--info", action="store_true",
                   help="Print a summary of each input and exit")
    p.add_argument("--version", action="version",
                   version=f"afm-convert (afm_files) {__version__}")
    return p


def _channel_arg(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _collect_filters(args) -> list:
    specs = []
    if args.flatten is not None:
        specs.append(f"flatten:{args.flatten}")
    if args.smooth is not None:
        specs.append(f"smooth:{args.smooth}")
    specs.extend(args.filters)
    return specs


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    channel = _channel_arg(args.channel)

    # --- info modes -------------------------------------------------------- #
    if args.list_channels:
        for src in args.inputs:
            print(f"{src}:")
            for ch in list_channels(src):
                print(f"  {ch}")
        return 0

    if args.info:
        for src in args.inputs:
            print(read_spm(src, channel=channel).summary())
        return 0

    # --- conversion -------------------------------------------------------- #
    filters = _collect_filters(args)
    inputs = [Path(s) for s in args.inputs]

    if args.output is None:
        print("error: -o/--output is required for conversion", file=sys.stderr)
        return 2

    out = Path(args.output)
    # Treat -o as a directory when converting many files, or when it has no
    # recognizable file suffix and a format was given.
    many = len(inputs) > 1
    out_is_dir = out.is_dir() or many or (args.format is not None and out.suffix == "")

    if out_is_dir:
        if args.format is None:
            print("error: -f/--format is required when writing to a directory",
                  file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for src in inputs:
            dst = out / f"{src.stem}.{args.format}"
            written.append(
                convert(src, dst, fmt=args.format, channel=channel,
                        filters=filters, colormap=args.colormap,
                        zmin=args.zmin, zmax=args.zmax)
            )
        for w in written:
            print(w)
    else:
        dst = convert(inputs[0], out, fmt=args.format, channel=channel,
                      filters=filters, colormap=args.colormap,
                      zmin=args.zmin, zmax=args.zmax)
        print(dst)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
