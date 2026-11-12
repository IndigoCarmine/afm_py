"""Recursively export a folder of Nanoscope SPM files to PNG images.

Scans ``INPUT_DIR`` recursively for SPM files, renders each one to a PNG, and
saves it under an export folder that is split into sub-folders by the image's
XY size.  Output file names are ``<UniqueID>_<original name>.png`` so that
identically-named scans from different sub-folders never collide.

Example layout produced::

    Export/
      2um/
        a1b2c3d4_scan001.png
      2umx1um/
        e5f6a7b8_lineprofile.png
      500nm/
        0f1e2d3c_grid.png

Usage::

    python examples/batch_export.py INPUT_DIR
    python examples/batch_export.py INPUT_DIR --export Export --flatten 2 \
        --colormap afmhot --group-by scan

Run with ``-h`` for all options.  Requires the ``png`` extra (Pillow):
``pip install "afm-py[png]"``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

# Allow running straight from a source checkout (python/ on the path) as well as
# from an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from afm_files import read_spm  # noqa: E402
from afm_files.convert import write_png  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def is_spm_file(path: Path) -> bool:
    """True for ``.spm`` files and Nanoscope numeric extensions (``.000`` etc.)."""
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix == ".spm":
        return True
    # Numeric extensions like .000, .001, .0034 …
    return len(suffix) > 1 and suffix[1:].isdigit()


def find_spm_files(root: Path, exclude: Path) -> list:
    """Recursively collect SPM files under ``root``, skipping the export dir."""
    exclude = exclude.resolve()
    files = []
    for path in sorted(root.rglob("*")):
        if not is_spm_file(path):
            continue
        # Don't re-ingest anything living inside the export directory.
        if exclude in path.resolve().parents:
            continue
        files.append(path)
    return files


def unique_id(path: Path, root: Path) -> str:
    """Short, stable id derived from the file's path relative to ``root``."""
    try:
        key = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        key = str(path.resolve())
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _fmt_nm(nm: float) -> str:
    """Format a length in nm as a compact folder-friendly label (e.g. ``2um``)."""
    if nm <= 0:
        return "unknown"
    if nm >= 1000:
        return f"{nm / 1000.0:.3g}um"
    return f"{nm:.3g}nm"


def size_folder(img, group_by: str) -> str:
    """Folder name describing the image's XY size."""
    if group_by == "pixels":
        return f"{img.samps_per_line}x{img.number_of_lines}px"
    # Physical scan size along the fast (X) and slow (Y) axes.
    sx, sy = img.scan_size_xy_nm
    x = _fmt_nm(sx)
    y = _fmt_nm(sy)
    return x if x == y else f"{x}x{y}"


def safe_name(stem: str) -> str:
    """Strip characters that are awkward in file names."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in stem).strip("_")


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
def export_tree(
    root: Path,
    export_dir: Path,
    *,
    group_by: str = "scan",
    channel=None,
    flatten=None,
    smooth_sigma: float = 0.0,
    colormap: str = "afmhot",
    zmin=None,
    zmax=None,
    overwrite: bool = True,
) -> tuple:
    """Export every SPM file under ``root`` to ``export_dir``.

    Returns ``(written, skipped, errors)`` where ``errors`` is a list of
    ``(path, message)`` tuples.
    """
    files = find_spm_files(root, exclude=export_dir)
    written, skipped, errors = [], [], []

    for path in files:
        try:
            img = read_spm(path, channel=channel, flatten=flatten, smooth_sigma=smooth_sigma)
            folder = export_dir / size_folder(img, group_by)
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / f"{unique_id(path, root)}_{safe_name(path.stem)}.png"
            if dst.exists() and not overwrite:
                skipped.append(dst)
                continue
            write_png(img, dst, colormap=colormap, zmin=zmin, zmax=zmax)
            written.append(dst)
            print(f"  {path}  ->  {dst.relative_to(export_dir.parent)}")
        except Exception as exc:  # keep going; report bad files at the end
            errors.append((path, str(exc)))
            print(f"  ! {path}: {exc}", file=sys.stderr)

    return written, skipped, errors


def _channel_arg(value):
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Recursively export SPM files to PNGs, grouped by XY size.")

    p.add_argument("-c", "--channel", default=None, help="Channel index or name substring (default: auto / Height)")
    p.add_argument(
        "--smooth", type=float, default=0.0, metavar="SIGMA", help="Gaussian smoothing sigma in pixels (default: 0)"
    )
    p.add_argument("--colormap", default="afmhot", help="Colormap: afmhot/gray/viridis or any matplotlib name")
    p.add_argument("--zmin", type=float, default=None, help="Lower z-limit (nm)")
    p.add_argument("--zmax", type=float, default=None, help="Upper z-limit (nm)")
    p.add_argument("--skip-existing", action="store_true", help="Do not overwrite PNGs that already exist")
    args = p.parse_args(argv)

    root = Path(r"E:\AFM")
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    export_dir = Path(r"e:\AFM_export")
    flatten = 2

    print(f"Scanning {root} -> {export_dir}")
    written, skipped, errors = export_tree(
        root,
        export_dir,
        group_by="scan",
        channel=_channel_arg(args.channel),
        flatten=flatten,
        smooth_sigma=0,
        colormap="gray",
        zmin=args.zmin,
        zmax=args.zmax,
        overwrite=not args.skip_existing,
    )

    print(f"\nDone: {len(written)} exported, {len(skipped)} skipped, {len(errors)} failed.")
    if errors:
        print("Failed files:", file=sys.stderr)
        for path, msg in errors:
            print(f"  {path}: {msg}", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
