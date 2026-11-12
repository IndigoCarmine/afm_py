# afm_files

A Python library for reading **Bruker Nanoscope SPM/AFM files** and converting
them to open formats, with a set of processing and file-conversion filters.

The file parser is a **native Rust core** (`afm_files._core`, built with
[maturin](https://www.maturin.rs/) + [pyo3](https://pyo3.rs/)), ported from
[IndigoCarmine/afm_files](https://github.com/IndigoCarmine/afm_files) (a Rust
desktop viewer): read the text header up to `\*File list end`, pick a channel
(Height/Retrace by default), decode the raw binary, and scale LSB integers to
nanometers using the embedded sensitivity calibration. A complete pure-Python
fallback ships alongside it, so the package still works if the extension is not
built — pick with `read_spm(..., backend="rust"|"python"|"auto")`
(`afm_files.HAS_NATIVE` reports whether the native core is available).

## Supported input formats

| Extension | Description |
|-----------|-------------|
| `.spm` | Bruker Nanoscope SPM (standard extension) |
| `.000`–`.009`, `.00000`… | Nanoscope files with numeric extensions |

Any file with a Nanoscope text header works regardless of extension.

## Install

Building the native core needs a **Rust toolchain** (`rustup`/`cargo`) and, on
Windows, the MSVC build tools. Then:

```sh
pip install maturin                          # build backend
python -m maturin build --release            # -> target/wheels/*.whl
pip install --force-reinstall target/wheels/afm_py-*.whl
```

Or, inside a virtualenv, the one-liner dev install:

```sh
python -m maturin develop --release          # builds + installs into the venv
```

Runtime install needs only NumPy. `png` extra (Pillow) is required for image
output; `viz` extra (matplotlib) adds extra colormaps and plotting:

```sh
pip install "afm-py[all] @ ."                # + Pillow and matplotlib
```

> The wheel is built against the stable ABI (`abi3-py39`), so one wheel works on
> CPython 3.9+. If you install the sdist without a Rust toolchain, the package
> still imports and runs via the pure-Python backend.

## Library usage

```python
from afm_files import read_spm

img = read_spm("scan.spm")                 # auto-selects the Height channel
print(img.summary())
img.data            # 2-D float64 array in nm, row 0 = top of image
img.scan_size_nm    # physical scan width (nm, fast axis)
img.scan_size_xy_nm # (x, y) scan extent (nm); y differs for rectangular scans
img.pixel_size_nm   # nm per pixel along x (pixel_size_y_nm for y)
img.metadata        # full header key/value dict

# pick a specific channel by index or name substring
img = read_spm("scan.spm", channel="Phase")

# apply processing while loading (matches the desktop viewer)
img = read_spm("scan.spm", flatten=2, smooth_sigma=1.0)

# list channels without decoding pixel data
from afm_files import list_channels
for ch in list_channels("scan.spm"):
    print(ch)       # [0] Height (Retrace)
```

### Processing filters

Filters take a 2-D array (nm) and return a new one. They are also registered by
name so they can be chained in a `Pipeline` or from the CLI.

```python
from afm_files import filters

flat  = filters.flatten_polynomial(img.data, order=2)     # per-line polynomial flatten
blur  = filters.gaussian_smooth(img.data, sigma=1.5)      # edge-clamped Gaussian
plane = filters.plane_subtract(img.data)                  # remove global tilt plane
rb    = filters.rolling_ball_subtract(img.data, radius=30)  # Sternberg rolling-ball

# chain by name / spec string
pipe = filters.Pipeline(["plane", "flatten:1", "rollball:30", "offset"])
img.data = pipe(img.data)
```

Built-in filters: `flatten:<order>`, `smooth:<sigma>`, `plane`,
`rollball:<radius>`, `offset` (zero-min), `zeromean`, `invert`,
`clip:<zmin>,<zmax>`.

**Rolling-ball background subtraction** (`rollball` / `rolling_ball_subtract`)
is the Sternberg algorithm ported from the desktop viewer's Rust core: a ball of
the given pixel radius is rolled under the surface (grayscale opening on a
0..100-normalized field, with an ImageJ-style shrink for large radii) and the
estimated background is subtracted. It runs in the native `_core` when built,
with an equivalent NumPy fallback otherwise. `filters.rolling_ball_background()`
returns just the background.

### File conversion

```python
from afm_files import convert, batch_convert

# read → optionally filter → write, format inferred from the extension
convert("scan.spm", "scan.png", filters=["flatten:2"], colormap="afmhot")
convert("scan.spm", "scan.csv")
convert("scan.spm", "scan.npy")

# convert a whole folder to flattened PNGs
from pathlib import Path
batch_convert(Path(".").glob("*.spm"), "out", "png", filters=["flatten:2"])
```

Output formats: `csv`, `tsv`, `txt`, `npy`, `png`, `tif`, `xyz`, `json`
(metadata). Colormaps: `afmhot`, `gray`, `viridis` (ports of the desktop app),
or any matplotlib colormap name when matplotlib is installed.

## Command line

```sh
# inspect
afm-convert scan.spm --list-channels
afm-convert scan.spm --info

# single file (format from the output extension)
afm-convert scan.spm -o scan.png --flatten 2 --colormap afmhot
afm-convert scan.spm -o scan.csv

# batch: many inputs → a directory, one output each
afm-convert *.spm -o pngs -f png --flatten 2

# arbitrary filter chain
afm-convert scan.spm -o scan.csv --filter plane --filter smooth:1.0
```

`python -m afm_files ...` is equivalent to the `afm-convert` command.

## Example: recursive batch export

[`examples/batch_export.py`](examples/batch_export.py) scans a directory tree
for SPM files and writes flattened PNGs into an export folder, split into
sub-folders by each image's XY size. File names are `<UniqueID>_<original>.png`
so identically-named scans from different sub-folders never collide.

```sh
python examples/batch_export.py /path/to/AFM_data --export Export --flatten 2
```

```
Export/
  4um/
    fbbf417a_scan.png
    b39eef26_scan.png      # same name, different source folder -> different id
  2um/
    2c1617b1_big.png
```

Options: `--group-by scan|pixels`, `--channel`, `--flatten N`, `--smooth SIGMA`,
`--colormap`, `--zmin/--zmax`, `--skip-existing`.

## How the scaling works

For each selected image block the loader reads `Z scale`. If it already carries
`nm/LSB`, that value is used directly; otherwise it combines `V/LSB` with the
referenced `@Sens.` `nm/V` sensitivity, taking the Bruker full-scale-voltage
convention into account (`min(V/LSB, Zscale_V / 2**bits)` × `nm/V`). Lines are
stored bottom-to-top in the file and are flipped so row 0 is the top of the
image.

## Development

Layout (maturin mixed project):

```
src/lib.rs              Rust core -> afm_files._core
python/afm_files/       Python API (parser fallback, filters, convert, CLI)
tests/                  synthetic-file tests (no instrument data needed)
```

```sh
python -m maturin build --release && pip install --force-reinstall target/wheels/afm_py-*.whl
python -m pytest tests -q          # or: python tests/test_afm_files.py
```

Rust source changes need a rebuild; Python-only changes just need a reinstall
(or a `maturin develop` in a venv). The test suite builds a tiny synthetic
Nanoscope file in memory and checks that the Rust and Python backends agree.

## License

MIT. Original Rust project © Yuhei Yamada.
