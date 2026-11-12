"""Tests for the afm_files package.

These build a tiny synthetic Nanoscope file in memory so the parser, filters
and converters can be exercised without a real instrument file.  Runnable both
under pytest and directly (``python tests/test_afm_files.py``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import afm_files
from afm_files import batch_convert, colormaps, convert, filters, read_spm
from afm_files.cli import main as cli_main

# Known scale factors used to build the synthetic file.
V_PER_LSB = 0.006176541
Z_SCALE_V = 1.988846
NM_PER_V = 45.0
SAMPS = 4
LINES = 4
DATA_OFFSET = 4096


def _expected_nm_per_lsb() -> float:
    max_int = float(1 << 16)  # bytes/pixel = 2
    effective = min(V_PER_LSB, Z_SCALE_V / max_int)
    return effective * NM_PER_V


def make_synthetic_spm(path: Path) -> np.ndarray:
    """Write a minimal valid SPM file; return the raw (bottom-to-top) int grid."""
    # Raw values as stored in the file (line order = bottom-to-top).
    raw = np.arange(SAMPS * LINES, dtype=np.int16).reshape(LINES, SAMPS) * 3
    data_bytes = raw.astype("<i2").tobytes()

    header = "\n".join(
        [
            r"\*File list",
            r"\Version: 0x09300201",
            r"\*Ciao scan list",
            r"\Scan Size: 4000 nm",
            r"\@Sens. Zsens: V 45.0 nm/V",
            r"\*Ciao image list",
            r"\Data offset: %d" % DATA_OFFSET,
            r"\Data length: %d" % len(data_bytes),
            r"\Samps/line: %d" % SAMPS,
            r"\Number of lines: %d" % LINES,
            r"\Bytes/pixel: 2",
            r"\Line Direction: Retrace",
            r'\@2:Image Data: S [Height] "Height"',
            r"\@2:Z scale: V [Sens. Zsens] (%s V/LSB) %s V" % (V_PER_LSB, Z_SCALE_V),
            r"\*File list end",
            "",
        ]
    ).encode("ascii")

    assert len(header) <= DATA_OFFSET, "header longer than data offset"
    blob = header + b"\x00" * (DATA_OFFSET - len(header)) + data_bytes
    path.write_bytes(blob)
    return raw


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def test_parse_scale_and_orientation(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    raw = make_synthetic_spm(spm)
    img = read_spm(spm)

    assert img.shape == (LINES, SAMPS)
    assert img.channel_name == "Height"
    assert img.scan_size_nm == 4000.0
    assert img.pixel_size_nm == 1000.0

    scale = _expected_nm_per_lsb()
    assert np.isclose(img.nm_per_lsb, scale)

    # File is stored bottom-to-top; parser flips it so row 0 is the top.
    expected = raw[::-1, :].astype(np.float64) * scale
    assert np.allclose(img.data, expected)


def test_backends_agree(tmpdir):
    """The pure-Python and native Rust parsers must produce identical results."""
    from afm_files.parser import HAS_NATIVE

    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)

    py = read_spm(spm, backend="python")
    assert np.allclose(py.data, read_spm(spm).data)  # auto backend

    if not HAS_NATIVE:
        return  # native extension not built in this environment
    rs = read_spm(spm, backend="rust")
    assert np.allclose(py.data, rs.data)
    assert py.data.dtype == rs.data.dtype == np.float64
    assert np.isclose(py.nm_per_lsb, rs.nm_per_lsb)
    assert py.scan_size_nm == rs.scan_size_nm
    assert py.channel_name == rs.channel_name
    assert py.metadata.get("Scan Size") == rs.metadata.get("Scan Size")
    assert [c.name for c in py.available_channels] == [c.name for c in rs.available_channels]


def make_scan_size_spm(path: Path, cols: int, rows: int, scan_list, image_list) -> None:
    """Write a 2-byte Height file whose header carries the given extra lines.

    ``scan_list`` / ``image_list`` are header lines (without the leading
    backslash) for the ``Ciao scan list`` and the image block.
    """
    raw = np.arange(cols * rows, dtype=np.int16).reshape(rows, cols)
    data_bytes = raw.astype("<i2").tobytes()
    header = "\n".join(
        [
            r"\*File list",
            r"\Version: 0x09300201",
            r"\*Ciao scan list",
            *("\\" + line for line in scan_list),
            r"\@Sens. Zsens: V 45.0 nm/V",
            r"\*Ciao image list",
            r"\Data offset: %d" % DATA_OFFSET,
            r"\Data length: %d" % len(data_bytes),
            r"\Samps/line: %d" % cols,
            r"\Number of lines: %d" % rows,
            r"\Bytes/pixel: 2",
            *("\\" + line for line in image_list),
            r'\@2:Image Data: S [Height] "Height"',
            r"\@2:Z scale: V [Sens. Zsens] (%s V/LSB) %s V" % (V_PER_LSB, Z_SCALE_V),
            r"\*File list end",
            "",
        ]
    ).encode("ascii")
    path.write_bytes(header + b"\x00" * (DATA_OFFSET - len(header)) + data_bytes)


def _read_all_backends(path: Path) -> list:
    from afm_files.parser import HAS_NATIVE

    return [read_spm(path, backend=b) for b in ["python"] + (["rust"] if HAS_NATIVE else [])]


# (scan list lines, image block lines, cols, rows, expected (x, y) in nm)
SCAN_SIZE_CASES = {
    # Square scan saved with fewer lines than samples: pixels are not square,
    # the rows still span the full 1 um.
    "square_scan_rectangular_pixels": (
        ["Scan Size: 1000 nm", "Slow Axis Size: 1000 nm", "Lines: 4", "Aspect Ratio: 1:1"],
        ["Scan Size: 1 1 ~m", "Aspect Ratio: 1:1"],
        8, 4, (1000.0, 1000.0),
    ),
    # Second number of the pair names y.
    "pair_gives_y": (
        ["Scan Size: 1000 nm", "Aspect Ratio: 1:1"],
        ["Scan Size: 1 0.25 ~m", "Aspect Ratio: 1:1"],
        8, 8, (1000.0, 250.0),
    ),
    # Slow Axis Size overrides the pair.
    "slow_axis_size": (
        ["Scan Size: 1000 nm", "Slow Axis Size: 400 nm", "Aspect Ratio: 1:1"],
        ["Scan Size: 1 1 ~m", "Aspect Ratio: 1:1"],
        8, 8, (1000.0, 400.0),
    ),
    # "Capture Now" on a 2:1 scan set up for 8 lines, saved after 4.
    "partly_captured": (
        ["Scan Size: 1000 nm", "Slow Axis Size: 500 nm", "Lines: 8", "Aspect Ratio: 2:1"],
        ["Scan Size: 1 0.5 ~m", "Aspect Ratio: 2:1"],
        8, 4, (1000.0, 250.0),
    ),
    # Some writers spell the key "Scan size".
    "lowercase_key": (
        ["Scan size: 4000 nm"],
        [],
        4, 4, (4000.0, 4000.0),
    ),
}


def test_scan_size_x_and_y(tmpdir):
    for name, (scan_list, image_list, cols, rows, (ex, ey)) in SCAN_SIZE_CASES.items():
        spm = Path(tmpdir) / f"{name}.spm"
        make_scan_size_spm(spm, cols, rows, scan_list, image_list)
        for img in _read_all_backends(spm):
            assert np.isclose(img.scan_size_nm, ex), name
            assert np.isclose(img.scan_size_y_nm, ey), name
            assert np.allclose(img.scan_size_xy_nm, (ex, ey)), name
            assert np.isclose(img.pixel_size_nm, ex / cols), name
            assert np.isclose(img.pixel_size_y_nm, ey / rows), name
            assert np.allclose(img.extent_nm, (0.0, ex, 0.0, ey)), name


def test_xyz_uses_pixel_size_per_axis(tmpdir):
    spm = Path(tmpdir) / "rect.spm"
    scan_list, image_list, cols, rows, (ex, ey) = SCAN_SIZE_CASES[
        "square_scan_rectangular_pixels"
    ]
    make_scan_size_spm(spm, cols, rows, scan_list, image_list)
    xyz = np.loadtxt(convert(spm, Path(tmpdir) / "rect.xyz"))
    assert np.isclose(xyz[:, 0].max(), ex / cols * (cols - 1))
    assert np.isclose(xyz[:, 1].max(), ey / rows * (rows - 1))


def test_scan_size_y_defaults_to_square_pixels():
    """An SpmImage built without ``scan_size_y_nm`` keeps the old behaviour."""
    img = afm_files.SpmImage(
        data=np.zeros((2, 4)),
        metadata={},
        scan_size_nm=400.0,
        samps_per_line=4,
        number_of_lines=2,
        channel_name="Height",
        channel_index=0,
        available_channels=[],
        nm_per_lsb=1.0,
    )
    assert img.pixel_size_y_nm == img.pixel_size_nm == 100.0
    assert img.extent_nm == (0.0, 400.0, 0.0, 200.0)


def test_list_channels(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    channels = afm_files.list_channels(spm)
    assert len(channels) == 1
    assert channels[0].name == "Height"
    assert channels[0].direction == "Retrace"

    # Selecting by name substring works too.
    img = read_spm(spm, channel="Height")
    assert img.channel_index == 0


def test_missing_marker_raises(tmpdir):
    bad = Path(tmpdir) / "bad.spm"
    bad.write_bytes(b"not an spm file")
    try:
        read_spm(bad)
    except ValueError as exc:
        assert "File list end" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def test_flatten_removes_line_tilt():
    # Each row is a pure ramp; order-1 flatten should zero it out.
    ramp = np.tile(np.linspace(0, 10, 8), (5, 1)) + 3.0
    flat = filters.flatten_polynomial(ramp, order=1)
    assert np.allclose(flat, 0.0, atol=1e-9)


def test_flatten_order_zero_subtracts_line_mean():
    data = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    flat = filters.flatten_polynomial(data, order=0)
    assert np.allclose(flat.mean(axis=1), 0.0)


def test_gaussian_smooth_preserves_constant():
    const = np.full((6, 6), 5.0)
    assert np.allclose(filters.gaussian_smooth(const, sigma=1.5), 5.0)


def test_plane_subtract():
    ys, xs = np.mgrid[0:5, 0:5]
    plane = 2.0 + 0.5 * xs + 1.5 * ys
    assert np.allclose(filters.plane_subtract(plane), 0.0, atol=1e-9)


def test_pipeline_and_specs():
    data = np.tile(np.linspace(0, 10, 8), (5, 1)) + 3.0
    pipe = filters.Pipeline(["flatten:1", "offset"])
    out = pipe(data)
    assert np.isclose(out.min(), 0.0)
    # parse_filter_spec round-trips
    func, kwargs = filters.parse_filter_spec("smooth:2.5")
    assert kwargs == {"sigma": 2.5}


# --------------------------------------------------------------------------- #
# Rolling ball (ports of the Rust analysis.rs tests)
# --------------------------------------------------------------------------- #
def test_rolling_ball_noop_for_zero_radius():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.array_equal(filters.rolling_ball_subtract(data, radius=0), data)


def test_rolling_ball_flattens_constant():
    const = np.full((32, 32), 5.0)
    out = filters.rolling_ball_subtract(const, radius=8)
    assert np.all(np.abs(out) < 1e-3)


def test_rolling_ball_keeps_narrow_feature_removes_background():
    data = np.zeros((40, 40))
    data[19:21, 19:21] = 10.0  # tiny bump, much smaller than the ball
    out = filters.rolling_ball_subtract(data, radius=10)
    assert out[20, 20] > 5.0            # feature preserved
    assert abs(out[0, 0]) < 1.0         # background removed
    assert np.all(np.isfinite(out)) and np.all(out >= -1e-3)


def test_rolling_ball_backends_agree():
    from afm_files.parser import HAS_NATIVE

    if not HAS_NATIVE:
        return
    rng = np.arange(48 * 48, dtype=np.float64).reshape(48, 48) % 7
    ys, xs = np.mgrid[0:48, 0:48]
    data = rng + 0.05 * xs + 0.03 * ys  # texture + gentle background
    data[24, 24] += 20.0

    native = filters.rolling_ball_subtract(data, radius=12)
    filters._core, saved = None, filters._core  # force the pure-Python path
    try:
        pure = filters.rolling_ball_subtract(data, radius=12)
    finally:
        filters._core = saved
    assert np.allclose(native, pure, atol=1e-6)


def test_rolling_ball_background_complement():
    data = np.zeros((30, 30))
    data[14:16, 14:16] = 8.0
    bg = filters.rolling_ball_background(data, radius=9)
    sub = filters.rolling_ball_subtract(data, radius=9)
    assert np.allclose(bg + sub, data, atol=1e-9)


# --------------------------------------------------------------------------- #
# Colormaps
# --------------------------------------------------------------------------- #
def test_colormaps_shape_and_endpoints():
    ramp = np.linspace(0, 1, 16).reshape(4, 4)
    for name in ("afmhot", "gray", "viridis"):
        rgb = colormaps.apply_colormap(ramp, name)
        assert rgb.shape == (4, 4, 3)
        assert rgb.dtype == np.uint8
    gray = colormaps.apply_colormap(np.array([[0.0, 1.0]]), "gray")
    assert gray[0, 0, 0] == 0 and gray[0, 1, 0] == 255


# --------------------------------------------------------------------------- #
# Converters
# --------------------------------------------------------------------------- #
def test_convert_csv_and_npy_roundtrip(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    img = read_spm(spm)

    csv_path = convert(spm, Path(tmpdir) / "out.csv")
    loaded = np.loadtxt(csv_path, delimiter=",")
    assert np.allclose(loaded, img.data, rtol=1e-5)

    npy_path = convert(img, Path(tmpdir) / "out.npy")
    assert np.array_equal(np.load(npy_path), img.data)


def test_convert_png(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    png = convert(spm, Path(tmpdir) / "out.png", filters=["flatten:2"])
    assert png.exists() and png.stat().st_size > 0


def test_convert_xyz_and_json(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    xyz = convert(spm, Path(tmpdir) / "out.xyz")
    rows = np.loadtxt(xyz)
    assert rows.shape == (SAMPS * LINES, 3)

    meta = convert(spm, Path(tmpdir) / "out.json")
    assert meta.exists()


def test_batch_convert(tmpdir):
    files = []
    for i in range(3):
        p = Path(tmpdir) / f"scan{i}.spm"
        make_synthetic_spm(p)
        files.append(p)
    out = batch_convert(files, Path(tmpdir) / "csvs", "csv")
    assert len(out) == 3
    assert all(p.exists() for p in out)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_single_file(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    dst = Path(tmpdir) / "cli.csv"
    rc = cli_main([str(spm), "-o", str(dst), "--flatten", "2"])
    assert rc == 0 and dst.exists()


def test_cli_batch_to_dir(tmpdir):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    outdir = Path(tmpdir) / "outdir"
    rc = cli_main([str(spm), "-o", str(outdir), "-f", "png", "--colormap", "viridis"])
    assert rc == 0 and (outdir / "scan.png").exists()


def test_cli_list_channels(tmpdir, capsys=None):
    spm = Path(tmpdir) / "scan.spm"
    make_synthetic_spm(spm)
    rc = cli_main([str(spm), "--list-channels"])
    assert rc == 0


# --------------------------------------------------------------------------- #
# Standalone runner (no pytest required)
# --------------------------------------------------------------------------- #
def _run_all():
    import inspect
    import traceback

    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        n_params = len(inspect.signature(fn).parameters)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                fn(tmp) if n_params else fn()
        except Exception:  # pragma: no cover
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
        else:
            print(f"ok   {name}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all() else 0)
