"""Parser for Bruker Nanoscope SPM/AFM files.

Parsing is done by the compiled Rust core (``afm_files._core``, built with
maturin) when it is available; this module also carries a complete pure-Python +
NumPy fallback so the package works even without the native extension.  Both
paths are ports of the Rust parser in https://github.com/IndigoCarmine/afm_files.

A Nanoscope file is a text header (key/value lines beginning with ``\\``)
followed by raw binary image data.  The header is terminated by the marker
``\\*File list end``.  Each measured channel is described by a
``\\*Ciao image list`` block that carries the byte offset/length of its data
and the scale factors needed to turn raw LSB integers into nanometers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    from . import _core  # native Rust parser (built by maturin)
except ImportError:  # pragma: no cover - pure-Python fallback path
    _core = None

#: True when the compiled ``_core`` extension is available.
HAS_NATIVE = _core is not None

__all__ = ["ChannelInfo", "SpmImage", "read_spm", "list_channels", "HAS_NATIVE"]

_FILE_LIST_END = b"\\*File list end"


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChannelInfo:
    """Description of one image channel found in the file."""

    index: int
    name: str
    direction: str

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        d = f" ({self.direction})" if self.direction else ""
        return f"[{self.index}] {self.name}{d}"


@dataclass
class SpmImage:
    """A single decoded AFM image channel, scaled to nanometers.

    Attributes
    ----------
    data:
        2-D ``float64`` array of shape ``(number_of_lines, samps_per_line)``
        in nanometers.  Row 0 is the top of the image.
    metadata:
        Flattened header key/value dictionary.  Keys are stored both plainly
        (first occurrence wins) and namespaced as ``"<section>::<key>"``.
    scan_size_nm:
        Physical width of the scan in nanometers.
    channel_name / channel_index:
        Which channel was decoded.
    available_channels:
        Every channel that could have been decoded from this file.
    nm_per_lsb:
        The LSB -> nm scale factor that was applied.
    source_path:
        Path the image was read from, if any.
    """

    data: np.ndarray
    metadata: dict
    scan_size_nm: float
    samps_per_line: int
    number_of_lines: int
    channel_name: str
    channel_index: int
    available_channels: list
    nm_per_lsb: float
    source_path: Optional[Path] = None

    # -- convenience ------------------------------------------------------- #
    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def pixel_size_nm(self) -> float:
        """Size of one pixel in nm (assumes square pixels)."""
        return self.scan_size_nm / self.samps_per_line if self.samps_per_line else 0.0

    @property
    def extent_nm(self) -> tuple:
        """``(x0, x1, y0, y1)`` extent in nm, suitable for ``imshow``."""
        px = self.pixel_size_nm
        return (0.0, self.scan_size_nm, 0.0, px * self.number_of_lines)

    def summary(self) -> str:
        return (
            f"SpmImage(channel={self.channel_name!r}, "
            f"shape={self.number_of_lines}x{self.samps_per_line}, "
            f"scan_size={self.scan_size_nm:g} nm, "
            f"z=[{float(self.data.min()):.3g}, {float(self.data.max()):.3g}] nm)"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


# --------------------------------------------------------------------------- #
# Low-level header parsing helpers (ports of the Rust functions)
# --------------------------------------------------------------------------- #
def _parse_header_and_blocks(header_text: str):
    """Return ``(metadata, image_blocks)`` from the decoded header text."""
    metadata: dict = {}
    blocks: list = []
    section = "root"
    current: Optional[dict] = None

    for raw in header_text.split("\n"):
        line = raw.rstrip("\r").strip()
        if not line.startswith("\\"):
            continue

        if line.startswith("\\*"):
            section = line[2:].strip()
            if section == "Ciao image list":
                if current is not None:
                    blocks.append(current)
                current = {}
            continue

        body = line.lstrip("\\")
        if ": " in body:
            key, value = body.split(": ", 1)
        elif ":" in body:
            key, value = body.split(":", 1)
        else:
            continue
        key = key.strip()
        value = value.strip()

        metadata[f"{section}::{key}"] = value
        metadata.setdefault(key, value)
        if current is not None:
            current[key] = value

    if current is not None:
        blocks.append(current)

    return metadata, blocks


def _parse_int_field(value: Optional[str], field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing field: {field_name}")
    digits = ""
    for ch in value.strip():
        if ch.isdigit() and ch.isascii():
            digits += ch
        else:
            break
    if not digits:
        raise ValueError(f"Failed to parse numeric field {field_name}: {value}")
    return int(digits)


def _parse_float_before_unit(text: str, unit: str) -> Optional[float]:
    idx = text.find(unit)
    if idx < 0:
        return None
    prefix = text[:idx].rstrip()
    tokens = prefix.split()
    if not tokens:
        return None
    token = tokens[-1].strip("(),")
    try:
        return float(token)
    except ValueError:
        return None


def _parse_sensitivity_name(scale_text: str) -> Optional[str]:
    marker = "[Sens."
    pos = scale_text.find(marker)
    if pos < 0:
        return None
    after = scale_text[pos + len(marker):]
    end = after.find("]")
    if end < 0:
        return None
    return after[:end].strip()


def _parse_last_float(text: str) -> Optional[float]:
    last = None
    for tok in text.split():
        try:
            last = float(tok.strip("(),"))
        except ValueError:
            continue
    return last


def _find_field(block: dict, name: str) -> Optional[str]:
    """Value whose key is ``name`` or ends with ``":name"`` (handles ``@2:`` prefixes)."""
    suffix = ":" + name
    for k, v in block.items():
        if k == name or k.endswith(suffix):
            return v
    return None


def _parse_nm_per_lsb(block: dict, metadata: dict, bytes_per_pixel: int) -> float:
    z_scale_text = _find_field(block, "Z scale")
    if z_scale_text is None:
        raise ValueError("Missing Z scale field in image block")

    nm_per_lsb = _parse_float_before_unit(z_scale_text, "nm/LSB")
    if nm_per_lsb is not None:
        return nm_per_lsb

    v_per_lsb = _parse_float_before_unit(z_scale_text, "V/LSB")
    if v_per_lsb is None:
        raise ValueError(f"Could not parse V/LSB from Z scale: {z_scale_text}")

    sens_name = _parse_sensitivity_name(z_scale_text)
    if sens_name is None:
        raise ValueError(
            f"Could not parse sensitivity reference from Z scale: {z_scale_text}"
        )
    sens_key = f"@Sens. {sens_name}"
    sens_value = metadata.get(sens_key)
    if sens_value is None:
        raise ValueError(f"Missing sensitivity field: {sens_key}")
    nm_per_v = _parse_float_before_unit(sens_value, "nm/V")
    if nm_per_v is None:
        raise ValueError(
            f"Could not parse nm/V from sensitivity field {sens_key}: {sens_value}"
        )

    # Bruker convention: the trailing full-scale voltage spans the whole unsigned
    # integer range (256**bytes_per_pixel steps).  min() picks whichever of the
    # two consistent readings is smaller (see upstream comment).
    z_scale_v = _parse_last_float(z_scale_text)
    if z_scale_v is None:
        raise ValueError(f"Could not parse full-scale V from Z scale: {z_scale_text}")
    max_int = float(1 << (bytes_per_pixel * 8))
    effective_v_per_lsb = min(v_per_lsb, z_scale_v / max_int)
    return effective_v_per_lsb * nm_per_v


_REQUIRED_FIELDS = (
    "Data offset",
    "Data length",
    "Samps/line",
    "Number of lines",
    "Bytes/pixel",
)


def _has_required_fields(block: dict) -> bool:
    return all(f in block for f in _REQUIRED_FIELDS)


def _extract_channel_name(image_data_value: str) -> str:
    # Format: 'S [Height] "Height"' -> text between the first pair of quotes.
    parts = image_data_value.split('"')
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return image_data_value


def _channel_name_of(block: dict, index: int) -> str:
    val = _find_field(block, "Image Data")
    if val is not None:
        return _extract_channel_name(val)
    return f"Channel {index}"


def _block_is_height(block: dict) -> bool:
    val = _find_field(block, "Image Data")
    return val is not None and "Height" in val


def _list_channels(blocks: list) -> list:
    channels = []
    for i, b in enumerate(blocks):
        if not _has_required_fields(b):
            continue
        channels.append(
            ChannelInfo(
                index=i,
                name=_channel_name_of(b, i),
                direction=b.get("Line Direction", ""),
            )
        )
    return channels


def _choose_block(blocks: list, channel: Union[int, str, None]):
    """Return ``(index, block)`` for the requested (or auto-selected) channel."""
    valid = [(i, b) for i, b in enumerate(blocks) if _has_required_fields(b)]

    if isinstance(channel, str):
        needle = channel.lower()
        for i, b in valid:
            if needle in _channel_name_of(b, i).lower():
                return i, b
        names = ", ".join(_channel_name_of(b, i) for i, b in valid)
        raise ValueError(f"No channel matching {channel!r}. Available: {names}")

    if channel is not None:  # integer index
        if channel < 0 or channel >= len(blocks):
            raise ValueError(f"Channel index {channel} out of range")
        block = blocks[channel]
        if not _has_required_fields(block):
            raise ValueError(f"Channel {channel} is missing required fields")
        return channel, block

    # Auto: prefer the last Height block (Retrace usually follows Trace).
    for i, b in reversed(valid):
        if _block_is_height(b):
            return i, b

    if valid:
        return valid[0]
    raise ValueError("No image block with required fields found")


def _decode_image(data_bytes, samps_per_line, number_of_lines,
                  bytes_per_pixel) -> np.ndarray:
    n_pixels = samps_per_line * number_of_lines
    expected = n_pixels * bytes_per_pixel
    if len(data_bytes) < expected:
        raise ValueError(
            f"Not enough image bytes: expected {expected}, got {len(data_bytes)}"
        )

    # Nanoscope image data is little-endian.  The header's ``Start context``
    # (e.g. ``OL1BIG``/``OL2BIG``) is not a byte-order flag: files carrying it
    # are little-endian too, and Gwyddion's reader ignores it for this purpose.
    if bytes_per_pixel == 2:
        dtype = np.dtype("<i2")
    elif bytes_per_pixel == 4:
        dtype = np.dtype("<i4")
    else:
        raise ValueError(f"Unsupported Bytes/pixel value: {bytes_per_pixel}")

    flat = np.frombuffer(data_bytes[:expected], dtype=dtype).astype(np.float64)
    image = flat.reshape(number_of_lines, samps_per_line)
    # Nanoscope stores lines bottom-to-top; flip so row 0 is the image top.
    return image[::-1, :].copy()


def _parse_scan_size_nm(metadata: dict) -> float:
    raw = metadata.get("Scan Size")
    if raw is None:
        return 1000.0
    tokens = raw.split()
    if not tokens:
        return 1000.0
    try:
        value = float(tokens[0])
    except ValueError:
        return 1000.0
    if ("~m" in raw) or ("um" in raw) or ("µm" in raw):
        return value * 1000.0
    return value


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _read_header(data: bytes):
    pos = data.find(_FILE_LIST_END)
    if pos < 0:
        raise ValueError("Could not find \\*File list end marker in file")
    header_end = pos + len(_FILE_LIST_END)
    header_text = data[:header_end].decode("utf-8", "replace")
    return _parse_header_and_blocks(header_text)


def list_channels(path) -> list:
    """List the decodable channels in an SPM file without decoding image data."""
    data = Path(path).read_bytes()
    _, blocks = _read_header(data)
    return _list_channels(blocks)


def _read_spm_python(path: Path, channel: Union[int, str, None]) -> SpmImage:
    """Pure-Python read path (no native extension)."""
    data = path.read_bytes()
    metadata, blocks = _read_header(data)
    available = _list_channels(blocks)
    index, block = _choose_block(blocks, channel)
    channel_name = next(
        (c.name for c in available if c.index == index), f"Channel {index}"
    )

    data_offset = _parse_int_field(block.get("Data offset"), "Data offset")
    data_length = _parse_int_field(block.get("Data length"), "Data length")
    samps_per_line = _parse_int_field(block.get("Samps/line"), "Samps/line")
    number_of_lines = _parse_int_field(block.get("Number of lines"), "Number of lines")
    bytes_per_pixel = _parse_int_field(block.get("Bytes/pixel"), "Bytes/pixel")

    end_offset = data_offset + data_length
    if end_offset > len(data):
        raise ValueError(
            f"Data slice out of range: offset={data_offset}, "
            f"length={data_length}, file_size={len(data)}"
        )

    image = _decode_image(
        data[data_offset:end_offset],
        samps_per_line,
        number_of_lines,
        bytes_per_pixel,
    )

    nm_per_lsb = _parse_nm_per_lsb(block, metadata, bytes_per_pixel)
    image *= nm_per_lsb

    return SpmImage(
        data=image,
        metadata=metadata,
        scan_size_nm=_parse_scan_size_nm(metadata),
        samps_per_line=samps_per_line,
        number_of_lines=number_of_lines,
        channel_name=channel_name,
        channel_index=index,
        available_channels=available,
        nm_per_lsb=nm_per_lsb,
        source_path=path,
    )


def _read_spm_rust(path: Path, channel: Union[int, str, None]) -> SpmImage:
    """Native read path via the compiled ``_core`` extension."""
    channel_index = channel if isinstance(channel, int) else None
    channel_name = channel if isinstance(channel, str) else None
    res = _core.load_spm(str(path), channel_index=channel_index, channel_name=channel_name)

    rows, cols = res["rows"], res["cols"]
    # ``data`` is row-major float64 LE bytes; copy because frombuffer is read-only.
    image = np.frombuffer(res["data"], dtype="<f8").reshape(rows, cols).copy()
    available = [ChannelInfo(i, n, d) for (i, n, d) in res["channels"]]

    return SpmImage(
        data=image,
        metadata=dict(res["metadata"]),
        scan_size_nm=res["scan_size_nm"],
        samps_per_line=cols,
        number_of_lines=rows,
        channel_name=res["channel_name"],
        channel_index=res["channel_index"],
        available_channels=available,
        nm_per_lsb=res["nm_per_lsb"],
        source_path=path,
    )


def read_spm(
    path,
    *,
    channel: Union[int, str, None] = None,
    flatten: Optional[int] = None,
    smooth_sigma: float = 0.0,
    backend: str = "auto",
) -> SpmImage:
    """Read one channel of a Bruker Nanoscope SPM file.

    Parameters
    ----------
    path:
        Path to the ``.spm`` (or ``.000`` etc.) file.
    channel:
        ``None`` (default) auto-selects the last Height channel.  Pass an
        integer block index or a channel-name substring (e.g. ``"Height"``,
        ``"Phase"``) to pick a specific channel.
    flatten:
        If given (0-3), apply per-line polynomial flattening of that order.
    smooth_sigma:
        If > 0, apply a Gaussian smoothing filter of this sigma (pixels).
    backend:
        ``"auto"`` (default) uses the native Rust core when it is installed and
        falls back to pure Python otherwise; ``"rust"`` forces the native core
        (raising if it is unavailable); ``"python"`` forces the pure-Python
        parser.

    Returns
    -------
    SpmImage
    """
    if backend not in ("auto", "rust", "python"):
        raise ValueError(f"Unknown backend {backend!r}; use auto/rust/python")
    path = Path(path)

    if backend == "rust" and _core is None:
        raise RuntimeError(
            "The native _core extension is not built. Build it with "
            "`maturin develop` / `pip install .`, or use backend='python'."
        )
    use_rust = backend == "rust" or (backend == "auto" and _core is not None)
    img = _read_spm_rust(path, channel) if use_rust else _read_spm_python(path, channel)

    # Optional in-line processing (mirrors the upstream loader).  Imported here
    # to avoid a circular import at module load time.
    if flatten is not None or smooth_sigma > 0:
        from . import filters

        if flatten is not None:
            img.data = filters.flatten_polynomial(img.data, order=flatten)
        if smooth_sigma > 0:
            img.data = filters.gaussian_smooth(img.data, sigma=smooth_sigma)
    return img
