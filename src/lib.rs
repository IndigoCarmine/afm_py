//! Native core for the `afm_files` Python package.
//!
//! This is a pyo3 port of the SPM parser from
//! https://github.com/IndigoCarmine/afm_files (the `parser.rs` variant, which
//! includes bytes-per-pixel-aware Z-scale handling and channel selection).
//!
//! It exposes a single function, `load_spm`, that parses a Bruker Nanoscope
//! file, decodes one channel, scales it to nanometers, and returns the raw
//! `float64` little-endian bytes together with the header metadata.  The Python
//! layer turns those bytes into a NumPy array, so this crate does not depend on
//! the rust-numpy bridge.

use std::collections::HashMap;
use std::fs;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

type Block = HashMap<String, String>;

// --------------------------------------------------------------------------- //
// Header parsing
// --------------------------------------------------------------------------- //
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn parse_header_and_blocks(header_text: &str) -> (HashMap<String, String>, Vec<Block>) {
    let mut metadata = HashMap::new();
    let mut blocks: Vec<Block> = Vec::new();
    let mut section = String::from("root");
    let mut current: Option<Block> = None;

    for raw_line in header_text.lines() {
        let line = raw_line.trim_end_matches('\r').trim();
        if !line.starts_with('\\') {
            continue;
        }

        if line.starts_with("\\*") {
            section = line.trim_start_matches("\\*").trim().to_string();
            if section == "Ciao image list" {
                if let Some(block) = current.take() {
                    blocks.push(block);
                }
                current = Some(Block::new());
            }
            continue;
        }

        let body = line.trim_start_matches('\\');
        let parsed = body.split_once(": ").or_else(|| body.split_once(':'));
        if let Some((key, value)) = parsed {
            let key = key.trim().to_string();
            let value = value.trim().to_string();
            metadata.insert(format!("{section}::{key}"), value.clone());
            metadata.entry(key.clone()).or_insert_with(|| value.clone());
            if let Some(block) = current.as_mut() {
                block.insert(key, value);
            }
        }
    }

    if let Some(block) = current.take() {
        blocks.push(block);
    }

    (metadata, blocks)
}

fn parse_usize_field(value: Option<&String>, name: &str) -> Result<usize, String> {
    let raw = value.ok_or_else(|| format!("Missing field: {name}"))?;
    let digits: String = raw.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Err(format!("Failed to parse numeric field {name}: {raw}"));
    }
    digits
        .parse::<usize>()
        .map_err(|e| format!("Failed to parse field {name}: {e}"))
}

fn parse_float_before_unit(text: &str, unit: &str) -> Option<f64> {
    let idx = text.find(unit)?;
    let prefix = text[..idx].trim_end();
    let token = prefix
        .split_whitespace()
        .last()
        .map(|s| s.trim_matches(|c: char| c == '(' || c == ')' || c == ','))?;
    token.parse::<f64>().ok()
}

fn parse_sensitivity_name(scale_text: &str) -> Option<String> {
    let pos = scale_text.find("[Sens.")?;
    let after = &scale_text[(pos + "[Sens.".len())..];
    let end = after.find(']')?;
    Some(after[..end].trim().to_string())
}

fn parse_last_float(text: &str) -> Option<f64> {
    text.split_whitespace()
        .filter_map(|t| {
            t.trim_matches(|c: char| c == '(' || c == ')' || c == ',')
                .parse::<f64>()
                .ok()
        })
        .last()
}

fn find_field<'a>(block: &'a Block, name: &str) -> Option<&'a String> {
    let suffix = format!(":{name}");
    block
        .iter()
        .find_map(|(k, v)| (k == name || k.ends_with(&suffix)).then_some(v))
}

fn parse_nm_per_lsb(
    block: &Block,
    metadata: &HashMap<String, String>,
    bytes_per_pixel: usize,
) -> Result<f64, String> {
    let z_scale_text =
        find_field(block, "Z scale").ok_or_else(|| "Missing Z scale field in image block".to_string())?;

    if let Some(nm_per_lsb) = parse_float_before_unit(z_scale_text, "nm/LSB") {
        return Ok(nm_per_lsb);
    }

    let v_per_lsb = parse_float_before_unit(z_scale_text, "V/LSB")
        .ok_or_else(|| format!("Could not parse V/LSB from Z scale: {z_scale_text}"))?;

    let sens_name = parse_sensitivity_name(z_scale_text)
        .ok_or_else(|| format!("Could not parse sensitivity reference from Z scale: {z_scale_text}"))?;
    let sens_key = format!("@Sens. {sens_name}");
    let sens_value = metadata
        .get(&sens_key)
        .ok_or_else(|| format!("Missing sensitivity field: {sens_key}"))?;
    let nm_per_v = parse_float_before_unit(sens_value, "nm/V")
        .ok_or_else(|| format!("Could not parse nm/V from sensitivity field {sens_key}: {sens_value}"))?;

    // Bruker full-scale-voltage convention (see upstream comment).
    let z_scale_v = parse_last_float(z_scale_text)
        .ok_or_else(|| format!("Could not parse full-scale V from Z scale: {z_scale_text}"))?;
    let max_int = (1u64 << (bytes_per_pixel * 8)) as f64;
    let effective_v_per_lsb = v_per_lsb.min(z_scale_v / max_int);
    Ok(effective_v_per_lsb * nm_per_v)
}

const REQUIRED_FIELDS: [&str; 5] = [
    "Data offset",
    "Data length",
    "Samps/line",
    "Number of lines",
    "Bytes/pixel",
];

fn has_required_fields(b: &Block) -> bool {
    REQUIRED_FIELDS.iter().all(|f| b.contains_key(*f))
}

fn extract_channel_name(value: &str) -> String {
    let parts: Vec<&str> = value.split('"').collect();
    if parts.len() >= 2 && !parts[1].is_empty() {
        parts[1].to_string()
    } else {
        value.to_string()
    }
}

fn channel_name_of(block: &Block, index: usize) -> String {
    find_field(block, "Image Data")
        .map(|v| extract_channel_name(v))
        .unwrap_or_else(|| format!("Channel {index}"))
}

fn block_is_height(block: &Block) -> bool {
    find_field(block, "Image Data")
        .map(|v| v.contains("Height"))
        .unwrap_or(false)
}

fn list_channels(blocks: &[Block]) -> Vec<(usize, String, String)> {
    blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| has_required_fields(b))
        .map(|(i, b)| {
            (
                i,
                channel_name_of(b, i),
                b.get("Line Direction").cloned().unwrap_or_default(),
            )
        })
        .collect()
}

fn choose_block<'a>(
    blocks: &'a [Block],
    channel_index: Option<usize>,
    channel_name: Option<&str>,
) -> Result<(usize, &'a Block), String> {
    let valid: Vec<(usize, &Block)> = blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| has_required_fields(b))
        .map(|(i, b)| (i, b))
        .collect();

    if let Some(name) = channel_name {
        let needle = name.to_lowercase();
        for (i, b) in &valid {
            if channel_name_of(b, *i).to_lowercase().contains(&needle) {
                return Ok((*i, *b));
            }
        }
        let names: Vec<String> = valid.iter().map(|(i, b)| channel_name_of(b, *i)).collect();
        return Err(format!("No channel matching {name:?}. Available: {}", names.join(", ")));
    }

    if let Some(idx) = channel_index {
        let block = blocks
            .get(idx)
            .ok_or_else(|| format!("Channel index {idx} out of range"))?;
        if !has_required_fields(block) {
            return Err(format!("Channel {idx} is missing required fields"));
        }
        return Ok((idx, block));
    }

    if let Some((i, b)) = valid.iter().rev().find(|(_, b)| block_is_height(b)) {
        return Ok((*i, *b));
    }
    valid
        .first()
        .copied()
        .ok_or_else(|| "No image block with required fields found".to_string())
}

/// Decode raw bytes into a flat row-major `Vec<f64>` (row 0 = top of image).
fn decode_image(
    data_bytes: &[u8],
    samps_per_line: usize,
    number_of_lines: usize,
    bytes_per_pixel: usize,
) -> Result<Vec<f64>, String> {
    let n_pixels = samps_per_line
        .checked_mul(number_of_lines)
        .ok_or_else(|| "Image dimensions are too large".to_string())?;
    let expected = n_pixels
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| "Image byte count overflow".to_string())?;
    if data_bytes.len() < expected {
        return Err(format!(
            "Not enough image bytes: expected {expected}, got {}",
            data_bytes.len()
        ));
    }

    // Nanoscope image data is little-endian. The header's `Start context`
    // (e.g. `OL1BIG`/`OL2BIG`) is not a byte-order flag: files carrying it are
    // little-endian too, and Gwyddion's reader ignores it for this purpose.
    let mut flat = Vec::with_capacity(n_pixels);
    match bytes_per_pixel {
        2 => {
            for chunk in data_bytes[..expected].chunks_exact(2) {
                flat.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f64);
            }
        }
        4 => {
            for chunk in data_bytes[..expected].chunks_exact(4) {
                flat.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64);
            }
        }
        other => return Err(format!("Unsupported Bytes/pixel value: {other}")),
    }

    // Lines are stored bottom-to-top; reverse the row order.
    let mut out = vec![0.0_f64; n_pixels];
    for (row_idx, row) in flat.chunks_exact(samps_per_line).enumerate() {
        let dst = number_of_lines - 1 - row_idx;
        out[dst * samps_per_line..(dst + 1) * samps_per_line].copy_from_slice(row);
    }
    Ok(out)
}

fn parse_scan_size_nm(metadata: &HashMap<String, String>) -> f64 {
    let raw = match metadata.get("Scan Size") {
        Some(v) => v,
        None => return 1000.0,
    };
    let value = match raw.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()) {
        Some(v) => v,
        None => return 1000.0,
    };
    if raw.contains("~m") || raw.contains("um") || raw.contains("µm") {
        value * 1000.0
    } else {
        value
    }
}

// --------------------------------------------------------------------------- //
// Rolling-ball background subtraction (ported from analysis.rs)
// --------------------------------------------------------------------------- //
/// Offsets (di, dj) within radius `r` and the ball's height there (sqrt(r^2 - d^2)).
fn build_ball(r: usize) -> Vec<(i32, i32, f64)> {
    let ri = r as i32;
    let r2 = (r * r) as f64;
    let mut ball = Vec::new();
    for dj in -ri..=ri {
        for di in -ri..=ri {
            let d2 = (di * di + dj * dj) as f64;
            if d2 <= r2 {
                ball.push((di, dj, (r2 - d2).sqrt()));
            }
        }
    }
    ball
}

/// One morphological pass with the symmetric ball SE: erosion (`dilate=false`)
/// computes min(g - ball); dilation (`dilate=true`) computes max(g + ball).
/// Out-of-bounds samples use edge replication.
fn ball_morph(g: &[f64], w: usize, h: usize, ball: &[(i32, i32, f64)], dilate: bool) -> Vec<f64> {
    let (wi, hi) = (w as i32, h as i32);
    let mut out = vec![0.0f64; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = if dilate { f64::NEG_INFINITY } else { f64::INFINITY };
            for &(di, dj, bh) in ball {
                let xx = (x as i32 + di).clamp(0, wi - 1) as usize;
                let yy = (y as i32 + dj).clamp(0, hi - 1) as usize;
                let s = g[yy * w + xx];
                if dilate {
                    acc = acc.max(s + bh);
                } else {
                    acc = acc.min(s - bh);
                }
            }
            out[y * w + x] = acc;
        }
    }
    out
}

/// Grayscale opening (erode then dilate) with a spherical structuring element.
fn ball_opening(g: &[f64], w: usize, h: usize, r: usize) -> Vec<f64> {
    let ball = build_ball(r);
    let eroded = ball_morph(g, w, h, &ball, false);
    ball_morph(&eroded, w, h, &ball, true)
}

/// Downsample by taking the minimum of each `k`x`k` block.
fn shrink_min(data: &[f64], w: usize, h: usize, k: usize) -> (usize, usize, Vec<f64>) {
    if k <= 1 {
        return (w, h, data.to_vec());
    }
    let sw = w.div_ceil(k);
    let sh = h.div_ceil(k);
    let mut out = vec![f64::INFINITY; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let o = (y / k) * sw + (x / k);
            out[o] = out[o].min(data[y * w + x]);
        }
    }
    (sw, sh, out)
}

/// Bilinearly enlarge an `sw`x`sh` field to `w`x`h` (corners aligned).
fn enlarge(small: &[f64], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f64> {
    if sw == w && sh == h {
        return small.to_vec();
    }
    let mut out = vec![0.0f64; w * h];
    let sx = if w > 1 { (sw - 1) as f64 / (w - 1) as f64 } else { 0.0 };
    let sy = if h > 1 { (sh - 1) as f64 / (h - 1) as f64 } else { 0.0 };
    for y in 0..h {
        let gy = y as f64 * sy;
        let y0 = gy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = gy - y0 as f64;
        for x in 0..w {
            let gx = x as f64 * sx;
            let x0 = gx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = gx - x0 as f64;
            let v00 = small[y0 * sw + x0];
            let v01 = small[y0 * sw + x1];
            let v10 = small[y1 * sw + x0];
            let v11 = small[y1 * sw + x1];
            let top = v00 + (v01 - v00) * tx;
            let bot = v10 + (v11 - v10) * tx;
            out[y * w + x] = top + (bot - top) * ty;
        }
    }
    out
}

/// Estimate the background of a 0..100 normalized field by rolling a ball of
/// `radius` px underneath it (grayscale opening), shrinking first for speed.
fn rolling_ball_background(norm: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    let shrink = if radius <= 10 {
        1
    } else if radius <= 30 {
        2
    } else if radius <= 100 {
        4
    } else {
        8
    };
    let small_r = (radius / shrink).max(1);
    let (sw, sh, small) = shrink_min(norm, w, h, shrink);
    let small_bg = ball_opening(&small, sw, sh, small_r);
    enlarge(&small_bg, sw, sh, w, h)
}

/// Rolling-ball background subtraction (Sternberg). `radius` is in pixels.
fn rolling_ball_subtract_impl(data: &[f64], w: usize, h: usize, radius: usize) -> Vec<f64> {
    if radius == 0 || w == 0 || h == 0 || data.len() < w * h {
        return data.to_vec();
    }
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(f64::EPSILON);
    const WORK: f64 = 100.0;

    let norm: Vec<f64> = data.iter().map(|&v| (v - min) / range * WORK).collect();
    let background = rolling_ball_background(&norm, w, h, radius);

    data.iter()
        .zip(background.iter())
        .map(|(&v, &bg)| v - (bg / WORK * range + min))
        .collect()
}

// --------------------------------------------------------------------------- //
// Python bindings
// --------------------------------------------------------------------------- //
fn err(msg: String) -> PyErr {
    PyValueError::new_err(msg)
}

/// Rolling-ball background subtraction on a float64 row-major image.
///
/// ``data`` is little-endian float64 bytes of length ``w * h * 8``; the result
/// is returned in the same layout.  Mirrors the desktop viewer's filter.
#[pyfunction]
fn rolling_ball_subtract(
    py: Python<'_>,
    data: &[u8],
    w: usize,
    h: usize,
    radius: usize,
) -> PyResult<Py<PyBytes>> {
    if data.len() % 8 != 0 {
        return Err(err("data length must be a multiple of 8 (float64)".to_string()));
    }
    if data.len() / 8 < w * h {
        return Err(err(format!(
            "data has {} values, expected at least {}",
            data.len() / 8,
            w * h
        )));
    }
    let values: Vec<f64> = data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let out = rolling_ball_subtract_impl(&values, w, h, radius);

    let mut raw = Vec::with_capacity(out.len() * 8);
    for v in &out {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyBytes::new(py, &raw).unbind())
}

/// Parse an SPM file and return a dict describing one channel.
///
/// Keys: ``data`` (float64 little-endian bytes, row-major, nm), ``rows``,
/// ``cols``, ``metadata`` (dict), ``channels`` (list of ``(index, name,
/// direction)``), ``channel_index``, ``channel_name``, ``scan_size_nm``,
/// ``nm_per_lsb``.
#[pyfunction]
#[pyo3(signature = (path, channel_index=None, channel_name=None))]
fn load_spm(
    py: Python<'_>,
    path: &str,
    channel_index: Option<usize>,
    channel_name: Option<String>,
) -> PyResult<Py<PyDict>> {
    let bytes = fs::read(path).map_err(|e| err(format!("Failed to read file: {e}")))?;

    let marker = b"\\*File list end";
    let marker_pos =
        find_bytes(&bytes, marker).ok_or_else(|| err("Could not find \\*File list end marker in file".to_string()))?;
    let header_end = marker_pos + marker.len();
    let header_text = String::from_utf8_lossy(&bytes[..header_end]);

    let (metadata, blocks) = parse_header_and_blocks(&header_text);
    let channels = list_channels(&blocks);
    let (index, block) =
        choose_block(&blocks, channel_index, channel_name.as_deref()).map_err(err)?;
    let chan_name = channels
        .iter()
        .find(|(i, _, _)| *i == index)
        .map(|(_, n, _)| n.clone())
        .unwrap_or_else(|| format!("Channel {index}"));

    let data_offset = parse_usize_field(block.get("Data offset"), "Data offset").map_err(err)?;
    let data_length = parse_usize_field(block.get("Data length"), "Data length").map_err(err)?;
    let samps_per_line = parse_usize_field(block.get("Samps/line"), "Samps/line").map_err(err)?;
    let number_of_lines =
        parse_usize_field(block.get("Number of lines"), "Number of lines").map_err(err)?;
    let bytes_per_pixel = parse_usize_field(block.get("Bytes/pixel"), "Bytes/pixel").map_err(err)?;

    let end_offset = data_offset
        .checked_add(data_length)
        .ok_or_else(|| err("Data offset overflow".to_string()))?;
    if end_offset > bytes.len() {
        return Err(err(format!(
            "Data slice out of range: offset={data_offset}, length={data_length}, file_size={}",
            bytes.len()
        )));
    }

    let mut flat = decode_image(
        &bytes[data_offset..end_offset],
        samps_per_line,
        number_of_lines,
        bytes_per_pixel,
    )
    .map_err(err)?;

    let nm_per_lsb = parse_nm_per_lsb(block, &metadata, bytes_per_pixel).map_err(err)?;
    for v in flat.iter_mut() {
        *v *= nm_per_lsb;
    }

    // Pack the f64 grid as little-endian bytes for zero-copy NumPy reconstruction.
    let mut raw = Vec::with_capacity(flat.len() * 8);
    for v in &flat {
        raw.extend_from_slice(&v.to_le_bytes());
    }

    let out = PyDict::new(py);
    out.set_item("data", PyBytes::new(py, &raw))?;
    out.set_item("rows", number_of_lines)?;
    out.set_item("cols", samps_per_line)?;

    let meta = PyDict::new(py);
    for (k, v) in &metadata {
        meta.set_item(k.as_str(), v.as_str())?;
    }
    out.set_item("metadata", meta)?;

    let chan_list = PyList::empty(py);
    for (i, name, dir) in &channels {
        chan_list.append((*i, name.as_str(), dir.as_str()))?;
    }
    out.set_item("channels", chan_list)?;
    out.set_item("channel_index", index)?;
    out.set_item("channel_name", chan_name.as_str())?;
    out.set_item("scan_size_nm", parse_scan_size_nm(&metadata))?;
    out.set_item("nm_per_lsb", nm_per_lsb)?;

    Ok(out.unbind())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_spm, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_ball_subtract, m)?)?;
    m.add("__doc__", "Native SPM parser and filters for afm_files.")?;
    Ok(())
}
