"""Camargo dataset parser and HDF5 loader.

Parses raw .mat files from the Camargo et al. (2021) locomotion dataset.

## Why this is non-trivial

The Camargo .mat files are NOT simple variable dumps. They contain MATLAB
*table* objects, which get saved as MCOS (MATLAB Class Object System) blobs.
scipy.io.loadmat can't deserialize MATLAB tables — it just returns a raw
byte stream called ``__function_workspace__``.

So we parse that byte stream directly:
  1. Scan for ASCII column names (e.g. "Header", "gastrocmed", "Speed")
  2. Scan for miDOUBLE arrays (MATLAB type tag = 9) — these are the columns

## Camargo directory layout

The raw dataset is organized as:
    {data_root}/
    ├── AB06/
    │   └── 20170907/               ← date folder (one per subject)
    │       ├── treadmill/
    │       │   ├── emg/            ← 7 .mat files (one per trial block)
    │       │   │   ├── AB06_treadmill_emg_01.mat
    │       │   │   └── ...
    │       │   ├── gcRight/        ← gait cycle: HeelStrike sawtooth (0–100%)
    │       │   ├── conditions/     ← treadmill speed at each time point
    │       │   └── ...             ← gon, fp, imu, markers, id, ik, jp
    │       ├── levelground/
    │       ├── ramp/
    │       └── stair/
    ├── AB07/
    └── ...

Each .mat file contains a MATLAB table with:
  - "Header" column: time vector in seconds
  - Sensor-specific columns (e.g. 11 EMG muscles, or HeelStrike/ToeOff)

Only numeric columns are extracted — string columns (e.g. "Label" in
conditions files) are skipped because the byte stream parser only finds
miDOUBLE arrays.
"""

import logging
import re
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from leaps.data.metadata import ALL_MODES, ALL_SUBJECT_IDS, SAMPLE_RATES

logger = logging.getLogger(__name__)

# ── Dataset constants ──────────────────────────────────────────────────

ALL_SUBJECTS = ALL_SUBJECT_IDS

ALL_SENSORS = [
    "conditions", "emg", "fp", "gcLeft", "gcRight",
    "gon", "id", "ik", "imu", "jp", "markers",
]

SENSOR_SAMPLE_RATES = SAMPLE_RATES

# MATLAB table metadata fields that appear after the column name list.
_TABLE_METADATA_FIELDS = frozenset({
    "useVariableNamesOriginal", "useDimensionNamesOriginal",
    "CustomProps", "VariableCustomProps", "versionSavedFrom",
    "minCompatibleVersion", "incompatibilityMsg", "VersionSavedFrom",
    "Description", "VariableNamesOriginal", "DimensionNames",
    "DimensionNamesOriginal", "UserData", "VariableDescriptions",
    "VariableUnits", "VariableContinuity", "Row", "Variables",
    "data", "MCOS", "table",
})


# ── .mat file parsing ──────────────────────────────────────────────────

def _get_workspace(filepath: str) -> bytes | None:
    """Read the __function_workspace__ from a v5 .mat file.

    When MATLAB saves a table to a .mat file, the actual table data lives
    inside a special variable called __function_workspace__. This is an
    opaque byte blob that scipy can't interpret — it just returns raw bytes.
    We extract those bytes here so the downstream functions can scan them.
    """
    try:
        raw = scipy.io.loadmat(filepath, squeeze_me=False, struct_as_record=True)
    except Exception:
        logger.warning("Failed to load %s", filepath)
        return None
    if "__function_workspace__" not in raw:
        logger.warning("No workspace in %s", filepath)
        return None
    return raw["__function_workspace__"][0].tobytes()


def _find_column_names(ws: bytes, first_col: str = "Header") -> list[str]:
    """Extract table column names from the workspace byte stream.

    Inside the workspace bytes, the column names are stored as consecutive
    ASCII strings. They always start with "Header" (the time column) and
    are followed by the sensor-specific column names.

    After the real column names, MATLAB appends internal metadata field
    names like "VariableDescriptions", "DimensionNames", etc. We stop
    collecting when we hit one of those (_TABLE_METADATA_FIELDS).

    Example for an EMG file, this extracts:
        ["Header", "gastrocmed", "tibialisanterior", "soleus", ...]

    Example for a gcRight file:
        ["Header", "HeelStrike", "ToeOff"]
    """
    start = ws.find(first_col.encode("ascii"))
    if start < 0:
        return []
    # Only search a limited region to avoid false matches deeper in the blob
    region = ws[start:start + 20_000]
    names: list[str] = []
    for m in re.finditer(rb"[a-zA-Z][a-zA-Z0-9_]{1,60}", region):
        token = m.group().decode("ascii")
        if token in _TABLE_METADATA_FIELDS:
            break
        names.append(token)
    return names


def _find_double_arrays(ws: bytes, min_elements: int = 100) -> list[np.ndarray]:
    """Scan the workspace for miDOUBLE (type tag = 9) data elements.

    MATLAB's MAT v5 format stores each numeric array with an 8-byte header:
        [4 bytes: type tag] [4 bytes: byte count]
    For double arrays, type tag = 9 (miDOUBLE). Each element is 8 bytes.

    This function walks through the byte stream looking for these headers,
    reads each array, and keeps only the ones whose length matches the most
    common length (= the number of rows in the table). This filters out
    small internal MATLAB arrays that aren't part of the table data.

    For example, an EMG file has ~30,000 rows and 12 columns (Header + 11
    muscles). This function finds 12 arrays of length ~30,000 each.
    """
    min_bytes = min_elements * 8
    ws_len = len(ws)
    candidates: list[tuple[int, np.ndarray]] = []

    for i in range(0, ws_len - 8, 4):
        # Check for miDOUBLE type tag (= 9)
        if int.from_bytes(ws[i:i + 4], "little") != 9:
            continue
        nbytes = int.from_bytes(ws[i + 4:i + 8], "little")
        if not (min_bytes <= nbytes < ws_len):
            continue
        n = nbytes // 8  # number of double elements
        end = i + 8 + nbytes
        if end > ws_len:
            continue
        arr = np.frombuffer(ws[i + 8:end], dtype=np.float64)
        if len(arr) == n and np.isfinite(arr[0]):
            candidates.append((n, arr.copy()))

    if not candidates:
        return []

    # The table columns all have the same row count — find the most common length
    lengths = [n for n, _ in candidates]
    row_count = max(set(lengths), key=lengths.count)
    return [arr for n, arr in candidates if n == row_count]


def load_mat_table(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load numeric columns from a MATLAB table stored in a v5 .mat file.

    Returns ``{column_name: 1d_array}``. String columns are skipped.

    The number of extracted double arrays may be less than the number of
    column names — this happens when the table has string columns (e.g.
    "Label" in conditions files). In that case, arrays are paired with
    column names from the start. This works because string columns are
    always at the end in the Camargo files.

    Example output for an EMG file:
        {
            "Header": array([0.000, 0.001, 0.002, ...]),  # time in seconds
            "gastrocmed": array([0.012, -0.003, ...]),     # raw EMG (mV)
            "tibialisanterior": array([...]),
            ...  # 11 muscle channels total
        }

    Example output for a gcRight file:
        {
            "Header": array([0.000, 0.005, 0.010, ...]),  # time (200 Hz)
            "HeelStrike": array([45.2, 46.1, ..., 99.8, 0.3, ...]),  # sawtooth 0-100%
            "ToeOff": array([...]),
        }
    """
    ws = _get_workspace(str(filepath))
    if ws is None:
        return {}

    columns = _find_column_names(ws)
    arrays = _find_double_arrays(ws)
    if not columns or not arrays:
        logger.warning("No extractable data in %s", filepath)
        return {}

    if len(arrays) != len(columns):
        logger.debug(
            "%s: %d numeric arrays, %d column names — pairing from start",
            Path(filepath).name, len(arrays), len(columns),
        )

    return dict(zip(columns, arrays))


# ── File discovery ─────────────────────────────────────────────────────

def discover_trials(
    data_root: str | Path,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    sensors: list[str] | None = None,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Walk the Camargo directory tree and return paths to .mat trial files.

    The Camargo dataset has a nested directory structure:
        {data_root}/{subject}/{date}/{mode}/{sensor}/*.mat

    Each subject has exactly one date subfolder (e.g. "20170907") plus
    optionally an "osimxml" folder (OpenSim models, ignored here).

    Returns nested dict: {subject: {mode: {sensor: [filepath, ...]}}}.
    Files within each sensor are sorted alphabetically (= trial order).
    """
    root = Path(data_root)
    subjects = subjects or ALL_SUBJECTS
    modes = modes or ALL_MODES
    sensors = sensors or ALL_SENSORS
    result: dict = {}

    for subj in subjects:
        subj_dir = root / subj
        if not subj_dir.is_dir():
            logger.warning("Missing subject dir: %s", subj_dir)
            continue

        date_dirs = [d for d in subj_dir.iterdir() if d.is_dir() and d.name != "osimxml"]
        if not date_dirs:
            logger.warning("No date folder in %s", subj_dir)
            continue
        date_dir = date_dirs[0]

        subj_data: dict = {}
        for mode in modes:
            mode_dir = date_dir / mode
            if not mode_dir.is_dir():
                continue
            mode_data: dict = {}
            for sensor in sensors:
                sensor_dir = mode_dir / sensor
                if not sensor_dir.is_dir():
                    continue
                files = sorted(str(f) for f in sensor_dir.glob("*.mat"))
                if files:
                    mode_data[sensor] = files
            if mode_data:
                subj_data[mode] = mode_data
        if subj_data:
            result[subj] = subj_data

    return result


# ── HDF5 export / import ──────────────────────────────────────────────

def parse_camargo(
    data_root: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    sensors: list[str] | None = None,
) -> None:
    """Parse the raw Camargo dataset into an HDF5 file.

    HDF5 layout::

        /{subject}/{mode}/{sensor}/{trial}   — float64 array (samples x cols)
            attrs: columns (list[str]), sample_rate (int)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trials = discover_trials(data_root, subjects, modes, sensors)
    logger.info("Found %d subjects", len(trials))

    with h5py.File(str(output_path), "w") as hf:
        for i, (subj, modes_dict) in enumerate(sorted(trials.items()), 1):
            logger.info("[%d/%d] %s", i, len(trials), subj)

            for mode, sensors_dict in sorted(modes_dict.items()):
                for sensor, files in sorted(sensors_dict.items()):
                    rate = SENSOR_SAMPLE_RATES.get(sensor, 0)

                    for fpath in files:
                        table = load_mat_table(fpath)
                        if not table:
                            continue

                        trial_name = Path(fpath).stem
                        group_path = f"{subj}/{mode}/{sensor}"
                        grp = hf.require_group(group_path)

                        cols = list(table.keys())
                        data = np.column_stack([table[c] for c in cols])
                        ds = grp.create_dataset(
                            trial_name, data=data,
                            compression="gzip", compression_opts=4,
                        )
                        ds.attrs["columns"] = cols
                        ds.attrs["sample_rate"] = rate

            hf.flush()

    logger.info("Wrote %s", output_path)


# ── Condition label extraction ─────────────────────────────────────────────
#
# levelground / ramp / stair conditions files have a string "Label" column.
# Our numeric parser (_find_double_arrays) ignores strings entirely.
# These functions scan the raw workspace bytes to recover condition labels.
#
# MATLAB v5 stores char arrays as miUINT16 (type tag = 6): each ASCII character
# occupies 2 bytes (little-endian uint16, high byte = 0 for ASCII).
# MATLAB categorical arrays store a cell of unique category strings PLUS a
# miINT32/miUINT8 index array that maps each row to a category.
# We try both a direct string scan and a categorical reconstruction.

# Known Camargo condition vocabulary (used to validate found strings)
_CAMARGO_LEVELGROUND_SPEEDS: list[str] = ["slow", "normal", "fast"]
_CAMARGO_RAMP_ANGLES: list[str] = ["18", "12.4", "11", "9.2", "7.8", "5.2"]  # longest first
_CAMARGO_STAIR_HEIGHTS_MM: list[str] = ["178", "152", "127", "102"]

# Strings found in every workspace that are NOT condition labels
_LABEL_SCAN_BLACKLIST: frozenset[str] = frozenset({
    "Header", "Speed", "Label", "HeelStrike", "ToeOff", "LevelGround",
    "Treadmill", "Ramp", "Stair", "Slow", "Normal", "Fast",
    "slow", "normal", "fast",  # added below via matching, not via scan
}) | _TABLE_METADATA_FIELDS


def _scan_utf16_strings(ws: bytes, min_chars: int = 2, max_chars: int = 60) -> list[str]:
    """Scan workspace bytes for MATLAB char arrays (miUINT16, type tag = 6).

    MATLAB stores char arrays as miUINT16 data elements. Each ASCII character
    occupies 2 bytes (little-endian, high byte = 0). The regular element header
    is 8 bytes: [4 bytes type=6] [4 bytes byte_count].

    Returns all decoded strings that consist entirely of printable ASCII.
    """
    results: list[str] = []
    ws_len = len(ws)
    i = 0

    while i < ws_len - 8:
        type_tag = int.from_bytes(ws[i:i + 4], "little")

        if type_tag == 6:  # miUINT16
            nbytes = int.from_bytes(ws[i + 4:i + 8], "little")
            n_chars = nbytes // 2

            if min_chars <= n_chars <= max_chars and i + 8 + nbytes <= ws_len:
                raw = ws[i + 8:i + 8 + nbytes]
                try:
                    s = raw.decode("utf-16-le").rstrip("\x00").strip()
                    if s and all(32 <= ord(c) < 128 for c in s):
                        results.append(s)
                except (UnicodeDecodeError, ValueError):
                    pass

            # Advance: 8-byte header + data + alignment padding to 8 bytes
            pad = (-nbytes) % 8 if nbytes > 0 else 0
            i += 8 + nbytes + pad
        else:
            i += 4

    return results


def _scan_ascii_label(ws: bytes, mode: str) -> str | None:
    """Fallback: scan workspace bytes for known Camargo label strings as raw ASCII.

    In some MATLAB versions strings are stored as miINT8 (plain ASCII bytes).
    We scan for exact byte matches of known label strings (case-insensitive).

    Returns the matched label in lowercase, or None.
    """
    if mode == "levelground":
        # Check "normal" first (contains "normal" not just "slow"/"fast" substrings)
        for label in ["normal", "fast", "slow"]:
            for variant in [label, label.capitalize(), label.upper()]:
                if variant.encode("ascii") in ws:
                    return label

    elif mode == "ramp":
        for angle in _CAMARGO_RAMP_ANGLES:
            if angle.encode("ascii") in ws:
                return angle

    elif mode == "stair":
        for height in _CAMARGO_STAIR_HEIGHTS_MM:
            if height.encode("ascii") in ws:
                return height

    return None


def _scan_index_array(ws: bytes, n_rows: int, tol: int = 20) -> np.ndarray | None:
    """Scan workspace for a MATLAB categorical index array of size ~n_rows.

    MATLAB categorical arrays store category membership as a packed integer
    array (miUINT8 type=2, miINT16 type=3, or miINT32 type=5). We look for
    arrays of approximately the right length whose values are small non-negative
    integers (valid category indices).

    Returns the first matching array as int64, or None.
    """
    INTEGER_TAGS: dict[int, tuple[type, int]] = {
        2: (np.uint8, 1),   # miUINT8
        3: (np.int16, 2),   # miINT16
        5: (np.int32, 4),   # miINT32
    }
    ws_len = len(ws)

    for i in range(0, ws_len - 8, 4):
        type_tag = int.from_bytes(ws[i:i + 4], "little")
        if type_tag not in INTEGER_TAGS:
            continue

        dtype, item_size = INTEGER_TAGS[type_tag]
        nbytes = int.from_bytes(ws[i + 4:i + 8], "little")
        n = nbytes // item_size

        if abs(n - n_rows) > tol or i + 8 + nbytes > ws_len:
            continue

        arr = np.frombuffer(ws[i + 8:i + 8 + nbytes], dtype=dtype).copy().astype(np.int64)
        # Valid categorical indices: non-negative, small (< 50 unique categories)
        if arr.min() >= 0 and 1 <= arr.max() < 50:
            return arr

    return None


def extract_condition_labels(
    filepath: str | Path,
    mode: str,
    n_rows: int | None = None,
) -> list[str] | str | None:
    """Extract condition label(s) from a Camargo conditions .mat file.

    Works for non-treadmill modes (levelground, ramp, stair) where the
    conditions file has a string Label column instead of a numeric Speed column.

    Strategy:
      1. Scan workspace for MATLAB char arrays (UTF-16LE, type=6) and match
         against the known Camargo label vocabulary for the mode.
      2. Fall back to ASCII scan if UTF-16LE scan finds nothing.
      3. If n_rows provided and multiple labels found (e.g. ramp up+down),
         attempt to reconstruct a per-sample label sequence by finding the
         MATLAB categorical index array and mapping indices → label strings.

    Args:
        filepath: Path to the conditions .mat file.
        mode:     "levelground", "ramp", or "stair".
        n_rows:   Expected row count (= n_samples in the conditions table).
                  Required for per-sample reconstruction; ignored otherwise.

    Returns:
        list[str] of length n_rows  — per-sample labels (reconstruction succeeded).
        str                         — single label for the whole trial.
        None                        — extraction failed.

    Label formats:
        levelground : "slow" | "normal" | "fast"
        ramp        : "5.2" | "5.2_up" | "5.2_down"  (angle in degrees)
        stair       : "102" | "102_up" | "102_down"   (height in mm)
    """
    if mode == "treadmill":
        return None  # treadmill uses numeric Speed column

    ws = _get_workspace(str(filepath))
    if ws is None:
        return None

    # --- 1. UTF-16LE scan ---
    utf16_strings = _scan_utf16_strings(ws)
    matched: list[str] = []

    def _add_unique(label: str) -> None:
        if label not in matched:
            matched.append(label)

    if mode == "levelground":
        for s in utf16_strings:
            if s.lower() in _CAMARGO_LEVELGROUND_SPEEDS:
                _add_unique(s.lower())

    elif mode == "ramp":
        for angle in _CAMARGO_RAMP_ANGLES:
            for s in utf16_strings:
                if angle in s:
                    sl = s.lower()
                    if any(d in sl for d in ("up", "asc")):
                        _add_unique(f"{angle}_up")
                    elif any(d in sl for d in ("down", "desc")):
                        _add_unique(f"{angle}_down")
                    else:
                        _add_unique(angle)

    elif mode == "stair":
        for height in _CAMARGO_STAIR_HEIGHTS_MM:
            for s in utf16_strings:
                if height in s:
                    sl = s.lower()
                    if any(d in sl for d in ("up", "asc")):
                        _add_unique(f"{height}_up")
                    elif any(d in sl for d in ("down", "desc")):
                        _add_unique(f"{height}_down")
                    else:
                        _add_unique(height)

    # --- 2. ASCII fallback ---
    if not matched:
        label = _scan_ascii_label(ws, mode)
        if label:
            matched = [label]

    if not matched:
        logger.debug(
            "extract_condition_labels: no labels found  mode=%s  file=%s",
            mode, Path(filepath).name,
        )
        return None

    if len(matched) == 1:
        return matched[0]

    # --- 3. Multiple labels → try per-sample reconstruction ---
    if n_rows is not None:
        idx_arr = _scan_index_array(ws, n_rows)
        if idx_arr is not None and int(idx_arr.max()) < len(matched):
            try:
                # MATLAB categorical indices are 1-based; check both 0- and 1-based
                offset = 0 if idx_arr.min() == 0 else 1
                return [matched[int(k) - offset] for k in idx_arr]
            except (IndexError, ValueError):
                pass
        logger.debug(
            "extract_condition_labels: %d labels found but index array reconstruction failed"
            "  mode=%s  file=%s  labels=%s",
            len(matched), mode, Path(filepath).name, matched,
        )

    # Return all found labels joined (can't determine per-sample distribution)
    # Ordered: ascending before descending for ramp/stair
    up = [m for m in matched if m.endswith("_up")]
    down = [m for m in matched if m.endswith("_down")]
    other = [m for m in matched if not m.endswith("_up") and not m.endswith("_down")]
    ordered = up + down + other
    return ",".join(ordered)


def load_camargo_dataset(
    hdf5_path: str | Path,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    sensors: list[str] | None = None,
) -> dict[str, dict[str, dict[str, dict[str, np.ndarray]]]]:
    """Load parsed data from an HDF5 file produced by :func:`parse_camargo`.

    Returns ``{subject: {mode: {sensor: {trial: array}}}}``.
    Each array has shape ``(n_samples, n_columns)``; column names are
    stored in the HDF5 dataset's ``columns`` attribute.
    """
    result: dict = {}
    with h5py.File(str(hdf5_path), "r") as hf:
        for subj in hf:
            if subjects and subj not in subjects:
                continue
            result[subj] = {}
            for mode in hf[subj]:
                if modes and mode not in modes:
                    continue
                result[subj][mode] = {}
                for sensor in hf[subj][mode]:
                    if sensors and sensor not in sensors:
                        continue
                    grp = hf[subj][mode][sensor]
                    result[subj][mode][sensor] = {
                        trial: grp[trial][:] for trial in grp
                    }
    return result
