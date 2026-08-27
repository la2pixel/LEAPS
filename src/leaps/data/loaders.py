"""Camargo dataset parser and HDF5 loader.

Parses raw .mat files from the Camargo dataset.


The raw dataset is organized as:
    {data_root}/
    ├── AB06/
    │   └── 20170907/               ← date folder (one per subject)
    │       ├── treadmill/
    │       │   ├── emg/            ← 7 .mat files (one per trial block)
    │       │   │   ├── AB06_treadmill_emg_01.mat
    │       │   │   └── ...
    │       │   ├── gcRight/        ← gait cycle: HeelStrike (0–100%)
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

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from leaps.data.metadata import ALL_MODES, ALL_SUBJECT_IDS, EMG_CHANNELS, SAMPLE_RATES, SUBJECTS

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
# levelground / ramp / stair conditions files store:
#   - a float64 Header array (timestamps)
#   - a MATLAB cell array of per-sample Label strings
# Our numeric parser ignores the cell array. These functions read it directly
# from the workspace bytes using the MAT v5 cell-array binary layout.

def _parse_label_cell_array(ws: bytes, cell_start: int, cell_nbytes: int) -> list[str]:
    """Parse a MAT v5 cell array of char strings from workspace bytes.

    Each cell element is a miMATRIX (tag=14) containing a char array.
    Short strings (≤4 chars) use the compressed small-element format;
    longer strings use the full 8-byte header.
    """
    cell_data_start = cell_start + 8  # skip outer miMATRIX tag+nbytes
    cell_data_end = cell_start + 8 + cell_nbytes
    # Cell array preamble: flags(16B) + dims(16B) + name(8B) = 40B
    dims = np.frombuffer(ws[cell_data_start + 16 + 8: cell_data_start + 16 + 16], dtype="<i4")
    n_elements = int(dims[0]) * int(dims[1])

    pos = cell_data_start + 40
    labels: list[str] = []

    while pos < cell_data_end - 8 and len(labels) < n_elements:
        tag = int.from_bytes(ws[pos: pos + 4], "little")
        if tag != 14:
            break
        elem_nbytes = int.from_bytes(ws[pos + 4: pos + 8], "little")
        elem_data = ws[pos + 8: pos + 8 + elem_nbytes]

        # Each char-array element has flags(16B) + dims(16B) + name(8B) = 40B before data.
        ds = 40
        if ds + 4 <= len(elem_data):
            type_low  = int.from_bytes(elem_data[ds: ds + 2], "little")      # always 16 (miUTF8)
            type_high = int.from_bytes(elem_data[ds + 2: ds + 4], "little")  # nonzero → compressed
            if type_high > 0:
                # Compressed small element: type_high is the byte count (≤4 chars)
                raw = elem_data[ds + 4: ds + 4 + type_high]
            else:
                # Full 8-byte element header: next 4 bytes are the byte count
                str_len = int.from_bytes(elem_data[ds + 4: ds + 8], "little")
                raw = elem_data[ds + 8: ds + 8 + str_len]
            try:
                labels.append(raw.decode("ascii"))
            except Exception:
                labels.append("")

        pos += 8 + elem_nbytes
        if elem_nbytes % 8 != 0:
            pos += 8 - (elem_nbytes % 8)

    return labels


def load_conditions_table(
    filepath: str | Path,
    mode: str,
) -> tuple[np.ndarray, list[str], dict] | None:
    """Load per-sample label sequence and trial metadata from a non-treadmill conditions file.

    Returns ``(header_times, per_sample_labels, meta)`` or ``None`` on failure.
    ``meta`` keys:
        levelground : leadingLegStart, leadingLegStop, turn
        ramp        : transLegAscent (ndarray[str]), transLegDescent (ndarray[str]), rampIncline
        stair       : transLegAscent (ndarray[str]), transLegDescent (ndarray[str]), stairHeight
    """
    if mode == "treadmill":
        return None

    import scipy.io as sio

    try:
        raw_mat = scipy.io.loadmat(str(filepath), squeeze_me=False, struct_as_record=True)
    except Exception as exc:
        logger.debug("load_conditions_table: failed %s: %s", Path(filepath).name, exc)
        return None

    if "__function_workspace__" not in raw_mat:
        return None

    ws = raw_mat["__function_workspace__"][0].tobytes()

    # The conditions table in the workspace begins with a float64 Header array,
    # immediately followed by a miMATRIX cell array of per-sample label strings.
    header: np.ndarray | None = None
    cell_start: int | None = None
    for i in range(0, len(ws) - 8, 4):
        if int.from_bytes(ws[i: i + 4], "little") != 9:  # miDOUBLE
            continue
        nbytes = int.from_bytes(ws[i + 4: i + 8], "little")
        n = nbytes // 8
        if n < 200:
            continue
        header = np.frombuffer(ws[i + 8: i + 8 + nbytes], dtype=np.float64).copy()
        cell_start = i + 8 + nbytes
        break

    if header is None or cell_start is None or cell_start + 8 > len(ws):
        return None

    cell_tag = int.from_bytes(ws[cell_start: cell_start + 4], "little")
    if cell_tag != 14:  # not miMATRIX
        return None
    cell_nbytes = int.from_bytes(ws[cell_start + 4: cell_start + 8], "little")
    labels = _parse_label_cell_array(ws, cell_start, cell_nbytes)

    n = min(len(labels), len(header))
    if n == 0:
        return None
    labels = labels[:n]
    header = header[:n]

    # Read trial-level metadata (scalar fields accessible via scipy)
    try:
        meta_raw = sio.loadmat(str(filepath), simplify_cells=True)
    except Exception:
        meta_raw = {}

    meta: dict = {}
    for key in ("leadingLegStart", "leadingLegStop", "turn",
                "transLegAscent", "transLegDescent",
                "rampIncline", "stairHeight"):
        val = meta_raw.get(key)
        if val is not None:
            meta[key] = val

    return header, labels, meta


def extract_condition_labels(
    filepath: str | Path,
    mode: str,
    n_rows: int | None = None,
) -> list[str] | str | None:
    """Extract condition label from a Camargo conditions .mat file.

    Uses scipy.io.loadmat to read the struct fields directly (conditions files
    are simple MATLAB structs, not table objects, so scipy handles them fine).

    Args:
        filepath: Path to the conditions .mat file.
        mode:     "levelground", "ramp", or "stair".
        n_rows:   Unused (kept for API compatibility).

    Returns:
        str   — single label for the whole trial, or None on failure.

    Label formats:
        levelground : "slow" | "normal" | "fast"
        ramp        : "5.2" | "7.8" | "9.2" | "11" | "12.4" | "18"  (degrees)
        stair       : "102" | "127" | "152" | "178"  (mm)
    """
    if mode == "treadmill":
        return None  # treadmill uses numeric Speed column

    import scipy.io as sio

    try:
        data = sio.loadmat(str(filepath), simplify_cells=True)
    except Exception as exc:
        logger.debug("extract_condition_labels: failed to load %s: %s", Path(filepath).name, exc)
        return None

    if mode == "levelground":
        speed = data.get("speed")
        if speed is not None:
            s = str(speed).strip().lower()
            if s in ("slow", "normal", "fast"):
                return s

    elif mode == "ramp":
        incline = data.get("rampIncline")
        if incline is not None:
            angle = float(incline)
            # Match to nearest known angle in vocabulary
            known = [5.2, 7.8, 9.2, 11.0, 12.4, 18.0]
            labels = ["5.2", "7.8", "9.2", "11", "12.4", "18"]
            closest_idx = min(range(len(known)), key=lambda i: abs(known[i] - angle))
            if abs(known[closest_idx] - angle) < 0.5:
                return labels[closest_idx]

    elif mode == "stair":
        height_in = data.get("stairHeight")
        if height_in is not None:
            height_mm = round(float(height_in) * 25.4)
            known_mm = [102, 127, 152, 178]
            labels = ["102", "127", "152", "178"]
            closest_idx = min(range(len(known_mm)), key=lambda i: abs(known_mm[i] - height_mm))
            if abs(known_mm[closest_idx] - height_mm) < 15:
                return labels[closest_idx]

    logger.debug(
        "extract_condition_labels: no label found  mode=%s  file=%s",
        mode, Path(filepath).name,
    )
    return None


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


# ── Full preprocessing pipeline ───────────────────────────────────────────────

def build_emg_h5(
    data_root: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    ref_speed: float = 1.35,
) -> None:
    """Preprocess raw Camargo EMG → HDF5 (replicates STRIDES.m).

    Per-subject normalization is derived from treadmill strides at ref_speed,
    then applied across all modes.

    HDF5 layout per subject (all arrays indexed by stride)::

        /{subject}.attrs          — age, gender, height_m, mass_kg
        /{subject}/strides        — (n, 101, 11) float32
        /{subject}/modes          — (n,) str  — treadmill | levelground | ramp | stair
        /{subject}/speeds         — (n,) float64  — m/s for treadmill, NaN otherwise
        /{subject}/conditions     — (n,) str  — speed "1.35" | "slow/normal/fast" | angle | height
        /{subject}/labels         — (n,) str  — phase label: "walk" | "rampascent" | "turnccw" | …
        /{subject}/durations      — (n,) float64  — stride duration in seconds
        /{subject}/trial_indices  — (n,) int32
        /{subject}/min_vals       — (11,) normalization min
        /{subject}/max_vals       — (11,) normalization max

    Pooled across all subjects::

        /activations        — (N*101, 11) float32 snapshot matrix for AE training
        /stride_modes       — (N,) str
        /stride_subjects    — (N,) str
        /stride_speeds      — (N,) float64
        /stride_conditions  — (N,) str
        /stride_labels      — (N,) str
        /stride_durations   — (N,) float64
    """
    from leaps.data.processing import compute_normalization, process_mode_trials

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    subjects = subjects or ALL_SUBJECTS
    trials = discover_trials(
        data_root,
        subjects=subjects,
        modes=ALL_MODES,
        sensors=["emg", "gcRight", "conditions"],
    )

    all_activations:       list[np.ndarray] = []
    all_stride_modes:      list[str]        = []
    all_stride_subjects:   list[str]        = []
    all_stride_speeds:     list[float]      = []
    all_stride_conditions: list[str]        = []
    all_stride_labels:     list[str]        = []
    all_stride_durations:  list[float]      = []

    str_dt = h5py.string_dtype()

    with h5py.File(output_path, "w") as hf:
        hf.attrs["emg_channels"] = EMG_CHANNELS

        for i, subj in enumerate(sorted(trials), 1):
            logger.info("[%d/%d] Processing %s", i, len(trials), subj)
            subj_trials = trials[subj]

            if "treadmill" not in subj_trials:
                logger.warning("Skipping %s: no treadmill data for normalization", subj)
                continue

            tm = subj_trials["treadmill"]
            tm_emg  = [load_mat_table(f) for f in sorted(tm.get("emg",        []))]
            tm_gc   = [load_mat_table(f) for f in sorted(tm.get("gcRight",     []))]
            tm_cond = [load_mat_table(f) for f in sorted(tm.get("conditions",  []))]

            if not all(tm_emg) or not all(tm_gc) or not all(tm_cond):
                logger.warning("Skipping %s: incomplete treadmill data", subj)
                continue

            min_vals, max_vals = compute_normalization(tm_emg, tm_gc, tm_cond, ref_speed=ref_speed)

            subj_strides:       list[np.ndarray] = []
            subj_modes:         list[str]        = []
            subj_speeds:        list[np.ndarray] = []
            subj_trial_indices: list[np.ndarray] = []
            subj_conditions:    list[str]        = []
            subj_labels:        list[str]        = []
            subj_durations:     list[np.ndarray] = []

            for mode in ALL_MODES:
                if mode not in subj_trials:
                    continue

                mode_data = subj_trials[mode]
                emg_files = sorted(mode_data.get("emg",     []))
                gc_files  = sorted(mode_data.get("gcRight", []))
                if not emg_files or not gc_files:
                    continue

                emg_tables = [load_mat_table(f) for f in emg_files]
                gc_tables  = [load_mat_table(f) for f in gc_files]
                if not all(emg_tables) or not all(gc_tables):
                    logger.warning("  %s/%s: some files failed to parse", subj, mode)
                    continue

                cond_files  = sorted(mode_data.get("conditions", []))
                cond_tables = [load_mat_table(f) for f in cond_files] if cond_files else None

                condition_labels = cond_label_data = None
                if mode != "treadmill" and cond_files:
                    condition_labels = [extract_condition_labels(cf, mode) for cf in cond_files]
                    cond_label_data  = [load_conditions_table(cf, mode) for cf in cond_files]
                    n_failed = sum(1 for d in cond_label_data if d is None)
                    if n_failed:
                        logger.warning(
                            "  %s/%s: failed to load per-sample labels for %d/%d trials",
                            subj, mode, n_failed, len(cond_files),
                        )

                strides_3d, mode_speeds, mode_trial_idx, mode_conditions, mode_labels, mode_durations = (
                    process_mode_trials(
                        emg_tables, gc_tables, min_vals, max_vals,
                        cond_tables=cond_tables,
                        condition_labels=condition_labels,
                        cond_label_data=cond_label_data,
                        mode=mode,
                    )
                )

                n = strides_3d.shape[0]
                if n == 0:
                    continue

                if mode != "treadmill":
                    dist = "  ".join(f"{k}:{v}" for k, v in sorted(Counter(mode_conditions).items()))
                    logger.info("  %s/%s: %d strides  [%s]", subj, mode, n, dist)
                else:
                    logger.info("  %s/%s: %d strides", subj, mode, n)

                subj_strides.append(strides_3d)
                subj_modes.extend([mode] * n)
                subj_speeds.append(mode_speeds)
                subj_trial_indices.append(mode_trial_idx)
                subj_conditions.extend(mode_conditions)
                subj_labels.extend(mode_labels)
                subj_durations.append(mode_durations)

            if not subj_strides:
                logger.warning("Skipping %s: no valid strides", subj)
                continue

            all_strides   = np.concatenate(subj_strides,       axis=0)
            all_speeds    = np.concatenate(subj_speeds,         axis=0)
            all_trial_idx = np.concatenate(subj_trial_indices,  axis=0)
            all_durations = np.concatenate(subj_durations,      axis=0)
            activations   = all_strides.reshape(-1, all_strides.shape[-1])

            grp = hf.create_group(subj)
            info = SUBJECTS.get(subj)
            if info:
                grp.attrs["age"]       = info.age
                grp.attrs["gender"]    = info.gender
                grp.attrs["height_m"]  = info.height
                grp.attrs["mass_kg"]   = info.mass

            grp.create_dataset("strides",       data=all_strides,                              compression="gzip")
            grp.create_dataset("modes",          data=np.array(subj_modes,       dtype=object), dtype=str_dt)
            grp.create_dataset("speeds",         data=all_speeds)
            grp.create_dataset("conditions",     data=np.array(subj_conditions,  dtype=object), dtype=str_dt)
            grp.create_dataset("labels",         data=np.array(subj_labels,      dtype=object), dtype=str_dt)
            grp.create_dataset("durations",      data=all_durations)
            grp.create_dataset("trial_indices",  data=all_trial_idx)
            grp.create_dataset("min_vals",       data=min_vals)
            grp.create_dataset("max_vals",       data=max_vals)

            all_activations.extend([activations])
            all_stride_modes.extend(subj_modes)
            all_stride_subjects.extend([subj] * len(subj_modes))
            all_stride_speeds.extend(all_speeds.tolist())
            all_stride_conditions.extend(subj_conditions)
            all_stride_labels.extend(subj_labels)
            all_stride_durations.extend(all_durations.tolist())

            logger.info("  %s total: %d strides, %d snapshots", subj, all_strides.shape[0], activations.shape[0])

        if not all_activations:
            logger.error("No data processed.")
            return

        pooled = np.concatenate(all_activations, axis=0)
        hf.create_dataset("activations",       data=pooled,                                               compression="gzip")
        hf.create_dataset("stride_modes",      data=np.array(all_stride_modes,      dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_subjects",   data=np.array(all_stride_subjects,   dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_speeds",     data=np.array(all_stride_speeds,     dtype=np.float64))
        hf.create_dataset("stride_conditions", data=np.array(all_stride_conditions, dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_labels",     data=np.array(all_stride_labels,     dtype=object),       dtype=str_dt)
        hf.create_dataset("stride_durations",  data=np.array(all_stride_durations,  dtype=np.float64))

    logger.info(
        "Wrote %s — %d strides, %d snapshots from %d subjects",
        output_path, len(all_stride_modes), pooled.shape[0], len(all_activations),
    )
    for mode in ALL_MODES:
        count = all_stride_modes.count(mode)
        if count:
            logger.info("  %s: %d strides", mode, count)
