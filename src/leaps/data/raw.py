"""Camargo dataset parsing.

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
  - Sensor specific columns (e.g. 11 EMG muscles, or HeelStrike/ToeOff)

Only numeric columns are extracted. Trial transition indicators in string columns ("Label") are skipped.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from leaps.data.metadata import ALL_MODES, ALL_SUBJECT_IDS, SAMPLE_RATES

logger = logging.getLogger(__name__)

# dataset constants

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


#.mat parsing
def _get_workspace(filepath: str) -> bytes | None:
    """Read the __function_workspace__ from a v5 .mat file."""
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

    Example for an EMG file, this extracts:
        ["Header", "gastrocmed", "tibialisanterior", "soleus", ...]

    Example for a gcRight file:
        ["Header", "HeelStrike", "ToeOff"]
    """
    start = ws.find(first_col.encode("ascii"))
    if start < 0:
        return []
    # only search a limited region as we might get false matches
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

    """
    min_bytes = min_elements * 8
    ws_len = len(ws)
    candidates: list[tuple[int, np.ndarray]] = []

    for i in range(0, ws_len - 8, 4):
        # check for miDOUBLE type tag (= 9)
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

    # the table columns all have the same row count so we  find the most common length
    lengths = [n for n, _ in candidates]
    row_count = max(set(lengths), key=lengths.count)
    return [arr for n, arr in candidates if n == row_count]


def load_mat_table(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load numeric columns from a MATLAB table stored in a v5 .mat file.

    Returns ``{column_name: 1d_array}``. Skip string cols.

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


# find files

def discover_trials(
    data_root: str | Path,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    sensors: list[str] | None = None,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Scrape the Camargo directory tree and return paths to .mat trial files."""

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


#HDF5

def parse_camargo(
    data_root: str | Path,
    output_path: str | Path,
    subjects: list[str] | None = None,
    modes: list[str] | None = None,
    sensors: list[str] | None = None,
) -> None:
    """Parse the raw Camargo dataset into an HDF5 file.

    HDF5 layout::

        /{subject}/{mode}/{sensor}/{trial}   : float64 array (samples x cols)
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


# extract ambulation mode labels
# levelground / ramp / stair conditions files store:
#   - a float64 Header array (timestamps)
#   - a MATLAB cell array of per-sample Label strings


def _parse_label_cell_array(ws: bytes, cell_start: int, cell_nbytes: int) -> list[str]:
    """Parse a MAT v5 cell array of char strings from workspace bytes."""
    cell_data_start = cell_start + 8  # skip outer miMATRIX tag+nbytes
    cell_data_end = cell_start + 8 + cell_nbytes

    dims = np.frombuffer(ws[cell_data_start + 16 + 8: cell_data_start + 16 + 16], dtype="<i4")
    n_elements = int(dims[0]) * int(dims[1])

    pos = cell_data_start + 40
    labels: list[str] = []
#tuned to my setup
    while pos < cell_data_end - 8 and len(labels) < n_elements:
        tag = int.from_bytes(ws[pos: pos + 4], "little")
        if tag != 14:
            break
        elem_nbytes = int.from_bytes(ws[pos + 4: pos + 8], "little")
        elem_data = ws[pos + 8: pos + 8 + elem_nbytes]

        ds = 40
        if ds + 4 <= len(elem_data):
            type_low  = int.from_bytes(elem_data[ds: ds + 2], "little")      # always 16 (miUTF8)
            type_high = int.from_bytes(elem_data[ds + 2: ds + 4], "little")  # nonzero → compressed
            if type_high > 0:
                # compressed small element: type_high is the byte count (≤4 chars)
                raw = elem_data[ds + 4: ds + 4 + type_high]
            else:
                # full 8-byte element header: next 4 bytes are the byte count
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

    # the conditions table in the workspace begins with a float64 Header array,
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

    # read trial-level metadata (scalar fields accessible via scipy)
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
            # match to nearest known angle in vocabulary
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
