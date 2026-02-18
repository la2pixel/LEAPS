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
