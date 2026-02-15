"""Camargo dataset parser and HDF5 loader.

Parses raw .mat files from the Camargo et al. (2021) locomotion dataset
into a structured HDF5 file. The .mat files store MATLAB tables as MCOS
objects inside a ``__function_workspace__`` byte stream. This module
extracts column names and numeric data by scanning that stream directly.

Only numeric columns are extracted — string columns (e.g. ``Label`` in
conditions files) are skipped.
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

    Names start at *first_col* and end at the first MATLAB metadata field.
    """
    start = ws.find(first_col.encode("ascii"))
    if start < 0:
        return []
    region = ws[start:start + 20_000]
    names: list[str] = []
    for m in re.finditer(rb"[a-zA-Z][a-zA-Z0-9_]{1,60}", region):
        token = m.group().decode("ascii")
        if token in _TABLE_METADATA_FIELDS:
            break
        names.append(token)
    return names


def _find_double_arrays(ws: bytes, min_elements: int = 100) -> list[np.ndarray]:
    """Scan the workspace for miDOUBLE (type tag = 9) elements.

    Returns arrays whose length matches the most common length found
    (i.e. the table row count). This filters out unrelated small arrays.
    """
    min_bytes = min_elements * 8
    ws_len = len(ws)
    candidates: list[tuple[int, np.ndarray]] = []

    for i in range(0, ws_len - 8, 4):
        if int.from_bytes(ws[i:i + 4], "little") != 9:
            continue
        nbytes = int.from_bytes(ws[i + 4:i + 8], "little")
        if not (min_bytes <= nbytes < ws_len):
            continue
        n = nbytes // 8
        end = i + 8 + nbytes
        if end > ws_len:
            continue
        arr = np.frombuffer(ws[i + 8:end], dtype=np.float64)
        if len(arr) == n and np.isfinite(arr[0]):
            candidates.append((n, arr.copy()))

    if not candidates:
        return []

    lengths = [n for n, _ in candidates]
    row_count = max(set(lengths), key=lengths.count)
    return [arr for n, arr in candidates if n == row_count]


def load_mat_table(filepath: str | Path) -> dict[str, np.ndarray]:
    """Load numeric columns from a MATLAB table stored in a v5 .mat file.

    Returns ``{column_name: 1d_array}``. String columns are skipped.
    When the number of numeric arrays doesn't match the column count,
    arrays are paired with column names from the start (safe when string
    columns appear at the end, which is the case for all Camargo files).
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

    Returns ``{subject: {mode: {sensor: [filepaths]}}}``.
    Auto-detects each subject's single date subfolder.
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
