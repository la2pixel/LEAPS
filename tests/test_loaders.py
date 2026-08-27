"""Tests for leaps.data.loaders.

These tests run against the real Camargo dataset at $CAMARGO_DATA.
Skip them with ``pytest -m "not slow"`` if the data isn't available.
"""

import os
import tempfile

import numpy as np
import pytest

from leaps.data.loaders import (
    ALL_SENSORS,
    SENSOR_SAMPLE_RATES,
    discover_trials,
    load_camargo_dataset,
    load_mat_table,
    parse_camargo,
)

DATA_ROOT = os.environ.get("CAMARGO_DATA", "")
HAS_DATA = bool(DATA_ROOT) and os.path.isdir(DATA_ROOT)
needs_data = pytest.mark.skipif(not HAS_DATA, reason="Camargo dataset not available")


@needs_data
class TestDiscoverTrials:
    def test_finds_subject(self):
        result = discover_trials(DATA_ROOT, subjects=["AB09"], modes=["treadmill"])
        assert "AB09" in result
        assert "treadmill" in result["AB09"]

    def test_all_sensors_present_for_treadmill(self):
        result = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"],
        )
        sensors = set(result["AB09"]["treadmill"].keys())
        assert sensors == set(ALL_SENSORS)

    def test_returns_sorted_mat_files(self):
        result = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )
        files = result["AB09"]["treadmill"]["emg"]
        assert all(f.endswith(".mat") for f in files)
        assert files == sorted(files)

    def test_missing_subject_skipped(self):
        result = discover_trials(DATA_ROOT, subjects=["AB99"])
        assert result == {}

    def test_sensor_filter(self):
        result = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )
        assert list(result["AB09"]["treadmill"].keys()) == ["emg"]


@needs_data
class TestLoadMatTable:
    def test_emg_columns_and_shape(self):
        path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )["AB09"]["treadmill"]["emg"][0]
        table = load_mat_table(path)

        assert "Header" in table
        assert "gastrocmed" in table
        assert "rightexternaloblique" in table
        assert len(table) == 12  # Header + 11 muscles
        n = len(table["Header"])
        assert all(len(v) == n for v in table.values())

    def test_ik_has_24_columns(self):
        path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["ik"],
        )["AB09"]["treadmill"]["ik"][0]
        table = load_mat_table(path)
        assert len(table) == 24

    def test_gait_cycle_has_heelstrike_and_toeoff(self):
        path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["gcRight"],
        )["AB09"]["treadmill"]["gcRight"][0]
        table = load_mat_table(path)
        assert set(table.keys()) == {"Header", "HeelStrike", "ToeOff"}

    def test_header_is_monotonic(self):
        path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )["AB09"]["treadmill"]["emg"][0]
        table = load_mat_table(path)
        header = table["Header"]
        assert np.all(np.diff(header) > 0)

    def test_conditions_string_column_skipped(self):
        """Levelground conditions have a string 'Label' column — should be skipped."""
        path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["levelground"], sensors=["conditions"],
        )["AB09"]["levelground"]["conditions"][0]
        table = load_mat_table(path)
        assert "Header" in table
        assert "Label" not in table  # string column skipped


@needs_data
class TestParseAndReload:
    def test_round_trip(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        parse_camargo(
            DATA_ROOT, h5_path,
            subjects=["AB09"], modes=["treadmill"], sensors=["emg", "gcRight"],
        )
        assert h5_path.exists()

        data = load_camargo_dataset(h5_path)
        assert "AB09" in data
        emg_trials = data["AB09"]["treadmill"]["emg"]
        assert len(emg_trials) == 7  # 7 treadmill trials

        # Verify shape matches direct .mat load
        trial_name = sorted(emg_trials.keys())[0]
        mat_path = discover_trials(
            DATA_ROOT, subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )["AB09"]["treadmill"]["emg"][0]
        direct = load_mat_table(mat_path)
        expected = np.column_stack([direct[c] for c in direct])
        np.testing.assert_array_equal(emg_trials[trial_name], expected)

    def test_hdf5_attributes(self, tmp_path):
        import h5py

        h5_path = tmp_path / "test.h5"
        parse_camargo(
            DATA_ROOT, h5_path,
            subjects=["AB09"], modes=["treadmill"], sensors=["emg"],
        )

        with h5py.File(str(h5_path), "r") as f:
            ds = f["AB09/treadmill/emg/treadmill_01_01"]
            assert "columns" in ds.attrs
            assert ds.attrs["sample_rate"] == 1000
            cols = list(ds.attrs["columns"])
            assert cols[0] == "Header"
            assert len(cols) == ds.shape[1]

    def test_filter_on_reload(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        parse_camargo(
            DATA_ROOT, h5_path,
            subjects=["AB09"], modes=["treadmill"], sensors=["emg", "ik"],
        )

        data = load_camargo_dataset(h5_path, sensors=["ik"])
        assert "ik" in data["AB09"]["treadmill"]
        assert "emg" not in data["AB09"]["treadmill"]
