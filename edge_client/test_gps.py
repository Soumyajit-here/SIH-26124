"""
SIH26124 — GPS Unit Tests
Tests for GPS timestamp parsing, normalization, interpolation,
course wraparound, quality checks, and edge cases.

All test timestamps use realistic Unix millisecond values (13+ digits)
to ensure the auto-detection parser handles them correctly.
"""
import os
import sys
import tempfile
import csv

sys.path.insert(0, os.path.dirname(__file__))

import pytest
from gps import GPSLog, SimulatedGPS

# Base timestamp: 2026-09-05T10:00:00Z in milliseconds
T0 = 1788519600000

def _write_csv(rows, path, fields=("timestamp", "latitude", "longitude", "speed_kmh", "course_deg")):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


class TestTimestampParsing:
    def test_unix_milliseconds(self):
        ts = GPSLog._parse_timestamp("1773629740000")
        assert ts == 1773629740000

    def test_unix_seconds(self):
        ts = GPSLog._parse_timestamp("1773629740")
        assert ts == 1773629740000

    def test_iso_utc(self):
        ts = GPSLog._parse_timestamp("2026-09-05T10:00:00Z")
        assert ts is not None
        assert ts > 1e12

    def test_iso_with_tz(self):
        ts = GPSLog._parse_timestamp("2026-09-05T10:00:00+05:30")
        assert ts is not None

    def test_iso_no_tz(self):
        ts = GPSLog._parse_timestamp("2026-09-05T10:00:00")
        assert ts is not None

    def test_garbage(self):
        ts = GPSLog._parse_timestamp("not-a-timestamp")
        assert ts is None


class TestExactMatch:
    def test_exact_gps_lookup(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0000", "longitude": "77.0000", "speed_kmh": "30", "course_deg": "90"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0002", "longitude": "77.0002", "speed_kmh": "31", "course_deg": "91"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0)
        assert pt.match_type == "exact"
        assert pt.latitude == 28.0000
        assert pt.longitude == 77.0000

        pt2 = gps.get_position(T0 + 1000)
        assert pt2.match_type == "exact"
        assert pt2.latitude == 28.0002

        os.unlink(f.name)


class TestInterpolation:
    def test_midpoint_interpolation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0000", "longitude": "77.0000", "speed_kmh": "30", "course_deg": "90"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0002", "longitude": "77.0002", "speed_kmh": "32", "course_deg": "92"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 + 500)  # midpoint
        assert pt.match_type == "interpolated"
        assert abs(pt.latitude - 28.0001) < 0.00001
        assert abs(pt.longitude - 77.0001) < 0.00001
        assert abs(pt.speed_kmh - 31.0) < 0.1
        assert abs(pt.course_deg - 91.0) < 0.1

        os.unlink(f.name)

    def test_quarter_interpolation(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0000", "longitude": "77.0000", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0004", "longitude": "77.0004", "speed_kmh": "40", "course_deg": "0"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 + 250)  # 25%
        assert pt.match_type == "interpolated"
        assert abs(pt.latitude - 28.0001) < 0.00001

        os.unlink(f.name)


class TestCourseWraparound:
    def test_wraparound_350_to_10(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "30", "course_deg": "350"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "30", "course_deg": "10"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 + 500)
        # Should be ~0 (or 360), NOT ~180
        assert pt.course_deg < 10 or pt.course_deg > 350

        os.unlink(f.name)

    def test_wraparound_10_to_350(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "30", "course_deg": "10"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "30", "course_deg": "350"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 + 500)
        assert pt.course_deg < 10 or pt.course_deg > 350

        os.unlink(f.name)


class TestOutOfRange:
    def test_before_first_sample(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 - 1000)
        assert pt.match_type == "unavailable"

        os.unlink(f.name)

    def test_after_last_sample(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
            ], f.name)
            gps = GPSLog(f.name)

        pt = gps.get_position(T0 + 2000)
        assert pt.match_type == "unavailable"

        os.unlink(f.name)


class TestDuplicates:
    def test_duplicate_timestamps_removed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0 + 1000), "latitude": "28.1", "longitude": "77.1", "speed_kmh": "0", "course_deg": "0"},
            ], f.name)
            gps = GPSLog(f.name)

        assert gps.quality.total_samples == 2
        assert gps.quality.duplicates_removed == 1

        os.unlink(f.name)


class TestInvalidCoordinates:
    def test_invalid_lat_rejected(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            _write_csv([
                {"timestamp": str(T0), "latitude": "95.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
                {"timestamp": str(T0 + 1000), "latitude": "28.0", "longitude": "77.0", "speed_kmh": "0", "course_deg": "0"},
            ], f.name)
            gps = GPSLog(f.name)

        assert gps.quality.invalid_coordinates == 1
        assert gps.quality.total_samples == 1

        os.unlink(f.name)


class TestSimulatedGPS:
    def test_simulated_at_zero(self):
        sim = SimulatedGPS(28.0, 77.0, 30.0, 90.0)
        pt = sim.get_position(0)
        assert pt.latitude == 28.0
        assert pt.longitude == 77.0
        assert pt.match_type == "simulated"

    def test_simulated_moves(self):
        sim = SimulatedGPS(28.0, 77.0, 30.0, 90.0)
        pt = sim.get_position(10)
        assert pt.longitude > 77.0  # Moved east
        assert abs(pt.latitude - 28.0) < 0.0001  # Barely changed lat


class TestSampleGPSFile:
    def test_load_sample_csv(self):
        sample = os.path.join(os.path.dirname(__file__), "sample_gps.csv")
        if not os.path.exists(sample):
            pytest.skip("sample_gps.csv not found")
        gps = GPSLog(sample)
        assert gps.quality.total_samples >= 100
        assert gps.quality.duplicates_removed == 0
