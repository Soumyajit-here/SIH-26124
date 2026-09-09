"""
SIH26124 Edge Client — GPS Log Parser & Synchronizer

Loads a timestamped GPS CSV, validates records, and provides
frame-accurate GPS lookups via linear interpolation.

Internal representation: all timestamps normalized to Unix milliseconds (int).

Refactored from the validated synchronize_thirdeye_gps.py logic.
"""

import csv
import bisect
import math
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GPSPoint:
    latitude: float
    longitude: float
    speed_kmh: float = 0.0
    course_deg: float = 0.0
    match_type: str = "unavailable"   # exact | interpolated | nearest_fallback | unavailable
    timestamp_ms: int = 0


@dataclass
class GPSQualityReport:
    total_samples: int = 0
    duplicates_removed: int = 0
    invalid_coordinates: int = 0
    large_jumps: int = 0
    gaps: List[Tuple[int, int]] = field(default_factory=list)  # (ts_ms, gap_ms) pairs
    time_span_s: float = 0.0
    avg_interval_ms: float = 0.0


class GPSLog:
    """
    Load, validate, and query a timestamped GPS CSV log.
    Supports Unix seconds, Unix milliseconds, and ISO-8601 timestamps.
    """

    GAP_THRESHOLD_MS = 5000  # gaps > 5s are reported

    def __init__(self, csv_path: str, timestamp_col: str = "timestamp",
                 lat_col: str = "latitude", lon_col: str = "longitude",
                 speed_col: str = "speed_kmh", course_col: str = "course_deg"):
        self.csv_path = csv_path
        self.col_names = {
            "ts": timestamp_col, "lat": lat_col, "lon": lon_col,
            "speed": speed_col, "course": course_col,
        }
        self.timestamps: List[int] = []        # sorted Unix ms
        self.latitudes: List[float] = []
        self.longitudes: List[float] = []
        self.speeds: List[float] = []
        self.courses: List[float] = []
        self.quality = GPSQualityReport()

        self._load(csv_path)

    # ── Loading ───────────────────────────────────────────────────────────

    def _load(self, path: str):
        raw_rows = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = row.get(self.col_names["ts"], "").strip()
                lat_raw = row.get(self.col_names["lat"], "").strip()
                lon_raw = row.get(self.col_names["lon"], "").strip()

                if not ts_raw or not lat_raw or not lon_raw:
                    continue

                ts_ms = self._parse_timestamp(ts_raw)
                if ts_ms is None:
                    continue

                try:
                    lat = float(lat_raw)
                    lon = float(lon_raw)
                except ValueError:
                    continue

                if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                    self.quality.invalid_coordinates += 1
                    continue

                speed = self._safe_float(row.get(self.col_names["speed"], ""), 0.0)
                course = self._safe_float(row.get(self.col_names["course"], ""), 0.0)

                raw_rows.append((ts_ms, lat, lon, speed, course))

        # Sort by timestamp
        raw_rows.sort(key=lambda r: r[0])

        # Deduplicate
        seen_ts = set()
        deduped = []
        for r in raw_rows:
            if r[0] in seen_ts:
                self.quality.duplicates_removed += 1
                continue
            seen_ts.add(r[0])
            deduped.append(r)

        # Detect large jumps
        for i in range(1, len(deduped)):
            dlat = abs(deduped[i][1] - deduped[i - 1][1])
            dlon = abs(deduped[i][2] - deduped[i - 1][2])
            if dlat > 0.01 or dlon > 0.01:  # ~1km jump
                self.quality.large_jumps += 1

        # Detect gaps
        for i in range(1, len(deduped)):
            gap = deduped[i][0] - deduped[i - 1][0]
            if gap > self.GAP_THRESHOLD_MS:
                self.quality.gaps.append((deduped[i - 1][0], gap))

        # Store
        for ts_ms, lat, lon, speed, course in deduped:
            self.timestamps.append(ts_ms)
            self.latitudes.append(lat)
            self.longitudes.append(lon)
            self.speeds.append(speed)
            self.courses.append(course)

        n = len(self.timestamps)
        self.quality.total_samples = n
        if n >= 2:
            self.quality.time_span_s = (self.timestamps[-1] - self.timestamps[0]) / 1000.0
            self.quality.avg_interval_ms = (
                (self.timestamps[-1] - self.timestamps[0]) / (n - 1)
            )

    # ── Timestamp parsing ─────────────────────────────────────────────────

    @staticmethod
    def _parse_timestamp(raw: str) -> Optional[int]:
        """Auto-detect and convert to Unix milliseconds.
        
        Heuristic:
        - 13+ digits (>= 1e12): already milliseconds
        - 10-12 digits (>= 1e9, < 1e12): seconds → multiply by 1000
        - < 1e9: treat as seconds → multiply by 1000
        """
        # Try numeric first
        try:
            val = float(raw)
            if val >= 1e12:
                # Already milliseconds (e.g. 1773629740000)
                return int(val)
            else:
                # Seconds → convert to ms
                return int(val * 1000)
        except ValueError:
            pass

        # Try ISO-8601
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.datetime.strptime(raw, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return int(dt.timestamp() * 1000)
            except ValueError:
                continue

        return None

    @staticmethod
    def _safe_float(val: str, default: float) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return default

    # ── Query ─────────────────────────────────────────────────────────────

    def get_position(self, timestamp_ms: int) -> GPSPoint:
        """
        Return the GPS position for a given timestamp (Unix ms).
        Uses binary search + linear interpolation.
        """
        n = len(self.timestamps)
        if n == 0:
            return GPSPoint(0.0, 0.0, match_type="unavailable")

        # Out of range
        if timestamp_ms < self.timestamps[0] or timestamp_ms > self.timestamps[-1]:
            return GPSPoint(0.0, 0.0, match_type="unavailable")

        # Binary search for insertion point
        idx = bisect.bisect_left(self.timestamps, timestamp_ms)

        # Exact match
        if idx < n and self.timestamps[idx] == timestamp_ms:
            return GPSPoint(
                latitude=self.latitudes[idx],
                longitude=self.longitudes[idx],
                speed_kmh=self.speeds[idx],
                course_deg=self.courses[idx],
                match_type="exact",
                timestamp_ms=timestamp_ms,
            )

        # We need before and after
        if idx == 0:
            return GPSPoint(0.0, 0.0, match_type="unavailable")
        if idx >= n:
            return GPSPoint(0.0, 0.0, match_type="unavailable")

        i0 = idx - 1
        i1 = idx
        t0 = self.timestamps[i0]
        t1 = self.timestamps[i1]
        dt = t1 - t0
        alpha = (timestamp_ms - t0) / dt if dt > 0 else 0.0

        lat = self.latitudes[i0] + alpha * (self.latitudes[i1] - self.latitudes[i0])
        lon = self.longitudes[i0] + alpha * (self.longitudes[i1] - self.longitudes[i0])
        speed = self.speeds[i0] + alpha * (self.speeds[i1] - self.speeds[i0])

        # Course interpolation with 0/360 wraparound
        course = self._interpolate_course(
            self.courses[i0], self.courses[i1], alpha
        )

        return GPSPoint(
            latitude=round(lat, 8),
            longitude=round(lon, 8),
            speed_kmh=round(speed, 2),
            course_deg=round(course, 2),
            match_type="interpolated",
            timestamp_ms=timestamp_ms,
        )

    @staticmethod
    def _interpolate_course(c0: float, c1: float, alpha: float) -> float:
        """Interpolate heading correctly across the 0°/360° boundary."""
        diff = c1 - c0
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        result = c0 + alpha * diff
        return result % 360


# ── Simulated GPS (fallback when no log supplied) ─────────────────────────

EARTH_RADIUS_M = 6_371_000.0


def move_point(lat: float, lon: float, bearing_deg: float, distance_m: float):
    """Move a geographic point along a bearing by distance_m metres."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_R = distance_m / EARTH_RADIUS_M

    new_lat = math.asin(
        math.sin(lat_r) * math.cos(d_R)
        + math.cos(lat_r) * math.sin(d_R) * math.cos(brng)
    )
    new_lon = lon_r + math.atan2(
        math.sin(brng) * math.sin(d_R) * math.cos(lat_r),
        math.cos(d_R) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


class SimulatedGPS:
    """Generate a synthetic GPS path for testing when no real log exists."""

    def __init__(self, start_lat: float, start_lon: float,
                 speed_kmh: float, course_deg: float):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.speed_kmh = speed_kmh
        self.course_deg = course_deg

    def get_position(self, elapsed_s: float) -> GPSPoint:
        distance_m = (self.speed_kmh / 3.6) * elapsed_s
        lat, lon = move_point(
            self.start_lat, self.start_lon, self.course_deg, distance_m
        )
        return GPSPoint(
            latitude=round(lat, 8),
            longitude=round(lon, 8),
            speed_kmh=self.speed_kmh,
            course_deg=self.course_deg,
            match_type="simulated",
        )
