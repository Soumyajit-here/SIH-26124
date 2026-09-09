"""
Generate a sample GPS CSV trace for integration testing.
This creates a SYNTHETIC 107-second trace at 1Hz along a Delhi road segment.
GPS source must be marked SIMULATED when using this file.
"""
import csv
import math
import datetime

EARTH_RADIUS_M = 6_371_000.0
START_LAT = 28.432738
START_LON = 77.014969
SPEED_KMH = 30.0
COURSE_DEG = 90.0
DURATION_S = 108  # slightly longer than 107.4s video
INTERVAL_S = 1


def move_point(lat, lon, bearing_deg, distance_m):
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


if __name__ == "__main__":
    start_dt = datetime.datetime(2026, 9, 5, 20, 4, 9, tzinfo=datetime.timezone.utc)
    rows = []

    for i in range(DURATION_S + 1):
        elapsed = i * INTERVAL_S
        distance = (SPEED_KMH / 3.6) * elapsed
        lat, lon = move_point(START_LAT, START_LON, COURSE_DEG, distance)
        ts_ms = int(start_dt.timestamp() * 1000) + (elapsed * 1000)

        # Add slight speed variation for realism
        speed = SPEED_KMH + (math.sin(i * 0.3) * 3.0)

        rows.append({
            "timestamp": str(ts_ms),
            "latitude": f"{lat:.8f}",
            "longitude": f"{lon:.8f}",
            "speed_kmh": f"{speed:.2f}",
            "course_deg": f"{COURSE_DEG:.1f}",
        })

    with open("sample_gps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "latitude", "longitude", "speed_kmh", "course_deg"])
        w.writeheader()
        w.writerows(rows)

    print(f"Generated sample_gps.csv with {len(rows)} points ({DURATION_S}s)")
