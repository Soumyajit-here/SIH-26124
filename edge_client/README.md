# SIH26124 — Edge Client

## Overview

The Edge Client simulates the software running on a camera-equipped public transport bus. It processes dashcam video alongside a timestamped GPS log, runs YOLO Model 1 for hazard detection, synchronizes detections with GPS coordinates, and transmits geotagged observations to the central FastAPI backend.

## Architecture

```
CAMERA / VIDEO
      +
GPS LOG (CSV)
      ↓
EDGE CLIENT
      ↓
YOLO Model 1 (best.pt)
      ↓
frame timestamp → GPS synchronization
      ↓
geotagged observation
      ↓
POST /observations → FastAPI → MySQL → Event Fusion → GIS Map
```

## GPS Input

### CSV Format

The GPS CSV must contain at minimum:

| Column | Required | Description |
|--------|----------|-------------|
| `timestamp` | Yes | Unix seconds, Unix milliseconds, or ISO-8601 |
| `latitude` | Yes | WGS84 latitude (-90 to 90) |
| `longitude` | Yes | WGS84 longitude (-180 to 180) |
| `speed_kmh` | Optional | Vehicle speed in km/h |
| `course_deg` | Optional | Heading in degrees (0-360) |

### Timestamp Formats

The parser auto-detects:
- Unix seconds: `1773629740`
- Unix milliseconds: `1773629740000`
- ISO-8601: `2026-09-05T10:00:00Z`

### GPS Synchronization

For each video frame:
1. Calculate frame timestamp from video FPS
2. Binary search the GPS log for surrounding samples
3. If exact match → use directly
4. If between two samples → linear interpolation (lat, lon, speed, course)
5. Course interpolation handles 0°/360° wraparound correctly
6. If outside GPS range → mark as `unavailable`

## GPS Source Labeling

| Mode | `gps_source` | When |
|------|-------------|------|
| Real GPS log supplied | `REAL_LOG` | `--gps gps.csv` |
| No GPS log | `SIMULATED` | No `--gps` flag |

**IMPORTANT:** The GPS coordinate represents the **bus/camera position**, NOT the exact physical location of the detected object.

## Usage

### With Real GPS Log
```bash
python edge_client.py \
    --video "video.mp4" \
    --gps "gps.csv" \
    --video-start-time "2026-09-05T20:04:09Z" \
    --bus-id BUS-001 \
    --backend "http://127.0.0.1:8000"
```

### Without GPS (Simulated)
```bash
python edge_client.py \
    --video "video.mp4" \
    --bus-id BUS-001 \
    --backend "http://127.0.0.1:8000"
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | Required | Path to input video |
| `--gps` | None | Path to GPS CSV |
| `--bus-id` | BUS-001 | Bus identifier |
| `--backend` | http://127.0.0.1:8000 | Backend URL |
| `--video-start-time` | Now (relative mode) | Absolute video start (ISO-8601) |
| `--realtime` | Off | Process at video's native FPS |
| `--send-interval` | 1 | Transmit every Nth detection-frame |
| `--conf` | 0.25 | YOLO confidence threshold |
| `--imgsz` | 640 | YOLO inference size |
| `--gps-accuracy` | 10.0 | GPS accuracy estimate (meters) |

## Output Files

```
edge_output/
├── observations.csv          # All detections with API status
├── observations.json         # Same in JSON
├── failed_observations.jsonl # Failed API transmissions
├── evidence/                 # Annotated detection frames
└── edge_integration_report.txt
```

## Network Resilience

- Failed HTTP requests do NOT crash the video loop
- Configurable retry policy (default: 3 attempts)
- Failed observations saved to `failed_observations.jsonl`

## Running Tests

```bash
pytest test_gps.py test_edge_client.py -v
```

## Limitations

- GPS coordinate = vehicle position, not object position
- Image-sequence input (e.g. ThirdEye 1-FPS frames) not yet supported
- No real-time GPS hardware integration
- No Model 2 support yet
