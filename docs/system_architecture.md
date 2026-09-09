# BharatPotHole — System Architecture

## Overview

BharatPotHole is a real-time road hazard detection and monitoring system.
It uses dual YOLO models running on edge devices to detect road defects
from bus dashcam footage, geolocates them via GPS, and fuses detections
from multiple buses into a unified city-wide hazard map.

---

## Two Operating Modes

### 1. DEMO MODE (Browser Inference)

For local demonstrations and hackathon presentations.

```
Browser Camera / Uploaded Video
        ↓
Frontend frame capture (canvas)
        ↓
POST /infer (base64 JPEG)
        ↓
Backend loads YOLO Model 1 + Model 2
        ↓
Returns detections as JSON
        ↓
Frontend draws bounding boxes on canvas
        ↓
Backend auto-creates Observations
        ↓
Backend Event Fusion (Haversine 10m)
        ↓
WebSocket → Frontend map update
```

**Important:** The browser is NOT performing inference. It captures frames
and sends them to the backend, which runs YOLO. This is a demonstration
convenience only.

### 2. DEPLOYMENT MODE (Edge Inference)

For actual bus fleet deployment.

```
Bus Dashcam (front-facing camera)
        ↓
Edge Client (Python, runs on bus computer)
        ↓
YOLO Model 1 + Model 2 (local inference)
        ↓
GPS Module (real hardware)
        ↓
POST /observations (one per detection)
        ↓
Central Backend (FastAPI + MySQL)
        ↓
Event Fusion Engine (Haversine 10m radius)
        ↓
GeoJSON API
        ↓
Frontend Dashboard (monitoring only, no inference)
```

**The browser is NOT required for inference in deployment mode.**

---

## Multi-Bus Architecture

```
BUS-001 Camera ─→ Edge Client A ──┐
                                   │
BUS-002 Camera ─→ Edge Client B ──┼──→ Central Backend ──→ Event Fusion ──→ GIS Map
                                   │
BUS-003 Camera ─→ Edge Client C ──┘
```

All buses communicate with the SAME backend.

The backend is the single source of truth for Events.

If BUS-001, BUS-002, and BUS-003 all detect the same pothole within 10m,
the backend creates ONE fused event with `unique_bus_count = 3`.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React + TypeScript + Vite |
| Map | Leaflet + OpenStreetMap |
| Backend | FastAPI (Python) |
| Database | MySQL + SQLAlchemy |
| Models | YOLOv11n (Ultralytics) |
| Edge Client | Python + OpenCV |
| GPS | Hardware NMEA / Browser Geolocation / Simulated |
| Live Updates | WebSocket |

---

## Data Models

### Observation
A single detection from a single frame on a single bus.
Contains: bus_id, class_name, confidence, bbox, GPS coordinates, model name.

### Event
A fused real-world hazard. Created by the backend when an observation
cannot be matched to any existing event within 10m (Haversine distance).
Contains: event_type, aggregated coordinates, observation_count, unique_bus_count.

### Telemetry
Bus position updates sent even when no detection occurs.
Enables live fleet tracking on the map.

---

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | /health | Health check |
| POST | /observations | Edge client submits detection |
| GET | /observations | List observations |
| GET | /observations/{id} | Get single observation |
| GET | /events | List fused events |
| GET | /events/{id} | Get single event |
| GET | /events/{id}/observations | Observations for an event |
| GET | /events/geojson | GeoJSON for map rendering |
| POST | /infer | **DEMO ONLY** — Send frame for inference |
| POST | /telemetry | Bus position update |
| GET | /fleet/status | Fleet bus statuses |
| WS | /ws/events | Live event/telemetry broadcasts |

---

## Event Fusion

When an observation arrives:

1. Query all ACTIVE events of the same class within last 24 hours
2. Calculate Haversine distance to each candidate
3. If within 10m → fuse into existing event (update coordinates, confidence, bus count)
4. If no match → create new event
5. Link observation to event

The frontend does NOT perform any spatial deduplication.

---

## GPS Sources

| Source | Description |
|--------|-----------|
| REAL_DEVICE | Hardware GPS module or browser Geolocation API |
| REPLAY | CSV file with timestamped coordinates |
| SIMULATED | Linear movement from configurable start point |

**Rule:** Simulated GPS is always visually labeled. Never presented as real.

---

## WebSocket Messages

| Type | When |
|------|------|
| NEW_EVENT | A new event is created |
| EVENT_UPDATED | Existing event gets new observation |
| TELEMETRY_UPDATED | Bus position update |
| BUS_UPDATED | Bus status change |

Only metadata is sent via WebSocket. Never video frames.
