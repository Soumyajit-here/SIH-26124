import os
os.environ["TESTING"] = "1"

import pytest
from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

def test_simulated_gps_observation():
    """Test that gps_source=SIMULATED is preserved through the API."""
    payload = {
        "bus_id": "BUS-TEST-GPS",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 0.85,
        "latitude": 28.432738,
        "longitude": 77.014969,
        "speed_kmh": 30.0,
        "course_deg": 90.0,
        "bbox_x1": 100.0,
        "bbox_y1": 150.0,
        "bbox_x2": 200.0,
        "bbox_y2": 250.0,
        "gps_source": "SIMULATED",
        "gps_accuracy_m": 10.0,
        "evidence_image": "inference_output/frames/frame_001.jpg"
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    data = response.json()
    
    obs = data["observation"]
    assert obs["bus_id"] == "BUS-TEST-GPS"
    assert obs["gps_source"] == "SIMULATED"
    assert obs["gps_accuracy_m"] == 10.0
    assert obs["class_name"] == "Pothole"
    assert obs["class_id"] == 3
    assert obs["model"] == "model_1"
    assert obs["latitude"] == 28.432738
    assert obs["longitude"] == 77.014969
    
    assert "event_id" in data
    assert "event_created" in data

def test_bus_id_preserved():
    """Verify bus_id flows through correctly."""
    payload = {
        "bus_id": "BUS-INTEGRATION-42",
        "model": "model_1",
        "class_name": "Crack",
        "class_id": 4,
        "confidence": 0.72,
        "latitude": 28.440000,
        "longitude": 77.020000,
        "bbox_x1": 50.0, "bbox_y1": 50.0, "bbox_x2": 150.0, "bbox_y2": 150.0,
        "gps_source": "SIMULATED",
        "gps_accuracy_m": 10.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    obs = response.json()["observation"]
    assert obs["bus_id"] == "BUS-INTEGRATION-42"

def test_class_info_preserved():
    """Verify class_id and class_name are preserved."""
    payload = {
        "bus_id": "BUS-001",
        "model": "model_1",
        "class_name": "SpeedBump",
        "class_id": 6,
        "confidence": 0.65,
        "latitude": 28.450000,
        "longitude": 77.030000,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 100.0, "bbox_y2": 100.0,
        "gps_source": "SIMULATED",
        "gps_accuracy_m": 10.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    obs = response.json()["observation"]
    assert obs["class_name"] == "SpeedBump"
    assert obs["class_id"] == 6

def test_event_fusion_from_inference():
    """Two nearby observations of same class should fuse into one event."""
    base = {
        "bus_id": "BUS-FUSE-TEST",
        "model": "model_1",
        "class_name": "Manhole",
        "class_id": 5,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 100.0, "bbox_y2": 100.0,
        "gps_source": "SIMULATED",
        "gps_accuracy_m": 10.0
    }
    
    obs1 = {**base, "confidence": 0.80, "latitude": 28.460000, "longitude": 77.040000}
    resp1 = client.post("/observations", json=obs1)
    assert resp1.status_code == 201
    event_id_1 = resp1.json()["event_id"]
    
    # ~3m away, same class -> should fuse
    obs2 = {**base, "confidence": 0.85, "latitude": 28.460020, "longitude": 77.040020}
    resp2 = client.post("/observations", json=obs2)
    assert resp2.status_code == 201
    event_id_2 = resp2.json()["event_id"]
    
    assert event_id_1 == event_id_2, "Nearby same-class observations should fuse to the same event"

def test_without_gps_source_backward_compat():
    """Existing payloads without gps_source should still work (backward compat)."""
    payload = {
        "bus_id": "BUS-OLD-STYLE",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 0.90,
        "latitude": 28.470000,
        "longitude": 77.050000,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 100.0, "bbox_y2": 100.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    obs = response.json()["observation"]
    assert obs["gps_source"] is None
    assert obs["gps_accuracy_m"] is None
