import os
os.environ["TESTING"] = "1"

import pytest
from fastapi.testclient import TestClient

from app import app
# The app automatically uses an in-memory SQLite database because TESTING=1 is set.

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_create_observation():
    payload = {
        "bus_id": "BUS-001",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 0.82,
        "latitude": 28.432738,
        "longitude": 77.014969,
        "speed_kmh": 34.2,
        "course_deg": 307.8,
        "bbox_x1": 100.0,
        "bbox_y1": 150.0,
        "bbox_x2": 200.0,
        "bbox_y2": 250.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert "observation" in data
    obs = data["observation"]
    assert "observation_id" in obs
    assert obs["bus_id"] == "BUS-001"
    assert obs["class_name"] == "Pothole"
    assert obs["latitude"] == 28.432738

def test_get_observations():
    response = client.get("/observations")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert data[0]["bus_id"] == "BUS-001"

def test_get_observation_by_id():
    # First, get all to find a valid ID
    response_all = client.get("/observations")
    obs_id = response_all.json()[0]["observation_id"]
    
    response = client.get(f"/observations/{obs_id}")
    assert response.status_code == 200
    assert response.json()["observation_id"] == obs_id

def test_get_observation_not_found():
    response = client.get("/observations/999999")
    assert response.status_code == 404

def test_invalid_latitude():
    payload = {
        "bus_id": "BUS-002",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 0.82,
        "latitude": 95.0, # Invalid (>90)
        "longitude": 77.014969,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 20.0, "bbox_y2": 20.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 422

def test_invalid_longitude():
    payload = {
        "bus_id": "BUS-002",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 0.82,
        "latitude": 28.0,
        "longitude": -190.0, # Invalid (<-180)
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 20.0, "bbox_y2": 20.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 422

def test_invalid_confidence():
    payload = {
        "bus_id": "BUS-002",
        "model": "model_1",
        "class_name": "Pothole",
        "class_id": 3,
        "confidence": 1.5, # Invalid (>1.0)
        "latitude": 28.0,
        "longitude": 77.0,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 20.0, "bbox_y2": 20.0
    }
    response = client.post("/observations", json=payload)
    assert response.status_code == 422

def test_geojson_endpoint():
    response = client.get("/events/geojson")
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert "features" in data
    
    # We should have at least one event from the first test
    features = data["features"]
    assert len(features) >= 1
    
    first_feature = features[0]
    assert first_feature["type"] == "Feature"
    assert first_feature["geometry"]["type"] == "Point"
    
    # Check [longitude, latitude] ordering
    coords = first_feature["geometry"]["coordinates"]
    assert len(coords) == 2
    assert coords[0] == 77.014969 # Longitude
    assert coords[1] == 28.432738 # Latitude
    
    props = first_feature["properties"]
    assert props["event_type"] == "Pothole"
    assert props["status"] == "ACTIVE"
    assert props["observation_count"] >= 1

def test_session_clear_and_event_recalculation():
    # 1. Create a baseline observation without session
    base_payload = {
        "bus_id": "BUS-BASE",
        "model": "model_1",
        "class_name": "Crack",
        "class_id": 4,
        "confidence": 0.70,
        "latitude": 28.500000,
        "longitude": 77.100000,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 20.0, "bbox_y2": 20.0
    }
    r_base = client.post("/observations", json=base_payload)
    assert r_base.status_code == 201
    event_id = r_base.json()["event_id"]

    # 2. Add an observation with session_id="DEMO-TEST-123" near the baseline crack
    demo_payload = {
        "bus_id": "BUS-DEMO",
        "model": "model_1",
        "class_name": "Crack",
        "class_id": 4,
        "confidence": 0.90,
        "latitude": 28.500010,
        "longitude": 77.100010,
        "bbox_x1": 10.0, "bbox_y1": 10.0, "bbox_x2": 20.0, "bbox_y2": 20.0,
        "session_id": "DEMO-TEST-123"
    }
    r_demo = client.post("/observations", json=demo_payload)
    assert r_demo.status_code == 201
    assert r_demo.json()["event_id"] == event_id

    # Verify event observation count increased to 2
    r_evt = client.get(f"/events/{event_id}")
    assert r_evt.json()["observation_count"] == 2
    assert r_evt.json()["aggregated_confidence"] == 0.90

    # 3. Clear session "DEMO-TEST-123"
    r_clear = client.post("/sessions/DEMO-TEST-123/clear")
    assert r_clear.status_code == 200
    res = r_clear.json()
    assert res["deleted_observations"] == 1
    assert res["updated_events"] == 1
    assert res["deleted_events"] == 0

    # 4. Verify baseline event was recalculated back to 1 observation and 0.70 confidence
    r_evt_after = client.get(f"/events/{event_id}")
    assert r_evt_after.status_code == 200
    assert r_evt_after.json()["observation_count"] == 1
    assert r_evt_after.json()["aggregated_confidence"] == 0.70

