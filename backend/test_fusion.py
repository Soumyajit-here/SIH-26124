import os
os.environ["TESTING"] = "1"

import pytest
from fastapi.testclient import TestClient
from app import app
from event_fusion import haversine_distance

client = TestClient(app)

def test_haversine_distance():
    # TEST 17: Haversine distance unit tests
    # Same coordinate -> 0 meters
    assert haversine_distance(28.0, 77.0, 28.0, 77.0) == 0.0
    
    # Points ~3m apart (1 degree lat is ~111km, so 3m is ~0.000027 degrees)
    dist_3m = haversine_distance(28.432700, 77.014900, 28.432727, 77.014900)
    assert 2.5 <= dist_3m <= 3.5
    
    # Points ~10m apart
    dist_10m = haversine_distance(28.432700, 77.014900, 28.432790, 77.014900)
    assert 9.0 <= dist_10m <= 11.0

def test_fusion_scenario_1_to_3():
    # TEST 1: One bus + one pothole observation -> creates one event
    obs1 = {
        "bus_id": "BUS-001",
        "latitude": 28.432700,
        "longitude": 77.014900,
        "model": "YOLO-V8",
        "class_id": 0,
        "class_name": "Pothole",
        "confidence": 0.8,
        "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10
    }
    resp1 = client.post("/observations", json=obs1)
    assert resp1.status_code == 201
    data1 = resp1.json()
    assert data1["event_created"] is True
    event_id = data1["event_id"]

    # TEST 2: Same bus + same pothole within 10m -> same event
    obs2 = obs1.copy()
    obs2["latitude"] = 28.432727 # ~3m away
    resp2 = client.post("/observations", json=obs2)
    assert resp2.status_code == 201
    data2 = resp2.json()
    assert data2["event_created"] is False
    assert data2["event_id"] == event_id

    # TEST 3: Different bus + same pothole within 10m -> same event
    obs3 = obs1.copy()
    obs3["bus_id"] = "BUS-002"
    obs3["latitude"] = 28.432740 # ~4.5m away
    resp3 = client.post("/observations", json=obs3)
    assert resp3.status_code == 201
    data3 = resp3.json()
    assert data3["event_created"] is False
    assert data3["event_id"] == event_id

def test_fusion_scenario_4_and_5():
    # Baseline observation to set an event
    obs1 = {
        "bus_id": "BUS-001",
        "latitude": 10.000000,
        "longitude": 10.000000,
        "model": "YOLO-V8",
        "class_id": 0,
        "class_name": "Pothole",
        "confidence": 0.8,
        "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10
    }
    client.post("/observations", json=obs1)
    
    # TEST 4: Different class within 10m -> separate event
    obs_diff_class = obs1.copy()
    obs_diff_class["class_name"] = "Crack"
    resp4 = client.post("/observations", json=obs_diff_class)
    assert resp4.json()["event_created"] is True

    # TEST 5: Same class outside 10m -> separate event
    obs_outside = obs1.copy()
    obs_outside["latitude"] = 10.000200 # ~22 meters away
    resp5 = client.post("/observations", json=obs_outside)
    assert resp5.json()["event_created"] is True

def test_fusion_scenario_6():
    # TEST 6: One bus reports same event 10 times -> obs_count=10, unique_bus=1
    obs_base = {
        "bus_id": "BUS-X",
        "latitude": 40.000000,
        "longitude": -70.000000,
        "model": "YOLO-V8",
        "class_id": 0,
        "class_name": "Pothole",
        "confidence": 0.8,
        "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10
    }
    event_id = None
    for i in range(10):
        resp = client.post("/observations", json=obs_base)
        if i == 0:
            assert resp.json()["event_created"] is True
            event_id = resp.json()["event_id"]
        else:
            assert resp.json()["event_created"] is False
            
    # Verify via Event API
    evt_resp = client.get(f"/events/{event_id}")
    evt_data = evt_resp.json()
    assert evt_data["observation_count"] == 10
    assert evt_data["unique_bus_count"] == 1

def test_fusion_scenario_7_8_9():
    # TEST 7: Three buses report same event -> one event, unique_buses=3
    base_lat, base_lon = 50.000, -10.000
    buses = ["BUS-A", "BUS-B", "BUS-C"]
    
    event_id = None
    for idx, bus in enumerate(buses):
        obs = {
            "bus_id": bus,
            "latitude": base_lat,
            "longitude": base_lon,
            "model": "YOLO-V8",
            "class_id": 0,
            "class_name": "TrafficCone",
            "confidence": 0.9,
            "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 10, "bbox_y2": 10
        }
        resp = client.post("/observations", json=obs)
        if idx == 0:
            event_id = resp.json()["event_id"]
            
    # TEST 8: Event API returns correct counts
    evt_resp = client.get(f"/events/{event_id}")
    assert evt_resp.status_code == 200
    evt_data = evt_resp.json()
    assert evt_data["observation_count"] == 3
    assert evt_data["unique_bus_count"] == 3
    assert evt_data["event_type"] == "TrafficCone"
    
    # TEST 9: Event observations endpoint returns all raw observations
    obs_resp = client.get(f"/events/{event_id}/observations")
    assert obs_resp.status_code == 200
    obs_list = obs_resp.json()
    assert len(obs_list) == 3
    buses_returned = [o["bus_id"] for o in obs_list]
    assert set(buses_returned) == {"BUS-A", "BUS-B", "BUS-C"}
