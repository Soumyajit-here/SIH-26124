import requests
import random
import time
import concurrent.futures

BASE_URL = "http://127.0.0.1:8000"

def simulate_bus(bus_id: str, scenario_points: list):
    """
    Simulates a single bus driving and capturing events.
    """
    results = []
    for p in scenario_points:
        # Add slight randomness to position to simulate GPS jitter (up to ~2 meters)
        jitter_lat = random.uniform(-0.000018, 0.000018)
        jitter_lon = random.uniform(-0.000018, 0.000018)
        
        obs_payload = {
            "bus_id": bus_id,
            "latitude": p["lat"] + jitter_lat,
            "longitude": p["lon"] + jitter_lon,
            "speed_kmh": p.get("speed", 45.0),
            "course_deg": p.get("course", 90.0),
            "model": "YOLO-V8-Step2",
            "class_id": p.get("class_id", 0),
            "class_name": p["class_name"],
            "confidence": random.uniform(0.75, 0.98),
            "bbox_x1": 0, "bbox_y1": 0, "bbox_x2": 100, "bbox_y2": 100
        }
        
        try:
            resp = requests.post(f"{BASE_URL}/observations", json=obs_payload)
            if resp.status_code == 201:
                results.append(resp.json())
            else:
                print(f"[{bus_id}] Error posting observation: {resp.text}")
        except Exception as e:
            print(f"[{bus_id}] Connection error: {e}")
            
        time.sleep(random.uniform(0.1, 0.5))
        
    return bus_id, results

def run_simulation():
    print("Starting Fleet Simulator...")
    
    # Event A: Pothole at 28.4327, 77.0149 (Seen by all 3 buses)
    # Event B: Crack at 28.4410, 77.0200 (Seen by Bus 1)
    # Event C: Pothole at 28.4410, 77.0200 (Seen by Bus 2, near the Crack but different class)
    
    scenario_bus1 = [
        {"lat": 28.4327, "lon": 77.0149, "class_name": "Pothole", "class_id": 0},
        {"lat": 28.4327, "lon": 77.0149, "class_name": "Pothole", "class_id": 0}, # Bus 1 sees it twice
        {"lat": 28.4410, "lon": 77.0200, "class_name": "Crack", "class_id": 1}
    ]
    
    scenario_bus2 = [
        {"lat": 28.4327, "lon": 77.0149, "class_name": "Pothole", "class_id": 0},
        {"lat": 28.4410, "lon": 77.0200, "class_name": "Pothole", "class_id": 0} # Event C (different class)
    ]
    
    scenario_bus3 = [
        {"lat": 28.4327, "lon": 77.0149, "class_name": "Pothole", "class_id": 0}
    ]
    
    buses = {
        "BUS-001": scenario_bus1,
        "BUS-002": scenario_bus2,
        "BUS-003": scenario_bus3
    }
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(simulate_bus, bus_id, points) for bus_id, points in buses.items()]
        
        for future in concurrent.futures.as_completed(futures):
            bus_id, res = future.result()
            print(f"[{bus_id}] Completed {len(res)} observations.")
            
    print("\n--- SIMULATION SUMMARY ---")
    try:
        events_resp = requests.get(f"{BASE_URL}/events")
        obs_resp = requests.get(f"{BASE_URL}/observations")
        
        events = events_resp.json()
        observations = obs_resp.json()
        
        print(f"Total observations: {len(observations)}")
        print(f"Total events: {len(events)}")
        
        unique_buses = set(o["bus_id"] for o in observations)
        print(f"Unique buses: {len(unique_buses)}\n")
        
        print("Events by type:")
        
        type_groups = {}
        for evt in events:
            t = evt["event_type"]
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(evt)
            
        for event_type, group in type_groups.items():
            print(f"\n{event_type}:")
            print(f"{len(group)} event(s)")
            total_buses = sum(e["unique_bus_count"] for e in group)
            total_obs = sum(e["observation_count"] for e in group)
            print(f"{total_buses} bus sightings total across events")
            print(f"{total_obs} observations total")
            for e in group:
                print(f"  -> Event {e['event_id']}: {e['observation_count']} obs from {e['unique_bus_count']} buses at {e['latitude']:.5f}, {e['longitude']:.5f}")
                
    except Exception as e:
        print("Could not retrieve final summary:", e)

if __name__ == "__main__":
    run_simulation()
