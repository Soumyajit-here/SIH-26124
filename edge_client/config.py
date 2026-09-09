"""
SIH26124 Edge Client — Configuration
"""

# YOLO Model 1
MODEL_PATH = "../best.pt"
IMGSZ = 640
CONF_THRESHOLD = 0.25
MODEL_NAME = "model_1"

EXPECTED_CLASSES = {
    0: "HMV",
    1: "LMV",
    2: "Pedestrian",
    3: "Pothole",
    4: "Crack",
    5: "Manhole",
    6: "SpeedBump",
}

# YOLO Model 2
MODEL2_PATH = r"C:\Users\soumy\OneDrive\Desktop\SIH-21624\best (model_2).pt"
MODEL2_NAME = "model_2"

EXPECTED_CLASSES_MODEL2 = {
    0: "TrafficCone",
    1: "Rock",
    2: "RoadDebris",
    3: "FallenTree",
}

# Backend
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"

# GPS defaults
DEFAULT_GPS_ACCURACY_M = 10.0

# Simulated GPS defaults (used only when --gps is not supplied)
SIM_START_LAT = 28.432738
SIM_START_LON = 77.014969
SIM_SPEED_KMH = 30.0
SIM_COURSE_DEG = 90.0

# Network resilience
MAX_RETRIES = 3
RETRY_DELAY_S = 0.5
HTTP_TIMEOUT_S = 5

# Output
OUTPUT_DIR = "edge_output"
EVIDENCE_DIR = "edge_output/evidence"
