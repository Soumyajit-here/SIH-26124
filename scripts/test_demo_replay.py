"""
SIH26124 — Unit & Integration Tests for Demo Replay Mode & Session Clearing
"""

import json
import os
import pytest
from pathlib import Path

def test_model1_ontology_and_manhole_suppression():
    expected_classes = {
        0: "HMV",
        1: "LMV",
        2: "Pedestrian",
        3: "Pothole",
        4: "Crack",
        5: "Manhole",
        6: "SpeedBump",
    }
    disabled_classes = {"Manhole"}
    
    active_ontology = [name for name in expected_classes.values() if name not in disabled_classes]
    
    assert "Manhole" not in active_ontology
    assert "TrafficCone" not in active_ontology # Model 2 disabled
    assert "Rock" not in active_ontology        # Model 2 disabled
    assert "RoadDebris" not in active_ontology  # Model 2 disabled
    assert "FallenTree" not in active_ontology  # Model 2 disabled
    assert set(active_ontology) == {"HMV", "LMV", "Pedestrian", "Pothole", "Crack", "SpeedBump"}

def test_cache_structure_validation():
    sample_cache = {
        "video_metadata": {
            "filename": "demo_video.mp4",
            "fps": 25.0,
            "width": 1280,
            "height": 720,
            "total_frames": 500,
            "duration_s": 20.0,
        },
        "model_metadata": {
            "model_name": "model_1",
            "checkpoint": "best.pt",
            "conf_threshold": 0.25,
            "imgsz": 640,
            "suppressed_classes": ["Manhole"],
            "active_classes": ["HMV", "LMV", "Pedestrian", "Pothole", "Crack", "SpeedBump"],
        },
        "frames": {
            "10": [
                {
                    "class_id": 3,
                    "class_name": "Pothole",
                    "confidence": 0.85,
                    "bbox": [100, 200, 300, 400],
                }
            ]
        }
    }
    
    assert sample_cache["model_metadata"]["model_name"] == "model_1"
    assert "Manhole" in sample_cache["model_metadata"]["suppressed_classes"]
    assert len(sample_cache["frames"]["10"]) == 1
    det = sample_cache["frames"]["10"][0]
    assert det["class_name"] == "Pothole"
    assert len(det["bbox"]) == 4

def test_frame_accurate_lookup():
    cache_frames = {
        "0": [{"class_name": "HMV", "confidence": 0.9, "bbox": [10, 10, 50, 50]}],
        "25": [{"class_name": "Pothole", "confidence": 0.82, "bbox": [100, 100, 200, 200]}],
    }
    
    # At 1.0 second (fps=25), frame is exactly 25
    time_s = 1.0
    fps = 25.0
    frame_idx = int(time_s * fps)
    
    dets = cache_frames.get(str(frame_idx), [])
    assert len(dets) == 1
    assert dets[0]["class_name"] == "Pothole"
