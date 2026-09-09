"""
SIH26124 — Model 2 Tests
Verifies that Model 2 loads, has the correct class mapping,
and produces the expected payload structures.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import pytest
import config
from inference import YOLORunner

class TestModel2:
    def test_model2_config(self):
        """Verify config settings for Model 2."""
        assert config.MODEL2_NAME == "model_2"
        assert os.path.isfile(config.MODEL2_PATH), f"Model 2 not found at {config.MODEL2_PATH}"
        
        expected = {
            0: "TrafficCone",
            1: "Rock",
            2: "RoadDebris",
            3: "FallenTree",
        }
        assert config.EXPECTED_CLASSES_MODEL2 == expected

    def test_model2_initialization(self):
        """Verify YOLORunner can initialize Model 2 and verify its classes."""
        runner = YOLORunner(
            model_path=config.MODEL2_PATH,
            expected_classes=config.EXPECTED_CLASSES_MODEL2,
            imgsz=320  # smaller size for fast test
        )
        assert runner.model is not None
        assert runner.class_names[0] == "TrafficCone"
        assert runner.class_names[1] == "Rock"
        assert runner.class_names[2] == "RoadDebris"
        assert runner.class_names[3] == "FallenTree"
