"""
SIH26124 Edge Client — YOLO Inference Runner

Loads best.pt, verifies the class mapping, and provides a simple
run_frame(frame) interface returning structured Detection objects.
"""

import os
import sys
from dataclasses import dataclass
from typing import List

from ultralytics import YOLO
import config


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float


class YOLORunner:
    def __init__(self, model_path: str, expected_classes: dict,
                 imgsz: int = config.IMGSZ,
                 conf: float = config.CONF_THRESHOLD):
        if not os.path.isfile(model_path):
            print(f"ERROR: Model checkpoint not found: {model_path}")
            sys.exit(1)

        self.model = YOLO(model_path)
        self.imgsz = imgsz
        self.conf = conf
        self.class_names = self.model.names  # {0: 'HMV', ...}

        # Verify class mapping
        for cid, expected_name in expected_classes.items():
            actual = self.class_names.get(cid)
            if actual != expected_name:
                print(f"FATAL: Class {cid} expected '{expected_name}', got '{actual}'")
                print("Model class mapping does not match. STOPPING.")
                sys.exit(1)

        print(f"Model loaded: {model_path}")
        print(f"Classes: {self.class_names}")

    def run_frame(self, frame) -> List[Detection]:
        """Run YOLO on a single frame, return list of Detection objects."""
        results = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)
        detections = []

        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
            conf_val = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=round(conf_val, 4),
                bbox_x1=round(x1, 1),
                bbox_y1=round(y1, 1),
                bbox_x2=round(x2, 1),
                bbox_y2=round(y2, 1),
            ))

        return detections
