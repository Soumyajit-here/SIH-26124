import os
import csv
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(r"c:\Users\soumy\OneDrive\Desktop\SIH-21624")
THIRDEYE_DIR = ROOT / "THIRDEYE_SIH_TEST"
SYNC_CSV_PATH = THIRDEYE_DIR / "synchronized" / "frame_gps_sync.csv"
OUTPUT_CSV_PATH = THIRDEYE_DIR / "synchronized" / "geotagged_detections.csv"
MODEL_PATH = ROOT / "best.pt"

print("==================================================")
print("SIH26124: AI GEOSPATIAL PROTOTYPE INFERENCE")
print("==================================================")

print(f"Loading YOLO model from: {MODEL_PATH.name}")
model = YOLO(MODEL_PATH)

print(f"Loading geospatial synchronization table...")
sync_data = []
with open(SYNC_CSV_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sync_data.append(row)

print(f"Total synchronized frames to process: {len(sync_data)}")

geotagged_results = []
total_detections = 0

print("Running inference and data fusion...")
for i, row in enumerate(sync_data):
    # Print progress
    if (i + 1) % 10 == 0:
        print(f"  Processing frame {i + 1}/{len(sync_data)}...")
        
    img_rel_path = row["image_path"]
    img_abs_path = THIRDEYE_DIR / img_rel_path
    
    if not img_abs_path.exists():
        print(f"  WARNING: Image not found: {img_abs_path}")
        continue
        
    # Run YOLO Inference (low base threshold to catch everything)
    results = model.predict(source=str(img_abs_path), conf=0.10, imgsz=640, verbose=False)
    
    # Per-class thresholds to balance the prototype demo
    CLASS_THRESHOLDS = {
        0: 0.15,  # HMV - Catch more heavy vehicles
        1: 0.15,  # LMV - Catch more cars
        2: 0.25,  # Pedestrian
        3: 0.60,  # Pothole - High to avoid shadows
        4: 0.70,  # Crack - High to avoid shadows
        5: 0.70,  # Manhole - High to avoid shadows
        6: 0.50   # SpeedBump
    }
    
    # Process Detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            cls_name = model.names[cls_id]
            
            # Apply per-class threshold filtering
            if conf < CLASS_THRESHOLDS.get(cls_id, 0.45):
                continue
            
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            # Fuse with GPS Data
            fused_record = {
                "clip_id": row["clip_id"],
                "frame_timestamp_ms": row["frame_timestamp_ms"],
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "speed_kmh": row["speed_kmh"],
                "course_deg": row["course_deg"],
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": round(conf, 4),
                "bbox_x1": round(x1, 2),
                "bbox_y1": round(y1, 2),
                "bbox_x2": round(x2, 2),
                "bbox_y2": round(y2, 2)
            }
            
            geotagged_results.append(fused_record)
            total_detections += 1

print(f"Inference complete! Extracted {total_detections} total geotagged detections.")

print("Saving results...")
if geotagged_results:
    fieldnames = geotagged_results[0].keys()
    with open(OUTPUT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(geotagged_results)
    print(f"SUCCESS: Saved to {OUTPUT_CSV_PATH}")
else:
    print("WARNING: No objects detected in any frames.")
