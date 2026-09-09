import csv
import cv2
from pathlib import Path
from collections import defaultdict
import time

ROOT = Path(r"c:\Users\soumy\OneDrive\Desktop\SIH-21624")
THIRDEYE_DIR = ROOT / "THIRDEYE_SIH_TEST"
SYNC_CSV_PATH = THIRDEYE_DIR / "synchronized" / "frame_gps_sync.csv"
DETECTIONS_CSV_PATH = THIRDEYE_DIR / "synchronized" / "geotagged_detections.csv"

def load_detections():
    if not DETECTIONS_CSV_PATH.exists():
        return {}
    
    dets = defaultdict(list)
    with open(DETECTIONS_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["clip_id"], row["frame_timestamp_ms"])
            dets[key].append(row)
    return dets

def main():
    print("Loading detections...")
    detections = load_detections()
    
    frames = []
    with open(SYNC_CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(row)
            
    print(f"Loaded {len(frames)} frames. Starting live viewer...")
    print("Press 'q' to quit, 'p' to pause, or any other key to skip to next frame.")
    
    cv2.namedWindow("SIH26124 - Live Inference Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SIH26124 - Live Inference Viewer", 1280, 720)
    
    # Colors for different classes (BGR format)
    colors = {
        "HMV": (255, 0, 0),      # Blue
        "LMV": (0, 255, 0),      # Green
        "Pedestrian": (0, 255, 255), # Yellow
        "Pothole": (0, 0, 255),  # Red
        "Crack": (0, 165, 255),  # Orange
        "Manhole": (255, 0, 255),# Magenta
        "SpeedBump": (255, 255, 0) # Cyan
    }
    
    paused = False
    
    for row in frames:
        img_path = row["image_path"]
        abs_img_path = THIRDEYE_DIR / img_path
        
        if not abs_img_path.exists():
            continue
            
        # Load image
        img = cv2.imread(str(abs_img_path))
        
        # Draw overlay header (GPS, Speed)
        header_text = f"GPS: {row['latitude']}, {row['longitude']} | Speed: {row['speed_kmh']} km/h"
        cv2.rectangle(img, (0, 0), (1280, 40), (0, 0, 0), -1)
        cv2.putText(img, header_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw bounding boxes
        key = (row["clip_id"], row["frame_timestamp_ms"])
        frame_dets = detections.get(key, [])
        for det in frame_dets:
            x1, y1 = int(float(det["bbox_x1"])), int(float(det["bbox_y1"]))
            x2, y2 = int(float(det["bbox_x2"])), int(float(det["bbox_y2"]))
            cls_name = det["class_name"]
            conf = float(det["confidence"])
            
            color = colors.get(cls_name, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(img, (x1, y1 - 25), (x1 + w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        cv2.imshow("SIH26124 - Live Inference Viewer", img)
        
        # Wait 500ms (simulate 2 FPS for easier viewing)
        while True:
            key = cv2.waitKey(500 if not paused else 0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            else:
                break
                
    cv2.destroyAllWindows()
    print("Playback finished!")

if __name__ == "__main__":
    main()
