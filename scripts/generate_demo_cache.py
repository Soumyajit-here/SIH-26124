"""
SIH26124 / BharatPotHole
Demo Detection Cache Generator (Model 1 Real Inference)

Runs the trained Model 1 (best.pt) on a target video offline, suppresses Manhole detections,
and generates a structured JSON detection cache for frame-accurate, zero-inference-latency
demo playback in the web dashboard.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import cv2
from ultralytics import YOLO

EXPECTED_CLASSES = {
    0: "HMV",
    1: "LMV",
    2: "Pedestrian",
    3: "Pothole",
    4: "Crack",
    5: "Manhole",
    6: "SpeedBump",
}

DISABLED_CLASSES = {"Manhole"}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Model 1 demo detection cache")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--model", default="best.pt", help="Path to Model 1 checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    parser.add_argument("--output", default="demo_replay/detections.json", help="Output JSON path")
    parser.add_argument("--copy-to-frontend", action="store_true", default=True,
                        help="Also copy to frontend/public/demo_replay/detections.json")
    return parser.parse_args()

def main():
    args = parse_args()
    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 75)
    print(" BharatPotHole — Model 1 Demo Cache Generator")
    print("=" * 75)
    print(f"Video  : {video_path}")
    print(f"Model  : {model_path}")
    print(f"Conf   : {args.conf}")
    print(f"Imgsz  : {args.imgsz}")

    # Load Model
    print("\nLoading Model 1...")
    model = YOLO(str(model_path))

    # Open Video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if total_frames > 0 else 0

    print(f"\nVideo Info: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames ({duration_s:.2f}s)")

    cache_data = {
        "video_metadata": {
            "filename": video_path.name,
            "fps": round(fps, 2),
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "duration_s": round(duration_s, 2),
        },
        "model_metadata": {
            "model_name": "model_1",
            "checkpoint": str(model_path.name),
            "conf_threshold": args.conf,
            "imgsz": args.imgsz,
            "suppressed_classes": list(DISABLED_CLASSES),
            "active_classes": [name for name in EXPECTED_CLASSES.values() if name not in DISABLED_CLASSES],
        },
        "frames": {},
    }

    frame_idx = 0
    total_detections = 0
    start_time = time.perf_counter()

    print("\nRunning Model 1 inference and building frame index...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        res = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        frame_detections = []

        if res.boxes is not None:
            names = res.names
            for box in res.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                cls_name = str(names[cls_id]) if isinstance(names, dict) else str(names[cls_id])

                # Suppress Manhole
                if cls_name in DISABLED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                frame_detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2]
                })
                total_detections += 1

        if frame_detections:
            cache_data["frames"][str(frame_idx)] = frame_detections

        frame_idx += 1
        if frame_idx % 100 == 0 or frame_idx == total_frames:
            elapsed = time.perf_counter() - start_time
            cur_fps = frame_idx / elapsed if elapsed > 0 else 0
            print(f"  Frame {frame_idx:05d}/{total_frames} | Speed: {cur_fps:.1f} FPS | Detections: {total_detections}")

    cap.release()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, indent=2)

    print(f"\n[OK] Detection cache saved to: {out_path.resolve()}")
    print(f"Total Frames: {frame_idx} | Total Detections: {total_detections}")

    if args.copy_to_frontend:
        frontend_dir = Path("frontend/public/demo_replay")
        frontend_dir.mkdir(parents=True, exist_ok=True)
        frontend_dest = frontend_dir / "detections.json"
        with open(frontend_dest, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        print(f"[OK] Copied default cache to frontend: {frontend_dest.resolve()}")

if __name__ == "__main__":
    main()
