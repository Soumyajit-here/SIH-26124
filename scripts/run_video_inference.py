import cv2
import os
from ultralytics import YOLO

VIDEO_PATH = r"C:\Users\soumy\Downloads\05.09.2026_20.04.09_REC.mp4"
MODEL_PATH = "best.pt"
OUTPUT_DIR = "rec_video_stress_test_output"

def run_video_inference():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Could not find video at {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_video_path = os.path.join(OUTPUT_DIR, "annotated_video.avi")

    print(f"Loading model {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    print(f"Processing video and saving to {out_video_path}...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run inference
        results = model(frame, conf=0.25, verbose=False)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    out.release()
    print(f"Finished! Annotated video saved to {out_video_path}")

if __name__ == "__main__":
    run_video_inference()
