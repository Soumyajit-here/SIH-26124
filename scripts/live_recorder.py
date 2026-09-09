import cv2
import os
import time
from ultralytics import YOLO

def run_live_inference():
    MODEL_PATH = "best.pt"
    OUTPUT_DIR = "live_recordings"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    
    # Open default webcam (0)
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps != fps: # Check for NaN or 0
        fps = 30.0
        
    # Setup video writer
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_video_path = os.path.join(OUTPUT_DIR, f"recording_{timestamp}.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    
    print(f"\nRecording to: {out_video_path}")
    print("Press 'q' to stop recording and exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            # Run inference
            results = model(frame, conf=0.25, verbose=False)
            
            # Plot results on frame
            annotated_frame = results[0].plot()
            
            # Save frame to video
            out.write(annotated_frame)
            
            # Display live feed
            cv2.imshow("Live Pothole Detection (Press 'q' to quit)", annotated_frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # Clean up
        print("Stopping recording...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved recording to {out_video_path}")

if __name__ == "__main__":
    run_live_inference()
