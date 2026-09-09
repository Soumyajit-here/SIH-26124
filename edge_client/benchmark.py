"""
SIH26124 — Performance Benchmark (Step 6)
Compares inference latency for Model 1 alone, Model 2 alone, and Both sequentially.
Outputs to edge_output/performance_comparison.csv
"""
import os
import time
import csv
import cv2
import statistics
import config
from inference import YOLORunner

def run_benchmark(video_path: str, num_frames: int = 100):
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        return

    # Load models
    print("Loading Model 1...")
    m1 = YOLORunner(config.MODEL_PATH, config.EXPECTED_CLASSES)
    print("Loading Model 2...")
    m2 = YOLORunner(config.MODEL2_PATH, config.EXPECTED_CLASSES_MODEL2)

    modes = ["model_1_only", "model_2_only", "model_1_plus_model_2"]
    results = []

    cap = cv2.VideoCapture(video_path)
    frames = []
    print(f"Reading {num_frames} frames for benchmark...")
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        print("No frames read.")
        return

    # Warmup
    print("Warming up models...")
    for _ in range(3):
        m1.run_frame(frames[0])
        m2.run_frame(frames[0])

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(config.OUTPUT_DIR, "performance_comparison.csv")

    for mode in modes:
        print(f"Benchmarking mode: {mode}")
        inf_times = []
        total_times = []

        for frame in frames:
            t0 = time.perf_counter()
            
            inf_t0 = time.perf_counter()
            if mode == "model_1_only":
                m1.run_frame(frame)
            elif mode == "model_2_only":
                m2.run_frame(frame)
            elif mode == "model_1_plus_model_2":
                m1.run_frame(frame)
                m2.run_frame(frame)
            inf_ms = (time.perf_counter() - inf_t0) * 1000
            
            total_ms = (time.perf_counter() - t0) * 1000
            
            inf_times.append(inf_ms)
            total_times.append(total_ms)

        avg_inf = statistics.mean(inf_times)
        avg_tot = statistics.mean(total_times)
        fps = 1000.0 / avg_tot if avg_tot > 0 else 0

        # Attempt to get GPU memory if possible, otherwise N/A
        gpu_mem = "N/A"
        try:
            import torch
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.memory_allocated() / (1024*1024):.1f}"
        except ImportError:
            pass

        results.append({
            "mode": mode,
            "avg_inference_ms": round(avg_inf, 2),
            "avg_total_frame_ms": round(avg_tot, 2),
            "fps": round(fps, 2),
            "gpu_memory_mb": gpu_mem
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["mode", "avg_inference_ms", "avg_total_frame_ms", "fps", "gpu_memory_mb"])
        w.writeheader()
        w.writerows(results)
    
    print(f"Benchmark saved to {out_csv}")
    for r in results:
        print(r)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=100)
    args = p.parse_args()
    run_benchmark(args.video, args.frames)
