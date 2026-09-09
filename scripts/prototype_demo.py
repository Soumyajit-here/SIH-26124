from __future__ import annotations

import argparse
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

# Manhole is suppressed ONLY for this demonstration.
DISABLED_CLASSES = {"Manhole"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SIH26124 Model 1 local inference demo"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video"
    )

    parser.add_argument(
        "--model",
        default="best.pt",
        help="Path to Model 1 best.pt"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference size"
    )

    parser.add_argument(
        "--output",
        default="prototype_demo_output",
        help="Output folder"
    )

    # Initial size only; window remains freely resizable.
    parser.add_argument(
        "--width",
        type=int,
        default=1000,
        help="Initial display window width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=700,
        help="Initial display window height"
    )

    parser.add_argument(
        "--wait",
        type=int,
        default=1,
        help="OpenCV waitKey delay in milliseconds"
    )

    return parser.parse_args()


def normalize_names(names):
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}

    raise RuntimeError("Unable to read model class mapping.")


def verify_model(model):
    actual = normalize_names(model.names)

    print("\n" + "=" * 75)
    print("MODEL 1 CLASS VERIFICATION")
    print("=" * 75)

    for class_id, expected_name in EXPECTED_CLASSES.items():
        actual_name = actual.get(class_id)

        print(
            f"{class_id}: "
            f"{actual_name if actual_name is not None else 'MISSING'}"
        )

        if actual_name != expected_name:
            raise RuntimeError(
                f"Class mismatch: ID {class_id}. "
                f"Expected '{expected_name}', got '{actual_name}'."
            )

    print("-" * 75)
    print("MODEL 1 VERIFIED")
    print()
    print("Demo configuration:")
    print("  Model 1 : ENABLED")
    print("  Model 2 : DISABLED")
    print("  Manhole : DISABLED")
    print("  GPS     : DISABLED")
    print("  Backend : DISABLED")
    print("  Map     : DISABLED")
    print("=" * 75)


def draw_detection(
    frame,
    x1,
    y1,
    x2,
    y2,
    class_name,
    confidence
):
    # Model 1 visual style
    color = (0, 255, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    label = f"{class_name} {confidence:.2f}"

    (tw, th), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    top = max(0, y1 - th - baseline - 6)
    bottom = top + th + baseline + 6

    cv2.rectangle(
        frame,
        (x1, top),
        (x1 + tw + 8, bottom),
        color,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, bottom - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        2,
        cv2.LINE_AA
    )


def main():
    args = parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found:\n{video_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    print("\n")
    print("=" * 80)
    print("      BHARATPOTHOLE — MODEL 1 LOCAL INFERENCE DEMO")
    print("=" * 80)

    print("\nLoading Model 1...")
    model = YOLO(str(model_path))

    verify_model(model)

    print("\nOpening video...")

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video:\n{video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration = (
        total_frames / fps
        if total_frames > 0
        else 0
    )

    print("\nVIDEO INFORMATION")
    print("-" * 60)
    print(f"File       : {video_path}")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.2f}")
    print(f"Frames     : {total_frames}")
    print(f"Duration   : {duration:.2f} sec")
    print("-" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir / "model1_demo.avi"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not create output video:\n{output_path}"
        )

    # Freely resizable OpenCV window.
    window_name = (
        "BharatPotHole - Model 1 Live Inference"
    )

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    )

    cv2.resizeWindow(
        window_name,
        args.width,
        args.height
    )

    frame_number = 0
    total_detections = 0

    final_totals = {
        "HMV": 0,
        "LMV": 0,
        "Pedestrian": 0,
        "Pothole": 0,
        "Crack": 0,
        "SpeedBump": 0,
    }

    overall_start = time.perf_counter()

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            inference_start = time.perf_counter()

            annotated = frame.copy()

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                conf=args.conf,
                verbose=False
            )

            result = results[0]

            frame_predictions = []

            if result.boxes is not None:

                names = normalize_names(
                    result.names
                )

                for box in result.boxes:

                    class_id = int(
                        box.cls[0].item()
                    )

                    confidence = float(
                        box.conf[0].item()
                    )

                    class_name = names.get(
                        class_id,
                        f"class_{class_id}"
                    )

                    # ----------------------------------------
                    # TEMPORARY MANHOLE SUPPRESSION
                    # ----------------------------------------
                    if class_name in DISABLED_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    frame_predictions.append(
                        {
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": (
                                x1,
                                y1,
                                x2,
                                y2
                            ),
                        }
                    )

                    draw_detection(
                        annotated,
                        x1,
                        y1,
                        x2,
                        y2,
                        class_name,
                        confidence
                    )

                    total_detections += 1

                    if class_name in final_totals:
                        final_totals[class_name] += 1

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            current_fps = (
                1.0 / inference_time
                if inference_time > 0
                else 0.0
            )

            video_time = frame_number / fps

            # ====================================================
            # TERMINAL — PRINT EVERY SINGLE FRAME
            # ====================================================

            print(
                f"\nFRAME {frame_number:05d}/{total_frames}"
                f" | TIME {video_time:7.2f}s"
                f" | INFERENCE FPS {current_fps:6.2f}"
            )

            if not frame_predictions:

                print(
                    "  → No active detections"
                )

            else:

                print(
                    f"  → {len(frame_predictions)} detection(s)"
                )

                for i, prediction in enumerate(
                    frame_predictions,
                    start=1
                ):

                    x1, y1, x2, y2 = (
                        prediction["bbox"]
                    )

                    print(
                        f"     [{i}] "
                        f"{prediction['class_name']:<12}"
                        f" | confidence="
                        f"{prediction['confidence']:.2f}"
                        f" | bbox="
                        f"({x1},{y1},{x2},{y2})"
                    )

            # ====================================================
            # DISPLAY HUD
            # ====================================================

            panel_width = min(
                width - 20,
                500
            )

            panel_height = 120

            cv2.rectangle(
                annotated,
                (10, 10),
                (
                    panel_width,
                    panel_height
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                "BharatPotHole | MODEL 1",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                f"Frame: {frame_number}/{total_frames}",
                (20, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                (
                    f"Time: {video_time:.2f}s | "
                    f"FPS: {current_fps:.1f}"
                ),
                (20, 92),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                annotated,
                (
                    f"Detections: "
                    f"{len(frame_predictions)}"
                ),
                (20, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ====================================================
            # SAVE + DISPLAY
            # ====================================================

            writer.write(annotated)

            cv2.imshow(
                window_name,
                annotated
            )

            # Q / ESC = stop
            key = cv2.waitKey(
                max(1, args.wait)
            ) & 0xFF

            if key == ord("q") or key == 27:
                print(
                    "\nInference stopped by user."
                )
                break

    finally:

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_time = (
        time.perf_counter()
        - overall_start
    )

    overall_fps = (
        frame_number / total_time
        if total_time > 0
        else 0.0
    )

    print("\n")
    print("=" * 80)
    print("                DEMONSTRATION COMPLETE")
    print("=" * 80)

    print(
        f"Frames processed : {frame_number}"
    )

    print(
        f"Total detections : {total_detections}"
    )

    print(
        f"Overall FPS      : {overall_fps:.2f}"
    )

    print("\nDETECTIONS BY CLASS")
    print("-" * 50)

    for class_name, count in final_totals.items():
        print(
            f"{class_name:<15}: {count}"
        )

    print("\nDEMO CONFIGURATION")
    print("-" * 50)
    print("Model 1          : ENABLED")
    print("Model 2          : DISABLED")
    print("Manhole          : DISABLED")
    print("GPS              : DISABLED")
    print("FastAPI          : DISABLED")
    print("MySQL            : DISABLED")
    print("GIS              : DISABLED")

    print("\nOUTPUT")
    print("-" * 50)
    print(
        f"Annotated video  : {output_path}"
    )

    print("\nThe inference window is freely resizable.")
    print("Drag any corner/edge of the window while running.")
    print("=" * 80)


if __name__ == "__main__":
    main()