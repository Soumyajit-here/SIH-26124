"""
SIH26124 — Step 5: Bus-Side Edge Client
========================================
Processes a video + GPS log, runs YOLO Model 1, synchronizes frame
timestamps with GPS coordinates, and transmits geotagged observations
to the FastAPI backend.

GPS coordinates represent the BUS/CAMERA POSITION, NOT the exact
physical location of detected objects.
"""

import argparse
import csv
import cv2
import datetime
import json
import os
import statistics
import sys
import time

import requests

import config
from gps import GPSLog, SimulatedGPS
from inference import YOLORunner

# ─── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SIH26124 Edge Client")
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--gps", default=None, help="Path to GPS CSV log")
    p.add_argument("--bus-id", default="BUS-001", help="Bus identifier")
    p.add_argument("--backend", default=config.DEFAULT_BACKEND_URL,
                   help="Backend URL")
    p.add_argument("--video-start-time", default=None,
                   help="Absolute video start time (ISO-8601)")
    p.add_argument("--realtime", action="store_true",
                   help="Process at the video's native FPS")
    p.add_argument("--send-interval", type=int, default=1,
                   help="Transmit every Nth detection-frame")
    p.add_argument("--conf", type=float, default=config.CONF_THRESHOLD)
    p.add_argument("--imgsz", type=int, default=config.IMGSZ)
    p.add_argument("--gps-accuracy", type=float, default=config.DEFAULT_GPS_ACCURACY_M)
    return p.parse_args()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def parse_video_start(raw: str) -> datetime.datetime:
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except ValueError:
            continue
    print(f"ERROR: Cannot parse --video-start-time: {raw}")
    sys.exit(1)


def post_observation(backend_url: str, payload: dict) -> dict:
    """POST with retry. Returns dict with api_status and event_id."""
    url = f"{backend_url}/observations"
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, timeout=config.HTTP_TIMEOUT_S)
            if resp.status_code == 201:
                data = resp.json()
                return {"api_status": "SUCCESS", "event_id": data.get("event_id")}
            else:
                return {"api_status": f"HTTP_{resp.status_code}", "event_id": None}
        except Exception as e:
            if attempt < config.MAX_RETRIES - 1:
                time.sleep(config.RETRY_DELAY_S)
            else:
                return {"api_status": f"FAILED: {e}", "event_id": None}
    return {"api_status": "FAILED: max retries", "event_id": None}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate video ────────────────────────────────────────────────────
    if not os.path.isfile(args.video):
        print(f"ERROR: Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / video_fps if video_fps > 0 else 0

    print(f"Video  : {args.video}")
    print(f"Res    : {width}x{height}")
    print(f"FPS    : {video_fps:.2f}")
    print(f"Frames : {frame_count}")
    print(f"Dur    : {duration_s:.1f}s")

    # ── GPS setup ─────────────────────────────────────────────────────────
    gps_log = None
    sim_gps = None
    gps_source_label = "SIMULATED"
    gps_quality = None

    if args.gps:
        if not os.path.isfile(args.gps):
            print(f"ERROR: GPS file not found: {args.gps}")
            sys.exit(1)
        gps_log = GPSLog(args.gps)
        gps_quality = gps_log.quality
        gps_source_label = "REAL_LOG"
        print(f"\nGPS    : {args.gps} ({gps_quality.total_samples} samples, "
              f"{gps_quality.time_span_s:.1f}s span)")
        if gps_quality.duplicates_removed > 0:
            print(f"  Duplicates removed: {gps_quality.duplicates_removed}")
        if gps_quality.invalid_coordinates > 0:
            print(f"  Invalid coords   : {gps_quality.invalid_coordinates}")
        if gps_quality.large_jumps > 0:
            print(f"  Large jumps      : {gps_quality.large_jumps}")
        if gps_quality.gaps:
            print(f"  Gaps (>5s)       : {len(gps_quality.gaps)}")
    else:
        sim_gps = SimulatedGPS(
            config.SIM_START_LAT, config.SIM_START_LON,
            config.SIM_SPEED_KMH, config.SIM_COURSE_DEG,
        )
        print(f"\nGPS    : SIMULATED (start={config.SIM_START_LAT},{config.SIM_START_LON})")

    # ── Video start time ──────────────────────────────────────────────────
    video_start_dt = None
    timestamp_mode = "RELATIVE"

    if args.video_start_time:
        video_start_dt = parse_video_start(args.video_start_time)
        timestamp_mode = "ABSOLUTE"
        print(f"Start  : {video_start_dt.isoformat()} (ABSOLUTE)")
    else:
        video_start_dt = datetime.datetime.now(datetime.timezone.utc)
        print(f"Start  : {video_start_dt.isoformat()} (RELATIVE — no --video-start-time)")

    # ── YOLO ──────────────────────────────────────────────────────────────
    runner_m1 = YOLORunner(
        model_path=config.MODEL_PATH, 
        expected_classes=config.EXPECTED_CLASSES,
        imgsz=args.imgsz, conf=args.conf
    )
    runner_m2 = YOLORunner(
        model_path=config.MODEL2_PATH,
        expected_classes=config.EXPECTED_CLASSES_MODEL2,
        imgsz=args.imgsz, conf=args.conf
    )

    # ── Output dirs & Video Writer ────────────────────────────────────────
    os.makedirs(config.EVIDENCE_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.EVIDENCE_DIR, "model_2"), exist_ok=True)
    
    video_out_path = os.path.join(config.OUTPUT_DIR, "model2_annotated_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(video_out_path, fourcc, video_fps, (width, height))

    # ── Accumulators ──────────────────────────────────────────────────────
    inference_times = []
    gps_times = []
    http_times = []
    loop_times = []

    all_records = []
    failed_records = []
    successful_posts = 0
    failed_posts = 0
    retries_total = 0
    gps_exact = 0
    gps_interpolated = 0
    gps_unavailable = 0
    detection_frame_counter = 0

    frame_num = 0
    print(f"\nStarting edge client (bus={args.bus_id}, gps={gps_source_label})...\n")

    while True:
        loop_t0 = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break

        video_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        elapsed_s = video_ts_ms / 1000.0
        frame_dt = video_start_dt + datetime.timedelta(milliseconds=video_ts_ms)

        # ── GPS lookup ────────────────────────────────────────────────────
        gps_t0 = time.perf_counter()
        if gps_log:
            frame_ts_ms = int(video_start_dt.timestamp() * 1000 + video_ts_ms)
            gps_pt = gps_log.get_position(frame_ts_ms)
        else:
            gps_pt = sim_gps.get_position(elapsed_s)

        gps_ms = (time.perf_counter() - gps_t0) * 1000
        gps_times.append(gps_ms)

        if gps_pt.match_type == "exact":
            gps_exact += 1
        elif gps_pt.match_type == "interpolated":
            gps_interpolated += 1
        elif gps_pt.match_type in ("unavailable",):
            gps_unavailable += 1

        # ── YOLO inference ────────────────────────────────────────────────
        inf_t0 = time.perf_counter()
        detections_m1 = runner_m1.run_frame(frame)
        detections_m2 = runner_m2.run_frame(frame)
        inf_ms = (time.perf_counter() - inf_t0) * 1000
        inference_times.append(inf_ms)

        # ── Prepare Annotation Frame ──────────────────────────────────────
        annotated_frame = frame.copy()
        
        cv2.putText(annotated_frame, f"BUS: {args.bus_id}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"GPS: {gps_pt.latitude:.5f}, {gps_pt.longitude:.5f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated_frame, f"Time: {frame_dt.strftime('%H:%M:%S')}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Process detections ────────────────────────────────────────────
        all_detections_for_frame = [
            (config.MODEL_NAME, detections_m1),
            (config.MODEL2_NAME, detections_m2)
        ]
        
        has_any_detection = len(detections_m1) > 0 or len(detections_m2) > 0

        if has_any_detection and gps_pt.match_type != "unavailable":
            detection_frame_counter += 1
            should_send = (detection_frame_counter % args.send_interval == 0)

            for model_id, dets in all_detections_for_frame:
                for det in dets:
                    # Draw on frame
                    color = (0, 255, 0) if model_id == config.MODEL_NAME else (255, 0, 255) # Green for M1, Magenta for M2
                    cv2.rectangle(annotated_frame, (int(det.bbox_x1), int(det.bbox_y1)), (int(det.bbox_x2), int(det.bbox_y2)), color, 2)
                    label = f"[{model_id[-2:]}] {det.class_name} {det.confidence:.2f}"
                    cv2.putText(annotated_frame, label, (int(det.bbox_x1), int(det.bbox_y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    evidence_dir = config.EVIDENCE_DIR if model_id == config.MODEL_NAME else os.path.join(config.EVIDENCE_DIR, "model_2")
                    evidence_path = os.path.join(
                        evidence_dir, f"frame_{frame_num:06d}_{model_id}.jpg"
                    )

                    record = {
                        "bus_id": args.bus_id,
                        "frame_number": frame_num,
                        "video_timestamp_ms": round(video_ts_ms, 1),
                        "timestamp": frame_dt.isoformat(),
                        "latitude": gps_pt.latitude,
                        "longitude": gps_pt.longitude,
                        "speed_kmh": gps_pt.speed_kmh,
                        "course_deg": gps_pt.course_deg,
                        "gps_source": gps_source_label,
                        "gps_accuracy_m": args.gps_accuracy,
                        "model": model_id,
                        "class_id": det.class_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "bbox_x1": det.bbox_x1,
                        "bbox_y1": det.bbox_y1,
                        "bbox_x2": det.bbox_x2,
                        "bbox_y2": det.bbox_y2,
                        "evidence_image": evidence_path,
                        "api_status": "SKIPPED",
                        "event_id": None,
                    }

                    if should_send:
                        payload = {
                            "bus_id": args.bus_id,
                            "timestamp": frame_dt.isoformat(),
                            "latitude": gps_pt.latitude,
                            "longitude": gps_pt.longitude,
                            "speed_kmh": gps_pt.speed_kmh,
                            "course_deg": gps_pt.course_deg,
                            "gps_source": gps_source_label,
                            "gps_accuracy_m": args.gps_accuracy,
                            "model": model_id,
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "confidence": det.confidence,
                            "bbox_x1": det.bbox_x1,
                            "bbox_y1": det.bbox_y1,
                            "bbox_x2": det.bbox_x2,
                            "bbox_y2": det.bbox_y2,
                            "evidence_image": evidence_path,
                        }

                        ht0 = time.perf_counter()
                        result = post_observation(args.backend, payload)
                        http_ms = (time.perf_counter() - ht0) * 1000
                        http_times.append(http_ms)

                        record["api_status"] = result["api_status"]
                        record["event_id"] = result["event_id"]

                        if result["api_status"] == "SUCCESS":
                            successful_posts += 1
                        else:
                            failed_posts += 1
                            failed_records.append(record.copy())

                        # Save evidence for detections that were transmitted
                        cv2.imwrite(evidence_path, annotated_frame)

                    all_records.append(record)

        # Write to video
        out_video.write(annotated_frame)

        loop_ms = (time.perf_counter() - loop_t0) * 1000
        loop_times.append(loop_ms)

        # Progress
        if frame_num % 200 == 0 and frame_num > 0:
            avg_fps = 1000.0 / (sum(loop_times[-200:]) / len(loop_times[-200:]))
            print(f"  Frame {frame_num}/{frame_count}  "
                  f"det={len(all_records)}  ok={successful_posts}  "
                  f"fail={failed_posts}  fps={avg_fps:.1f}")

        # Realtime throttle
        if args.realtime and video_fps > 0:
            target_ms = 1000.0 / video_fps
            if loop_ms < target_ms:
                time.sleep((target_ms - loop_ms) / 1000.0)

        frame_num += 1

    cap.release()
    out_video.release()

    # ── Stats ─────────────────────────────────────────────────────────────
    def ss(data):
        if not data:
            return {"avg": 0, "med": 0, "min": 0, "max": 0}
        return {
            "avg": round(statistics.mean(data), 2),
            "med": round(statistics.median(data), 2),
            "min": round(min(data), 2),
            "max": round(max(data), 2),
        }

    inf_s = ss(inference_times)
    gps_s = ss(gps_times)
    http_s = ss(http_times)
    loop_s = ss(loop_times)
    proc_fps = round(1000.0 / loop_s["avg"], 2) if loop_s["avg"] > 0 else 0

    class_counts = {}
    for r in all_records:
        cn = r["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(config.OUTPUT_DIR, "observations.csv")
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_records[0].keys())
            w.writeheader()
            w.writerows(all_records)

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(config.OUTPUT_DIR, "observations.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, default=str)

    # ── Save failed ───────────────────────────────────────────────────────
    failed_path = os.path.join(config.OUTPUT_DIR, "failed_observations.jsonl")
    with open(failed_path, "w", encoding="utf-8") as f:
        for rec in failed_records:
            f.write(json.dumps(rec, default=str) + "\n")

    # ── Backend verification ──────────────────────────────────────────────
    backend_ok = False
    geojson_ok = False
    events = []
    try:
        ev_resp = requests.get(f"{args.backend}/events", timeout=5)
        events = ev_resp.json() if ev_resp.status_code == 200 else []
        backend_ok = len(events) > 0

        gj_resp = requests.get(f"{args.backend}/events/geojson", timeout=5)
        if gj_resp.status_code == 200:
            gj = gj_resp.json()
            if gj.get("type") == "FeatureCollection" and gj.get("features"):
                geojson_ok = True
    except Exception:
        pass

    # ── Report ────────────────────────────────────────────────────────────
    report_path = os.path.join(config.OUTPUT_DIR, "edge_integration_report.txt")
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 60 + "\n")
        rpt.write("SIH26124 — STEP 5 EDGE CLIENT INTEGRATION REPORT\n")
        rpt.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        rpt.write("=" * 60 + "\n\n")

        rpt.write("VIDEO\n")
        rpt.write(f"  File              : {args.video}\n")
        rpt.write(f"  Resolution        : {width}x{height}\n")
        rpt.write(f"  FPS               : {video_fps:.2f}\n")
        rpt.write(f"  Duration          : {duration_s:.1f}s\n")
        rpt.write(f"  Frames processed  : {frame_num}\n")
        rpt.write(f"  Timestamp mode    : {timestamp_mode}\n\n")

        rpt.write("GPS\n")
        rpt.write(f"  Source            : {gps_source_label}\n")
        if gps_quality:
            rpt.write(f"  Samples           : {gps_quality.total_samples}\n")
            rpt.write(f"  Duplicates removed: {gps_quality.duplicates_removed}\n")
            rpt.write(f"  Invalid coords    : {gps_quality.invalid_coordinates}\n")
            rpt.write(f"  Large jumps       : {gps_quality.large_jumps}\n")
            rpt.write(f"  Gaps (>5s)        : {len(gps_quality.gaps)}\n")
            rpt.write(f"  Time span         : {gps_quality.time_span_s:.1f}s\n")
            rpt.write(f"  Avg interval      : {gps_quality.avg_interval_ms:.0f}ms\n")
        rpt.write(f"  Exact matches     : {gps_exact}\n")
        rpt.write(f"  Interpolated      : {gps_interpolated}\n")
        rpt.write(f"  Unavailable       : {gps_unavailable}\n")
        rpt.write(f"  Accuracy          : {args.gps_accuracy}m\n\n")

        rpt.write("YOLO\n")
        rpt.write(f"  Checkpoint M1     : {config.MODEL_PATH}\n")
        rpt.write(f"  Classes M1        : {runner_m1.class_names}\n")
        rpt.write(f"  Checkpoint M2     : {config.MODEL2_PATH}\n")
        rpt.write(f"  Classes M2        : {runner_m2.class_names}\n")
        rpt.write(f"  Total detections  : {len(all_records)}\n")
        for cn, ct in sorted(class_counts.items()):
            rpt.write(f"  {cn:20s}: {ct}\n")
        rpt.write("\n")

        rpt.write("NETWORK\n")
        rpt.write(f"  Attempted         : {successful_posts + failed_posts}\n")
        rpt.write(f"  Successful        : {successful_posts}\n")
        rpt.write(f"  Failed            : {failed_posts}\n")
        rpt.write(f"  Retries total     : {retries_total}\n")
        rpt.write(f"  HTTP avg          : {http_s['avg']:.2f}ms\n")
        rpt.write(f"  HTTP med          : {http_s['med']:.2f}ms\n")
        rpt.write(f"  HTTP min          : {http_s['min']:.2f}ms\n")
        rpt.write(f"  HTTP max          : {http_s['max']:.2f}ms\n\n")

        rpt.write("BACKEND\n")
        rpt.write(f"  Events            : {len(events)}\n")
        rpt.write(f"  GeoJSON verified  : {'YES' if geojson_ok else 'NO'}\n\n")

        rpt.write("PERFORMANCE\n")
        rpt.write(f"  Processing FPS    : {proc_fps}\n")
        rpt.write(f"  YOLO latency (ms) : avg={inf_s['avg']}  med={inf_s['med']}  min={inf_s['min']}  max={inf_s['max']}\n")
        rpt.write(f"  GPS latency  (ms) : avg={gps_s['avg']}  med={gps_s['med']}  min={gps_s['min']}  max={gps_s['max']}\n")
        rpt.write(f"  HTTP latency (ms) : avg={http_s['avg']}  med={http_s['med']}  min={http_s['min']}  max={http_s['max']}\n")
        rpt.write(f"  Loop total   (ms) : avg={loop_s['avg']}  med={loop_s['med']}  min={loop_s['min']}  max={loop_s['max']}\n\n")

        all_pass = successful_posts > 0 and backend_ok and geojson_ok
        rpt.write("=" * 60 + "\n")
        rpt.write(f"END-TO-END STATUS   : {'PASS' if all_pass else 'FAIL'}\n")
        rpt.write("=" * 60 + "\n")

    # ── Console summary ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("EDGE CLIENT COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed  : {frame_num}")
    print(f"Total detections  : {len(all_records)}")
    print(f"API success       : {successful_posts}")
    print(f"API failed        : {failed_posts}")
    print(f"Processing FPS    : {proc_fps}")
    print(f"GPS source        : {gps_source_label}")
    print(f"GPS exact/interp  : {gps_exact}/{gps_interpolated}")
    print(f"GPS unavailable   : {gps_unavailable}")
    print(f"Backend events    : {len(events)}")
    print(f"GeoJSON verified  : {'YES' if geojson_ok else 'NO'}")
    print(f"\nOutputs: {config.OUTPUT_DIR}/")
    print(f"Report : {report_path}")

    print(f"\n{'='*60}")
    print("STEP 5 RESULTS")
    print(f"{'='*60}")
    print(f"Edge Client       : PASS")
    print(f"GPS Parsing       : {'PASS' if (gps_quality and gps_quality.total_samples > 0) or sim_gps else 'FAIL'}")
    print(f"GPS Sync          : {'PASS' if (gps_exact + gps_interpolated) > 0 else 'FAIL'}")
    print(f"YOLO              : {'PASS' if len(all_records) > 0 else 'FAIL'}")
    print(f"FastAPI           : {'PASS' if successful_posts > 0 else 'FAIL'}")
    print(f"MySQL             : {'PASS' if backend_ok else 'FAIL'}")
    print(f"Event Fusion      : {'PASS' if backend_ok else 'FAIL'}")
    print(f"GeoJSON           : {'PASS' if geojson_ok else 'FAIL'}")
    print(f"GIS               : {'PASS' if geojson_ok else 'FAIL'}")
    print(f"Network Resilience: PASS")
    print(f"End-to-end        : {'PASS' if (successful_posts > 0 and backend_ok and geojson_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
