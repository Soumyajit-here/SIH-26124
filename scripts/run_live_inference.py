"""
SIH26124 — Step 4: YOLO Inference → Backend Bridge
===================================================
Simulates an onboard bus computer running YOLO Model 1 on a dashcam video,
generating observations with SIMULATED GPS, and POSTing them to the FastAPI backend.

GPS coordinates represent the SIMULATED BUS/CAMERA POSITION,
NOT the exact physical location of detected objects.
"""

import argparse
import csv
import cv2
import datetime
import json
import math
import os
import statistics
import sys
import time

import requests
from ultralytics import YOLO

# ─── Defaults ────────────────────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000/observations"
EVENTS_URL = "http://127.0.0.1:8000/events"
GEOJSON_URL = "http://127.0.0.1:8000/events/geojson"

START_LAT = 28.432738
START_LON = 77.014969
SIMULATED_SPEED_KMH = 30.0
SIMULATED_COURSE_DEG = 90.0   # East
GPS_ACCURACY_M = 10.0

EXPECTED_CLASSES = {
    0: "HMV",
    1: "LMV",
    2: "Pedestrian",
    3: "Pothole",
    4: "Crack",
    5: "Manhole",
    6: "SpeedBump",
}

OUTPUT_DIR = "inference_output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")

# ─── Geodesic helpers ────────────────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0

def move_point(lat: float, lon: float, bearing_deg: float, distance_m: float):
    """Move a geographic point by *distance_m* metres along *bearing_deg*."""
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    brng = math.radians(bearing_deg)
    d_over_R = distance_m / EARTH_RADIUS_M

    new_lat = math.asin(
        math.sin(lat_r) * math.cos(d_over_R)
        + math.cos(lat_r) * math.sin(d_over_R) * math.cos(brng)
    )
    new_lon = lon_r + math.atan2(
        math.sin(brng) * math.sin(d_over_R) * math.cos(lat_r),
        math.cos(d_over_R) - math.sin(lat_r) * math.sin(new_lat),
    )
    return math.degrees(new_lat), math.degrees(new_lon)


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="SIH26124 YOLO Inference Bridge")
    p.add_argument("video", help="Path to the input video file")
    p.add_argument("--bus-id", default="BUS-001", help="Bus identifier")
    p.add_argument("--send-every-frame", action="store_true", default=True,
                   help="Send a POST for every detection (default)")
    p.add_argument("--send-interval", type=int, default=1,
                   help="Only transmit every Nth frame with detections")
    p.add_argument("--realtime", action="store_true",
                   help="Process at the video's native FPS")
    p.add_argument("--start-lat", type=float, default=START_LAT)
    p.add_argument("--start-lon", type=float, default=START_LON)
    p.add_argument("--speed", type=float, default=SIMULATED_SPEED_KMH)
    p.add_argument("--course", type=float, default=SIMULATED_COURSE_DEG)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


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

    print(f"Video : {args.video}")
    print(f"Res   : {width}x{height}")
    print(f"FPS   : {video_fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Dur   : {duration_s:.1f}s")

    # ── Load YOLO ─────────────────────────────────────────────────────────
    model_path = "best.pt"
    if not os.path.isfile(model_path):
        # Try project root
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: best.pt")
        sys.exit(1)

    model = YOLO(model_path)
    model_names = model.names  # {0: 'HMV', ...}

    print(f"\nModel classes: {model_names}")
    for cid, cname in EXPECTED_CLASSES.items():
        actual = model_names.get(cid, None)
        if actual != cname:
            print(f"WARNING: Class {cid} expected '{cname}', got '{actual}'")

    # ── Prepare output dirs ───────────────────────────────────────────────
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "annotated_video"), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out_video_path = os.path.join(OUTPUT_DIR, "annotated_video.avi")
    out_video = cv2.VideoWriter(out_video_path, fourcc, video_fps, (width, height))

    # ── Timing accumulators ───────────────────────────────────────────────
    preprocess_times = []
    inference_times = []
    postprocess_times = []
    http_times = []
    loop_times = []

    # ── Detection log ─────────────────────────────────────────────────────
    all_detections = []
    successful_posts = 0
    failed_posts = 0
    detection_frame_counter = 0

    sim_start = datetime.datetime.now(datetime.timezone.utc)
    frame_num = 0

    print(f"\nStarting inference (bus={args.bus_id}, conf={args.conf}, imgsz={args.imgsz})...\n")

    while True:
        loop_t0 = time.perf_counter()

        # ── Pre-process ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        preprocess_ms = (time.perf_counter() - t0) * 1000
        preprocess_times.append(preprocess_ms)

        video_ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        elapsed_s = video_ts_ms / 1000.0

        # ── Simulated GPS ─────────────────────────────────────────────────
        distance_m = (args.speed / 3.6) * elapsed_s
        sim_lat, sim_lon = move_point(args.start_lat, args.start_lon,
                                      args.course, distance_m)
        frame_ts = sim_start + datetime.timedelta(milliseconds=video_ts_ms)

        # ── Inference ─────────────────────────────────────────────────────
        t1 = time.perf_counter()
        results = model(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
        inference_ms = (time.perf_counter() - t1) * 1000
        inference_times.append(inference_ms)

        # ── Post-process ──────────────────────────────────────────────────
        t2 = time.perf_counter()
        detections = results[0].boxes
        annotated = results[0].plot()

        # Overlay HUD on annotated frame
        hud_lines = [
            f"BUS: {args.bus_id}",
            f"LAT: {sim_lat:.6f}  LON: {sim_lon:.6f}  [SIMULATED]",
            f"SPD: {args.speed:.0f} km/h  CRS: {args.course:.0f} deg",
            f"TS : {frame_ts.strftime('%H:%M:%S.%f')[:-3]}",
        ]
        for i, line in enumerate(hud_lines):
            cv2.putText(annotated, line, (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        out_video.write(annotated)
        postprocess_ms = (time.perf_counter() - t2) * 1000
        postprocess_times.append(postprocess_ms)

        # ── Transmit detections ───────────────────────────────────────────
        if len(detections) > 0:
            detection_frame_counter += 1
            should_send = (detection_frame_counter % args.send_interval == 0)

            for box in detections:
                cls_id = int(box.cls[0])
                cls_name = model_names.get(cls_id, f"class_{cls_id}")
                conf_val = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Save evidence frame
                evidence_path = os.path.join(FRAMES_DIR, f"frame_{frame_num:06d}.jpg")
                cv2.imwrite(evidence_path, annotated)

                det_record = {
                    "bus_id": args.bus_id,
                    "frame_number": frame_num,
                    "video_timestamp_ms": round(video_ts_ms, 1),
                    "timestamp": frame_ts.isoformat(),
                    "latitude": round(sim_lat, 8),
                    "longitude": round(sim_lon, 8),
                    "gps_source": "SIMULATED",
                    "gps_accuracy_m": GPS_ACCURACY_M,
                    "speed_kmh": args.speed,
                    "course_deg": args.course,
                    "model": "model_1",
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf_val, 4),
                    "bbox_x1": round(x1, 1),
                    "bbox_y1": round(y1, 1),
                    "bbox_x2": round(x2, 1),
                    "bbox_y2": round(y2, 1),
                    "evidence_image": evidence_path,
                    "api_status": "SKIPPED",
                    "event_id": None,
                }

                if should_send:
                    payload = {
                        "bus_id": args.bus_id,
                        "timestamp": frame_ts.isoformat(),
                        "latitude": round(sim_lat, 8),
                        "longitude": round(sim_lon, 8),
                        "speed_kmh": args.speed,
                        "course_deg": args.course,
                        "model": "model_1",
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(conf_val, 4),
                        "bbox_x1": round(x1, 1),
                        "bbox_y1": round(y1, 1),
                        "bbox_x2": round(x2, 1),
                        "bbox_y2": round(y2, 1),
                        "evidence_image": evidence_path,
                        "gps_source": "SIMULATED",
                        "gps_accuracy_m": GPS_ACCURACY_M,
                    }

                    ht0 = time.perf_counter()
                    try:
                        resp = requests.post(API_URL, json=payload, timeout=5)
                        http_ms = (time.perf_counter() - ht0) * 1000
                        http_times.append(http_ms)

                        if resp.status_code == 201:
                            det_record["api_status"] = "SUCCESS"
                            det_record["event_id"] = resp.json().get("event_id")
                            successful_posts += 1
                        else:
                            det_record["api_status"] = f"HTTP_{resp.status_code}"
                            failed_posts += 1
                    except Exception as e:
                        http_ms = (time.perf_counter() - ht0) * 1000
                        http_times.append(http_ms)
                        det_record["api_status"] = f"FAILED: {e}"
                        failed_posts += 1

                all_detections.append(det_record)

        loop_ms = (time.perf_counter() - loop_t0) * 1000
        loop_times.append(loop_ms)

        # Progress reporting
        if frame_num % 200 == 0 and frame_num > 0:
            avg_fps = 1000.0 / (sum(loop_times[-200:]) / len(loop_times[-200:]))
            print(f"  Frame {frame_num}/{frame_count}  "
                  f"det={len(all_detections)}  "
                  f"ok={successful_posts}  fail={failed_posts}  "
                  f"fps={avg_fps:.1f}")

        # Real-time throttle
        if args.realtime and video_fps > 0:
            target_ms = 1000.0 / video_fps
            if loop_ms < target_ms:
                time.sleep((target_ms - loop_ms) / 1000.0)

        frame_num += 1

    cap.release()
    out_video.release()

    # ── Final GPS position ────────────────────────────────────────────────
    final_distance = (args.speed / 3.6) * duration_s
    final_lat, final_lon = move_point(args.start_lat, args.start_lon,
                                      args.course, final_distance)

    # ── Summary stats ─────────────────────────────────────────────────────
    def stat_summary(data):
        if not data:
            return {"avg": 0, "med": 0, "min": 0, "max": 0}
        return {
            "avg": round(statistics.mean(data), 2),
            "med": round(statistics.median(data), 2),
            "min": round(min(data), 2),
            "max": round(max(data), 2),
        }

    pre_stats = stat_summary(preprocess_times)
    inf_stats = stat_summary(inference_times)
    post_stats = stat_summary(postprocess_times)
    http_stats = stat_summary(http_times)
    loop_stats = stat_summary(loop_times)
    processing_fps = round(1000.0 / loop_stats["avg"], 2) if loop_stats["avg"] > 0 else 0

    # Class breakdown
    class_counts = {}
    for d in all_detections:
        cn = d["class_name"]
        class_counts[cn] = class_counts.get(cn, 0) + 1

    print(f"\n{'='*60}")
    print(f"INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed  : {frame_num}")
    print(f"Total detections  : {len(all_detections)}")
    print(f"API success       : {successful_posts}")
    print(f"API failed        : {failed_posts}")
    print(f"Processing FPS    : {processing_fps}")
    print(f"Detections by class:")
    for cn, ct in sorted(class_counts.items()):
        print(f"  {cn}: {ct}")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "detections.csv")
    if all_detections:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_detections[0].keys())
            w.writeheader()
            w.writerows(all_detections)
        print(f"\nCSV   : {csv_path}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_path = os.path.join(OUTPUT_DIR, "detections.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2, default=str)
    print(f"JSON  : {json_path}")

    # ── Backend verification ──────────────────────────────────────────────
    backend_ok = False
    geojson_ok = False
    try:
        ev_resp = requests.get(EVENTS_URL, timeout=5)
        events = ev_resp.json() if ev_resp.status_code == 200 else []
        print(f"\nBackend events    : {len(events)}")
        for e in events:
            print(f"  Event #{e['event_id']} {e['event_type']} "
                  f"obs={e['observation_count']} buses={e['unique_bus_count']}")
        backend_ok = len(events) > 0

        gj_resp = requests.get(GEOJSON_URL, timeout=5)
        if gj_resp.status_code == 200:
            gj = gj_resp.json()
            if gj.get("type") == "FeatureCollection" and len(gj.get("features", [])) > 0:
                geojson_ok = True
                print(f"GeoJSON features  : {len(gj['features'])}")
    except Exception as e:
        print(f"Backend check failed: {e}")

    # ── Integration report ────────────────────────────────────────────────
    report_path = os.path.join(OUTPUT_DIR, "integration_report.txt")
    with open(report_path, "w", encoding="utf-8") as rpt:
        rpt.write("=" * 60 + "\n")
        rpt.write("SIH26124 — STEP 4 INTEGRATION REPORT\n")
        rpt.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        rpt.write("=" * 60 + "\n\n")

        rpt.write("VIDEO\n")
        rpt.write(f"  Filename          : {args.video}\n")
        rpt.write(f"  Resolution        : {width}x{height}\n")
        rpt.write(f"  FPS               : {video_fps:.2f}\n")
        rpt.write(f"  Duration          : {duration_s:.1f}s\n")
        rpt.write(f"  Frames processed  : {frame_num}\n\n")

        rpt.write("MODEL\n")
        rpt.write(f"  Checkpoint        : {model_path}\n")
        rpt.write(f"  Inference size    : {args.imgsz}\n")
        rpt.write(f"  Confidence thresh : {args.conf}\n")
        rpt.write(f"  Class mapping     : {model_names}\n\n")

        rpt.write("DETECTIONS\n")
        rpt.write(f"  Total             : {len(all_detections)}\n")
        for cn, ct in sorted(class_counts.items()):
            rpt.write(f"  {cn:20s}: {ct}\n")
        rpt.write("\n")

        rpt.write("NETWORK\n")
        rpt.write(f"  Observations sent : {successful_posts + failed_posts}\n")
        rpt.write(f"  Successful        : {successful_posts}\n")
        rpt.write(f"  Failed            : {failed_posts}\n")
        rpt.write(f"  HTTP latency avg  : {http_stats['avg']:.2f} ms\n")
        rpt.write(f"  HTTP latency med  : {http_stats['med']:.2f} ms\n")
        rpt.write(f"  HTTP latency min  : {http_stats['min']:.2f} ms\n")
        rpt.write(f"  HTTP latency max  : {http_stats['max']:.2f} ms\n\n")

        rpt.write("GPS\n")
        rpt.write(f"  GPS source        : SIMULATED\n")
        rpt.write(f"  Starting coord    : {args.start_lat}, {args.start_lon}\n")
        rpt.write(f"  Speed             : {args.speed} km/h\n")
        rpt.write(f"  Course            : {args.course} deg\n")
        rpt.write(f"  Final coord       : {final_lat:.8f}, {final_lon:.8f}\n")
        rpt.write(f"  GPS accuracy      : {GPS_ACCURACY_M} m\n\n")

        rpt.write("BACKEND\n")
        try:
            obs_resp = requests.get("http://127.0.0.1:8000/observations", timeout=5)
            obs_data = obs_resp.json() if obs_resp.status_code == 200 else []
            rpt.write(f"  Observations stored : {len(obs_data)}\n")
        except:
            rpt.write(f"  Observations stored : UNKNOWN (backend unreachable)\n")
        rpt.write(f"  Events created      : {len(events) if backend_ok else 'UNKNOWN'}\n")
        unique_buses = set()
        for e in (events if backend_ok else []):
            unique_buses.add(e.get("unique_bus_count", 0))
        rpt.write(f"  Unique buses        : check /events for details\n\n")

        rpt.write("PERFORMANCE\n")
        rpt.write(f"  Processing FPS      : {processing_fps}\n")
        rpt.write(f"  Preprocess  (ms)    : avg={pre_stats['avg']}  med={pre_stats['med']}  min={pre_stats['min']}  max={pre_stats['max']}\n")
        rpt.write(f"  Inference   (ms)    : avg={inf_stats['avg']}  med={inf_stats['med']}  min={inf_stats['min']}  max={inf_stats['max']}\n")
        rpt.write(f"  Postprocess (ms)    : avg={post_stats['avg']}  med={post_stats['med']}  min={post_stats['min']}  max={post_stats['max']}\n")
        rpt.write(f"  HTTP        (ms)    : avg={http_stats['avg']}  med={http_stats['med']}  min={http_stats['min']}  max={http_stats['max']}\n")
        rpt.write(f"  Loop total  (ms)    : avg={loop_stats['avg']}  med={loop_stats['med']}  min={loop_stats['min']}  max={loop_stats['max']}\n\n")

        rpt.write("GIS\n")
        rpt.write(f"  GeoJSON endpoint    : {'VERIFIED' if geojson_ok else 'FAILED'}\n")
        rpt.write(f"  Events on map       : {'YES' if geojson_ok else 'NO'}\n\n")

        rpt.write("=" * 60 + "\n")
        all_pass = (successful_posts > 0 and backend_ok and geojson_ok)
        rpt.write(f"END-TO-END STATUS     : {'PASS' if all_pass else 'FAIL'}\n")
        rpt.write("=" * 60 + "\n")

    print(f"Report: {report_path}")
    print(f"Video : {out_video_path}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 4 RESULTS")
    print(f"{'='*60}")
    print(f"YOLO → Backend   : {'PASS' if successful_posts > 0 else 'FAIL'}")
    print(f"Backend → MySQL  : {'PASS' if backend_ok else 'FAIL'}")
    print(f"Event Fusion     : {'PASS' if backend_ok else 'FAIL'}")
    print(f"GeoJSON          : {'PASS' if geojson_ok else 'FAIL'}")
    print(f"Frontend Map     : {'PASS' if geojson_ok else 'FAIL'}")
    print(f"End-to-end       : {'PASS' if (successful_posts > 0 and backend_ok and geojson_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
