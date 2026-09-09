"""
run_live_end_to_end.py
======================
End-to-End Live Video Intelligence & Risk Prevention Engine.
Combines Person A (Perception, Skeletons, Box Tracking) with
Person B (Kinematics, Predictive Near-Miss, Dynamic Risk HUD).

Controls:
- Press 'q' or 'ESC' in the live window to STOP and save deliverables.
- Press 's' to take a snapshot of the current perception + risk state.

Usage:
    python run_live_end_to_end.py
    python run_live_end_to_end.py --camera 0
    python run_live_end_to_end.py --video path/to/video.mp4
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
from pipeline_person_a import PersonAPipeline, KEYPOINT_NAMES, SKELETON_PAIRS
from pipeline_person_b import RiskEngine


def draw_hud_banner(vis_frame, active_alert: Optional[Dict], frame_idx: int, fps: float, n_persons: int, n_boxes: int):
    """Renders the real-time AI supervisor HUD bar and risk alert banners."""
    h, w, _ = vis_frame.shape

    # 1. Top Status HUD Bar
    cv2.rectangle(vis_frame, (0, 0), (w, 36), (20, 20, 20), -1)
    cv2.line(vis_frame, (0, 36), (w, 36), (60, 60, 60), 1)

    hud_text = f"GODREJ AI FIELD INTELLIGENCE | FPS: {fps:4.1f} | WORKERS: {n_persons} | BOXES: {n_boxes} | FRAME: {frame_idx}"
    cv2.putText(vis_frame, hud_text, (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

    # 2. Real-Time Risk Alert Banner
    if active_alert and active_alert.get("frames_left", 0) > 0:
        level = active_alert.get("risk_level", "Medium")
        btype = active_alert.get("behaviour_type", "Safety Alert")
        action = active_alert.get("action", "Follow safe handling guidelines.")
        prob = active_alert.get("near_miss_prob", 0.0)

        # Color coding per risk level
        if level == "Critical":
            bg_color = (0, 0, 180)       # Bright Crimson Red
            txt_color = (255, 255, 255)
            tag = "CRITICAL ALERT"
        elif level == "High":
            bg_color = (0, 120, 230)      # Amber / Orange
            txt_color = (255, 255, 255)
            tag = "HIGH RISK"
        elif level == "Medium":
            bg_color = (0, 180, 220)      # Yellow / Ochre
            txt_color = (20, 20, 20)
            tag = "RISK WARNING"
        else:
            bg_color = (30, 140, 50)      # Forest Green
            txt_color = (255, 255, 255)
            tag = "BENCHMARK"

        # Banner Dimensions
        banner_h = 65
        banner_y1 = h - banner_h - 15
        banner_y2 = h - 15
        cv2.rectangle(vis_frame, (20, banner_y1), (w - 20, banner_y2), bg_color, -1)
        cv2.rectangle(vis_frame, (20, banner_y1), (w - 20, banner_y2), (255, 255, 255), 2)

        # Title line
        if prob > 0.0:
            title_str = f"[{tag}] {btype.upper()} (Near-Miss Risk: {int(prob * 100)}%)"
        else:
            title_str = f"[{tag}] {btype.upper()}"
        cv2.putText(vis_frame, title_str, (35, banner_y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, txt_color, 2, cv2.LINE_AA)

        # Action / Guidance line
        action_str = f"RECOMMENDATION: {action}"
        if len(action_str) > 95:
            action_str = action_str[:92] + "..."
        cv2.putText(vis_frame, action_str, (35, banner_y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.48, txt_color, 1, cv2.LINE_AA)


def run_live_pipeline(source: Any = 0, output_dir: str = "outputs_live", box_conf: float = 0.15):
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    is_cam = isinstance(source, int)

    video_out_path = os.path.join(output_dir, f"live_session_{timestamp_str}_perception.mp4")
    tracking_json_path = os.path.join(output_dir, f"live_session_{timestamp_str}_tracking.json")
    events_json_path = os.path.join(output_dir, f"live_session_{timestamp_str}_warehouse_events.json")

    print("=" * 75)
    print("PERSON A + B: INTEGRATED LIVE VIDEO INTELLIGENCE & RISK PREVENTION")
    print(f"Video Source      : {'Webcam Index ' + str(source) if is_cam else source}")
    print(f"Box Confidence    : {box_conf}")
    print(f"Output Directory  : {output_dir}")
    print("=" * 75)

    # 1. Connect to Video / Camera Stream
    if is_cam:
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[Warning] Failed with DirectShow backend. Retrying default backend on index {source}...")
            cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        print("Please check camera connections, permissions, or video filepath.")
        return

    # Set camera resolution
    if is_cam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 60:
        fps = 30.0

    print(f"[Stream Initialized] Resolution: {width}x{height} @ {fps:.1f} FPS")

    # 2. Initialize Pipeline Engines
    print("Loading Person A Perception Models (YOLO-Pose + Box Tracker)...")
    pipeline_a = PersonAPipeline(box_conf=box_conf, person_conf=0.35)

    print("Initializing Person B Risk Engine & Kinematic Window...")
    risk_engine = RiskEngine(video_id=f"live_session_{timestamp_str}.mp4", fps=fps, resolution=[width, height])

    # Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    # Tracking Data Storage (Person A Contract)
    tracking_data = {
        "session_metadata": {
            "source": f"Live Camera {source}" if is_cam else os.path.basename(str(source)),
            "resolution": [width, height],
            "fps": round(fps, 2),
            "started_at": timestamp_str
        },
        "frames": []
    }

    print("\n" + "=" * 75)
    print("LIVE SYSTEM RUNNING!")
    print("  -> Press 'q' or 'ESC' on the live window to STOP and save.")
    print("  -> Press 's' to take a snapshot.")
    print("=" * 75 + "\n")

    frame_idx = 0
    t_start = time.time()
    last_fps_time = time.time()
    fps_display = float(fps)
    active_alert: Optional[Dict] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n[End of Stream] Video finished or camera disconnected.")
                break

            frame_idx += 1
            now = time.time()
            timestamp_sec = round(frame_idx / fps, 3)

            # FPS calculation
            if frame_idx % 10 == 0:
                elapsed = now - last_fps_time
                if elapsed > 0:
                    fps_display = 10.0 / elapsed
                last_fps_time = now

            # -----------------------------------------------------------------
            # Step 1: Person A - Worker Detection & Keypoint Extraction
            # -----------------------------------------------------------------
            pose_results = pipeline_a.pose_model.track(
                source=frame,
                persist=True,
                tracker=pipeline_a.tracker_config,
                conf=pipeline_a.person_conf,
                verbose=False
            )

            person_tracks = []
            if pose_results and len(pose_results) > 0:
                r = pose_results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    boxes = r.boxes
                    kps_data = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None

                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else (i + 1)
                        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]

                        p_entry = {
                            "track_id": track_id,
                            "class": "person",
                            "bbox": [x1, y1, x2, y2],
                            "bbox_normalized": [round(x1 / width, 3), round(y1 / height, 3), round(x2 / width, 3), round(y2 / height, 3)],
                            "confidence": round(conf, 3)
                        }

                        if kps_data is not None and i < len(kps_data):
                            kp_dict = {}
                            for k_idx, (kx, ky, kc) in enumerate(kps_data[i]):
                                kp_dict[KEYPOINT_NAMES[k_idx]] = [
                                    round(float(kx), 1),
                                    round(float(ky), 1),
                                    round(float(kc), 3)
                                ]
                            p_entry["keypoints"] = kp_dict

                        person_tracks.append(p_entry)

            # -----------------------------------------------------------------
            # Step 2: Person A - Package Detection & Floor-Clamped HOI Tracking
            # -----------------------------------------------------------------
            box_results = pipeline_a.box_model.track(
                source=frame,
                persist=True,
                tracker=pipeline_a.tracker_config,
                conf=pipeline_a.box_conf,
                verbose=False
            )

            raw_box_tracks = []
            used_box_ids = set()
            if box_results and len(box_results) > 0:
                br = box_results[0]
                if br.boxes is not None and len(br.boxes) > 0:
                    for j, bbox_obj in enumerate(br.boxes):
                        bx1, by1, bx2, by2 = [round(float(v), 1) for v in bbox_obj.xyxy[0].tolist()]
                        bw, bh = bx2 - bx1, by2 - by1
                        if bw > (width * 0.58) or bh > (height * 0.58) or (bw * bh) > (width * height * 0.30):
                            continue

                        if bbox_obj.id is not None:
                            box_tid = int(bbox_obj.id[0]) + 1000
                        else:
                            box_tid = None
                            best_distance = 180.0
                            current_center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
                            for existing_tid, history in pipeline_a.interaction_tracker.box_history.items():
                                if existing_tid in used_box_ids or not history:
                                    continue
                                previous_center = history[-1][:2]
                                center_distance = ((current_center[0] - previous_center[0]) ** 2 +
                                                   (current_center[1] - previous_center[1]) ** 2) ** 0.5
                                if center_distance < best_distance:
                                    best_distance = center_distance
                                    box_tid = existing_tid
                            if box_tid is None:
                                box_tid = 1001 + j
                        used_box_ids.add(box_tid)
                        raw_box_tracks.append({
                            "track_id": box_tid,
                            "class": "cardboard box",
                            "bbox": [bx1, by1, bx2, by2],
                            "bbox_normalized": [round(bx1 / width, 3), round(by1 / height, 3), round(bx2 / width, 3), round(by2 / height, 3)],
                            "confidence": round(float(bbox_obj.conf[0]), 3)
                        })

            final_box_tracks = pipeline_a.interaction_tracker.update(
                raw_box_tracks, person_tracks, timestamp_sec
            )

            for fb in final_box_tracks:
                if "bbox_normalized" not in fb:
                    fx1, fy1, fx2, fy2 = fb["bbox"]
                    fb["bbox_normalized"] = [round(fx1 / width, 3), round(fy1 / height, 3), round(fx2 / width, 3), round(fy2 / height, 3)]

            all_tracks = person_tracks + final_box_tracks

            # Record frame to Person A's dataset
            current_frame_dict = {
                "frame_idx": frame_idx,
                "frame_id": frame_idx,
                "timestamp_sec": timestamp_sec,
                "tracks": all_tracks
            }
            tracking_data["frames"].append(current_frame_dict)

            # -----------------------------------------------------------------
            # Step 3: Person B - Real-Time Risk Engine Processing
            # -----------------------------------------------------------------
            new_events = risk_engine.process_frame(current_frame_dict)

            if new_events:
                # Update the active HUD alert with the most critical event
                top_evt = max(new_events, key=lambda e: (
                    3 if e["risk_level"] == "Critical" else
                    2 if e["risk_level"] == "High" else
                    1 if e["risk_level"] == "Medium" else 0
                ))
                active_alert = {
                    "risk_level": top_evt["risk_level"],
                    "behaviour_type": top_evt["behaviour_type"],
                    "action": top_evt["recommended_action"],
                    "near_miss_prob": top_evt.get("near_miss_probability", 0.0),
                    "frames_left": int(fps * 2.0)  # Display alert banner for 2 seconds
                }

            if active_alert:
                active_alert["frames_left"] -= 1

            # -----------------------------------------------------------------
            # Step 4: Render Combined Visualization
            # -----------------------------------------------------------------
            vis_frame = frame.copy()

            # Draw Tracks (Skeletons & Boxes)
            for t in all_tracks:
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                cls_name = t["class"]
                conf = t["confidence"]

                if cls_name == "person":
                    color = (255, 200, 0)
                    label = f"Worker #{t['track_id']}"
                else:
                    state = t.get("state", "RESTING")
                    held_by = t.get("held_by")
                    if state == "ROLLING":
                        color = (0, 215, 255)
                        label = f"Box #{t['track_id']} [ROLLING]"
                    elif state == "DROPPED":
                        color = (0, 80, 255)
                        label = f"Box #{t['track_id']} [DROPPED]"
                    elif held_by:
                        color = (0, 215, 255)
                        label = f"Box #{t['track_id']} [Worker #{held_by}]"
                    else:
                        color = (0, 140, 255)
                        label = f"Box #{t['track_id']}"

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                badge_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                cv2.rectangle(vis_frame, (x1, max(0, y1 - 18)), (x1 + badge_sz[0] + 4, max(18, y1)), color, -1)
                cv2.putText(vis_frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                # Draw worker skeleton bones
                if "keypoints" in t:
                    kps = t["keypoints"]
                    for p1, p2 in SKELETON_PAIRS:
                        n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                        if n1 in kps and n2 in kps and kps[n1][2] > 0.25 and kps[n2][2] > 0.25:
                            cv2.line(vis_frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)

                    for k_name, (kx, ky, kc) in kps.items():
                        if kc > 0.30:
                            if "wrist" in k_name:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 6, (0, 0, 255), -1)
                            elif "ankle" in k_name:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)
                            else:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)

            # Draw Person B HUD & Risk Alert Banner
            draw_hud_banner(vis_frame, active_alert, frame_idx, fps_display, len(person_tracks), len(final_box_tracks))

            writer.write(vis_frame)
            cv2.imshow("Godrej Warehouse AI: Person A Perception + Person B Risk Engine", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n[User Requested Exit] Finalizing and exporting deliverables...")
                break
            elif key == ord('s'):
                snap_path = os.path.join(output_dir, f"snapshot_{int(time.time())}.jpg")
                cv2.imwrite(snap_path, vis_frame)
                print(f"[Snapshot Saved] {snap_path}")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    tracking_data["session_metadata"]["total_frames"] = frame_idx
    tracking_data["session_metadata"]["duration_sec"] = round(total_time, 2)
    tracking_data["session_metadata"]["avg_fps"] = round(frame_idx / max(0.001, total_time), 2)

    # Export Person A Tracking JSON
    with open(tracking_json_path, "w") as f:
        json.dump(tracking_data, f, indent=2)

    # Finalize & Export Person B Warehouse Events JSON
    events_data = risk_engine.finalize()
    with open(events_json_path, "w") as f:
        json.dump(events_data, f, indent=2)

    # Consolidated Shift Log
    summary = events_data.get("shift_summary", {})
    print("\n" + "=" * 75)
    print("SESSION DELIVERABLES EXPORTED SUCCESSFULLY")
    print(f"1. Annotated Preview Video   : {video_out_path}")
    print(f"2. Person A Tracking JSON   : {tracking_json_path}")
    print(f"3. Person B Risk Events JSON : {events_json_path}")
    print("-" * 75)
    print(f"Total Frames Processed : {frame_idx} in {total_time:.1f}s ({frame_idx/max(0.001, total_time):.1f} FPS)")
    print(f"Total Risk Events Logged: {summary.get('total_events', 0)}")
    print(f"Near-Misses Prevented   : {summary.get('near_misses_prevented', 0)}")
    print(f"High / Critical Incidents: {summary.get('high_critical_risks', 0)}")
    print(f"Damage Avoided (Est INR): Rs. {summary.get('estimated_damage_avoided_inr', 0):,}")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live End-to-End Warehouse Perception + Risk Engine")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Optional video file path to run stream on")
    parser.add_argument("--box_conf", type=float, default=0.15, help="Detection threshold for cardboard boxes")
    parser.add_argument("--output_dir", type=str, default="outputs_live", help="Output directory for deliverables")
    args = parser.parse_args()

    src = args.video if args.video else args.camera
    run_live_pipeline(source=src, output_dir=args.output_dir, box_conf=args.box_conf)
