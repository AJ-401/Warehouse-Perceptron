"""
run_live_webcam.py
==================
Runs Person A's HOI Perception Pipeline on your live webcam/camera in real-time.

Controls:
- Press 'q' or 'ESC' on the live window to exit and save the session.
- Press 's' to save a screenshot of the current perception frame.

Usage:
    python run_live_webcam.py
    python run_live_webcam.py --camera 0
"""

import argparse
import json
import os
import sys
import time
import cv2
from pipeline_person_a import PersonAPipeline, KEYPOINT_NAMES, SKELETON_PAIRS


def run_webcam(camera_index: int = 0, output_dir: str = "outputs_live", box_conf: float = 0.30):
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    video_out_path = os.path.join(output_dir, f"live_session_{timestamp_str}.mp4")
    json_out_path = os.path.join(output_dir, f"live_session_{timestamp_str}_tracking.json")

    print("=" * 70)
    print("INITIALIZING PERSON A LIVE WEBCAM PERCEPTION")
    print(f"Connecting to Camera Index: {camera_index}...")
    print(f"Box Detection Confidence Threshold: {box_conf}")
    print("=" * 70)

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Warning] Failed with DirectShow. Retrying default backend on index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"[Error] Could not open camera at index {camera_index}.")
        print("Please check if your webcam is plugged in and permissions are allowed.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0 or fps > 60:
        fps = 30.0

    print(f"[Camera Connected] Resolution: {width}x{height} @ {fps:.1f} FPS")

    # Initialize Pipeline
    pipeline = PersonAPipeline(box_conf=box_conf, person_conf=0.35)

    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))

    tracking_data = {
        "session_metadata": {
            "source": f"Live Webcam Index {camera_index}",
            "resolution": [width, height],
            "fps": fps,
            "started_at": timestamp_str
        },
        "frames": []
    }

    print("\n" + "=" * 70)
    print("LIVE FEED RUNNING!")
    print("  -> Press 'q' or 'ESC' in the live window to STOP and save.")
    print("  -> Press 's' to take a snapshot.")
    print("=" * 70 + "\n")

    frame_idx = 0
    t_start = time.time()
    last_fps_time = time.time()
    fps_display = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Error] Failed to read frame from webcam.")
                break

            frame_idx += 1
            now = time.time()
            timestamp_sec = round(now - t_start, 3)

            # Step 1: Detect Workers & Skeletons
            pose_results = pipeline.pose_model.track(
                source=frame,
                persist=True,
                tracker=pipeline.tracker_config,
                conf=pipeline.person_conf,
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

                        person_entry = {
                            "track_id": track_id,
                            "class": "person",
                            "bbox": [x1, y1, x2, y2],
                            "confidence": round(conf, 3)
                        }

                        if kps_data is not None and i < len(kps_data):
                            person_kps = kps_data[i]
                            kp_dict = {}
                            for k_idx, (kx, ky, kc) in enumerate(person_kps):
                                kp_dict[KEYPOINT_NAMES[k_idx]] = [
                                    round(float(kx), 1),
                                    round(float(ky), 1),
                                    round(float(kc), 3)
                                ]
                            person_entry["keypoints"] = kp_dict

                        person_tracks.append(person_entry)

            # Step 2: Detect Boxes with YOLO11
            box_results = pipeline.box_model.track(
                source=frame,
                persist=True,
                tracker=pipeline.tracker_config,
                conf=pipeline.box_conf,
                verbose=False
            )

            raw_box_tracks = []
            if box_results and len(box_results) > 0:
                br = box_results[0]
                if br.boxes is not None and len(br.boxes) > 0:
                    for j, bbox_obj in enumerate(br.boxes):
                        bx1, by1, bx2, by2 = [round(float(v), 1) for v in bbox_obj.xyxy[0].tolist()]
                        bw = bx2 - bx1
                        bh = by2 - by1
                        # Filter oversized false positives (background shelves, furniture, walls)
                        if bw > (width * 0.58) or bh > (height * 0.58) or (bw * bh) > (width * height * 0.30):
                            continue

                        # Robust ID Assignment without proliferating ghost tracks
                        if bbox_obj.id is not None:
                            box_track_id = int(bbox_obj.id[0]) + 1000
                        else:
                            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            matched_id = None
                            for existing_tid, hist in pipeline.interaction_tracker.box_history.items():
                                if hist:
                                    dist = ((bcx - hist[-1][0])**2 + (bcy - hist[-1][1])**2)**0.5
                                    if dist < 180:
                                        matched_id = existing_tid
                                        break
                            box_track_id = matched_id if matched_id is not None else 1001

                        norm_bbox = [round(bx1 / width, 3), round(by1 / height, 3), round(bx2 / width, 3), round(by2 / height, 3)]
                        raw_box_tracks.append({
                            "track_id": box_track_id,
                            "class": "cardboard box",
                            "bbox": [bx1, by1, bx2, by2],
                            "bbox_normalized": norm_bbox,
                            "confidence": round(float(bbox_obj.conf[0]), 3)
                        })

            # Step 3: Dynamic Floor-Clamped HOI Tracker
            final_box_tracks = pipeline.interaction_tracker.update(
                raw_box_tracks, person_tracks, timestamp_sec
            )

            for fb in final_box_tracks:
                if "bbox_normalized" not in fb:
                    fx1, fy1, fx2, fy2 = fb["bbox"]
                    fb["bbox_normalized"] = [round(fx1 / width, 3), round(fy1 / height, 3), round(fx2 / width, 3), round(fy2 / height, 3)]

            all_frame_tracks = person_tracks + final_box_tracks

            # Record to JSON contract
            tracking_data["frames"].append({
                "frame_idx": frame_idx,
                "frame_id": frame_idx,
                "timestamp_sec": timestamp_sec,
                "tracks": all_frame_tracks
            })

            # Step 4: Render UI Visualization
            vis_frame = frame.copy()
            n_persons = len(person_tracks)
            n_boxes = len(final_box_tracks)

            # Draw tracks
            for t in all_frame_tracks:
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                cls_name = t["class"]
                track_id = t["track_id"]
                conf = t["confidence"]

                if cls_name == "person":
                    color = (255, 200, 0)  # Cyan
                    label = f"Person #{track_id} ({conf:.2f})"
                else:
                    state = t.get("state", "RESTING")
                    held_by = t.get("held_by", None)

                    if state == "ROLLING":
                        color = (0, 215, 255)  # Gold
                        label = f"Box #{track_id} (HOI) [Worker #{held_by}] - ROLLING"
                    elif state == "DROPPED":
                        color = (0, 165, 255)  # Amber
                        label = f"Box #{track_id} (FREE-FALL)"
                    elif held_by is not None:
                        color = (0, 215, 255)  # Gold
                        label = f"Box #{track_id} (HOI) [Worker #{held_by}]"
                    else:
                        color = (0, 140, 255)  # Orange
                        label = f"Box #{track_id} ({conf:.2f})"

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                badge_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
                cv2.rectangle(vis_frame, (x1, max(0, y1 - 20)), (x1 + badge_size[0] + 6, max(20, y1)), color, -1)
                cv2.putText(vis_frame, label, (x1 + 3, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                # Draw worker skeleton
                if "keypoints" in t:
                    kps = t["keypoints"]
                    for p1, p2 in SKELETON_PAIRS:
                        name1, name2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                        if name1 in kps and name2 in kps:
                            x_a, y_a, c_a = kps[name1]
                            x_b, y_b, c_b = kps[name2]
                            if c_a > 0.30 and c_b > 0.30:
                                cv2.line(vis_frame, (int(x_a), int(y_a)), (int(x_b), int(y_b)), (0, 255, 128), 2)

                    for k_name, (kx, ky, kc) in kps.items():
                        if kc > 0.30:
                            if "wrist" in k_name:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 6, (0, 0, 255), -1)    # Red wrists
                            elif "ankle" in k_name:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)  # Magenta ankles
                            else:
                                cv2.circle(vis_frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)

            # Measure smooth FPS
            if frame_idx % 10 == 0:
                elapsed_fps = now - last_fps_time
                fps_display = 10.0 / max(0.001, elapsed_fps)
                last_fps_time = now

            # Top HUD Bar
            hud_bg = vis_frame.copy()
            cv2.rectangle(hud_bg, (0, 0), (width, 45), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.75, vis_frame, 0.25, 0, vis_frame)

            hud_text = (
                f"LIVE WEBCAM PERCEPTION | Time: {timestamp_sec:.1f}s | "
                f"Workers: {n_persons} | Boxes: {n_boxes} | Speed: {fps_display:.1f} FPS | "
                f"Press 'q' to Exit"
            )
            cv2.putText(vis_frame, hud_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            # Record frame
            writer.write(vis_frame)

            # Show Interactive Window
            cv2.imshow("Person A - Live Warehouse AI Perception", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\n[User Exit] Stopping live feed...")
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

    with open(json_out_path, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print("\n" + "=" * 70)
    print("LIVE SESSION COMPLETED SUCCESSFULLY")
    print(f"Recorded Video : {video_out_path}")
    print(f"Tracking Data  : {json_out_path}")
    print(f"Total Frames   : {frame_idx} in {total_time:.1f}s ({frame_idx/max(0.001, total_time):.1f} FPS)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person A Live Webcam Perception")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--box_conf", type=float, default=0.30, help="Confidence threshold for boxes (default: 0.30)")
    parser.add_argument("--output_dir", type=str, default="outputs_live", help="Output directory")
    args = parser.parse_args()

    run_webcam(camera_index=args.camera, output_dir=args.output_dir, box_conf=args.box_conf)
