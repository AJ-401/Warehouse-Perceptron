"""
pipeline_person_a.py
====================
Person A Pipeline: Production Perception with Dynamic Floor-Clamped HOI Tracking.
(Patched for Velocity Calculation, ID Splits, Ghost Boxes, and Premature Disappearance)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ── Person B inline risk engine (Method 1 integration) ──────────────────────
try:
    from pipeline_person_b import RiskEngine as _RiskEngine
    _RISK_ENGINE_AVAILABLE = True
except ImportError:
    _RISK_ENGINE_AVAILABLE = False
    _RiskEngine = None

# COCO Keypoint names for YOLO-Pose (17 points)
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# SKELETON CONNECTIONS FOR VISUALIZATION
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]


class DynamicHOITracker:
    def __init__(self, max_missing_frames: int = 90):
        self.max_missing = max_missing_frames
        self.anchored_objects: Dict[int, Dict[str, Any]] = {}
        self.wrist_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.box_history: Dict[int, List[Tuple[float, float, float]]] = {}
        self.freefall_objects: Dict[int, Dict[str, Any]] = {}

    def _get_worker_pose_anchors(self, person: Dict[str, Any]) -> Tuple[Optional[Tuple[float, float]], float, float]:
        px1, py1, px2, py2 = person["bbox"]
        p_height = max(1.0, py2 - py1)
        kps = person.get("keypoints", {})

        lw = kps.get("left_wrist", [0, 0, 0])
        rw = kps.get("right_wrist", [0, 0, 0])
        hands = []
        if lw[2] > 0.25: hands.append((lw[0], lw[1]))
        if rw[2] > 0.25: hands.append((rw[0], rw[1]))

        hand_center = None
        if hands:
            hand_center = (
                sum(h[0] for h in hands) / len(hands),
                sum(h[1] for h in hands) / len(hands)
            )

        la = kps.get("left_ankle", [0, 0, 0])
        ra = kps.get("right_ankle", [0, 0, 0])
        ankles_y = []
        if la[2] > 0.25: ankles_y.append(la[1])
        if ra[2] > 0.25: ankles_y.append(ra[1])

        y_floor = max(ankles_y) if ankles_y else py2
        return hand_center, y_floor, p_height

    def update(self, detected_boxes: List[Dict[str, Any]], detected_persons: List[Dict[str, Any]], timestamp_sec: float) -> List[Dict[str, Any]]:
        final_boxes: List[Dict[str, Any]] = []
        matched_box_ids = set()
        person_map = {p["track_id"]: p for p in detected_persons}

        # Update wrist history
        for pid, p in person_map.items():
            h_center, _, _ = self._get_worker_pose_anchors(p)
            if h_center is not None:
                if pid not in self.wrist_history:
                    self.wrist_history[pid] = []
                self.wrist_history[pid].append((h_center[0], h_center[1], timestamp_sec))
                if len(self.wrist_history[pid]) > 10:
                    self.wrist_history[pid].pop(0)

        # ---------------------------------------------------------------------
        # Pre-Step: Spatial Re-identification (Ghost Box Fix)
        # ---------------------------------------------------------------------
        for b in detected_boxes:
            tid = b["track_id"]
            bx1, by1, bx2, by2 = b["bbox"]
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            
            merged = False
            for old_tid, anchor in list(self.anchored_objects.items()):
                if tid != old_tid and anchor["missing_count"] > 0:
                    last_pos = self.box_history.get(old_tid, [])
                    if last_pos:
                        dist = ((bcx - last_pos[-1][0])**2 + (bcy - last_pos[-1][1])**2)**0.5
                        if dist < 120:  
                            b["track_id"] = old_tid
                            tid = old_tid
                            del self.anchored_objects[old_tid]
                            merged = True
                            break
            
            if not merged:
                for old_tid, ff in list(self.freefall_objects.items()):
                    if tid != old_tid:
                        fbx1, fby1, fbx2, fby2 = ff["bbox"]
                        fbcx, fbcy = (fbx1 + fbx2) / 2.0, (fby1 + fby2) / 2.0
                        dist = ((bcx - fbcx)**2 + (bcy - fbcy)**2)**0.5
                        if dist < 120:
                            b["track_id"] = old_tid
                            tid = old_tid
                            del self.freefall_objects[old_tid]
                            break

        # ---------------------------------------------------------------------
        # Step 1: Process Directly Detected Boxes
        # ---------------------------------------------------------------------
        for b in detected_boxes:
            tid = b["track_id"]
            bx1, by1, bx2, by2 = b["bbox"]
            bw, bh = bx2 - bx1, by2 - by1
            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            matched_box_ids.add(tid)

            if tid not in self.box_history: self.box_history[tid] = []
            self.box_history[tid].append((bcx, bcy, timestamp_sec))
            if len(self.box_history[tid]) > 10: self.box_history[tid].pop(0)

            best_holder_id = None
            min_dist = float("inf")
            hx_for_offset = 0.0

            for pid, p in person_map.items():
                h_center, y_floor, p_height = self._get_worker_pose_anchors(p)
                if h_center is not None:
                    hx, hy = h_center
                    dx = max(0.0, bx1 - hx, hx - bx2)
                    dy = max(0.0, by1 - hy, hy - by2)
                    dist = (dx**2 + dy**2)**0.5
                    max_grasp_dist = max(110.0, 0.18 * p_height)
                    if dist < max_grasp_dist and dist < min_dist:
                        min_dist = dist
                        best_holder_id = pid
                        hx_for_offset = hx

            if best_holder_id is not None:
                self.anchored_objects[tid] = {
                    "width": bw,
                    "height": bh,
                    "offset_x": bcx - hx_for_offset,  # Preserve horizontal grasp point
                    "holder_person_id": best_holder_id,
                    "missing_count": 0,
                    "last_conf": b["confidence"],
                    "class": b["class"]
                }
                b["held_by"] = best_holder_id
                b["state"] = "HELD"
            else:
                b["state"] = "RESTING"

            final_boxes.append(b)

        # ---------------------------------------------------------------------
        # Step 2: Dynamic Floor-Clamped Projection
        # ---------------------------------------------------------------------
        to_delete_anchors = []

        for tid, anchor in self.anchored_objects.items():
            if tid in matched_box_ids: continue

            anchor["missing_count"] += 1
            if anchor["missing_count"] > self.max_missing:
                to_delete_anchors.append(tid)
                continue

            holder_id = anchor["holder_person_id"]
            if holder_id not in person_map:
                to_delete_anchors.append(tid)
                continue

            worker = person_map[holder_id]
            h_center, y_floor, p_height = self._get_worker_pose_anchors(worker)

            if h_center is None:
                to_delete_anchors.append(tid)
                continue

            hx, hy = h_center
            
            # Velocity Calculation Fix
            wrist_vy = 0.0
            wrist_vx = 0.0
            if holder_id in self.wrist_history and len(self.wrist_history[holder_id]) >= 2:
                prev_h = self.wrist_history[holder_id][-2]
                dt = max(0.001, timestamp_sec - prev_h[2])
                wrist_vy = (hy - prev_h[1]) / dt
                wrist_vx = (hx - prev_h[0]) / dt

            box_center_x = hx + anchor.get("offset_x", 0.0)
            half_w = anchor["width"] / 2.0
            box_h = anchor["height"]

            is_rolling = False
            if abs(y_floor - hy) < (0.45 * p_height) and abs(wrist_vx) > 10.0:
                is_rolling = True

            # If rolling on floor, center is between wrists and floor; if carried in air, center tracks wrist height
            pred_bcy_current = (hy + y_floor) / 2.0 if is_rolling else hy

            box_vy = 0.0
            if tid in self.box_history and len(self.box_history[tid]) >= 2:
                prev_b = self.box_history[tid][-2]
                dt = max(0.001, timestamp_sec - prev_b[2])
                box_vy = (pred_bcy_current - prev_b[1]) / dt

            # Drop trigger: Only uncouple if box is genuinely accelerating DOWNWARD faster than hands
            is_drop_event = False
            if (box_vy - wrist_vy) > 120.0 and box_vy > 80.0:
                is_drop_event = True
            elif wrist_vy < -80.0 and box_vy > 60.0 and hy < (y_floor - 0.35 * p_height):
                # Worker rapidly recoils hands upward while box is accelerating downward
                is_drop_event = True

            if is_drop_event:
                to_delete_anchors.append(tid)
                curr_y1 = hy + 15
                curr_y2 = min(y_floor, curr_y1 + box_h)
                self.freefall_objects[tid] = {
                    "bbox": [box_center_x - half_w, curr_y1, box_center_x + half_w, curr_y2],
                    "v_y": max(150.0, box_vy),
                    "y_floor": y_floor,
                    "width": anchor["width"],
                    "height": anchor["height"],
                    "class": anchor["class"],
                    "conf": 0.65,
                    "missing": 0
                }
                continue

            # Projection: If rolling on floor, clamp bottom to floor; if carried in air, keep centered around wrists
            if is_rolling:
                y_min = hy
                y_max = y_floor
                min_h = max(25.0, box_h * 0.4)
                if (y_max - y_min) < min_h:
                    y_min = y_max - min_h
            else:
                # Carried in hands: center vertically with respect to wrists
                y_min = max(0.0, hy - box_h * 0.5)
                y_max = min(y_floor, y_min + box_h)

            projected_bbox = [
                round(box_center_x - half_w, 1),
                round(y_min, 1),
                round(box_center_x + half_w, 1),
                round(y_max, 1)
            ]

            if tid not in self.box_history: self.box_history[tid] = []
            self.box_history[tid].append((box_center_x, (y_min + y_max) / 2.0, timestamp_sec))
            if len(self.box_history[tid]) > 10: self.box_history[tid].pop(0)

            decay_conf = round(max(0.40, anchor["last_conf"] * (0.99 ** anchor["missing_count"])), 3)
            final_boxes.append({
                "track_id": tid,
                "class": anchor["class"],
                "bbox": projected_bbox,
                "confidence": decay_conf,
                "held_by": holder_id,
                "inferred": True,
                "state": "ROLLING" if is_rolling else "HELD",
                "event": "ROLLING" if is_rolling else "MANIPULATING"
            })

        for tid in to_delete_anchors:
            if tid in self.anchored_objects: del self.anchored_objects[tid]

        # ---------------------------------------------------------------------
        # Step 3: Handle Free-Fall & Settling
        # ---------------------------------------------------------------------
        to_delete_freefall = []
        for tid, ff in self.freefall_objects.items():
            if tid in matched_box_ids:
                to_delete_freefall.append(tid)
                continue

            ff["missing"] += 1
            # Once settled on floor, expire after 8 frames; if still in freefall, allow up to 20 frames
            max_ff_missing = 8 if ff.get("settled", False) else 20
            if ff["missing"] > max_ff_missing: 
                to_delete_freefall.append(tid)
                continue

            bx1, by1, bx2, by2 = ff["bbox"]
            if ff["v_y"] > 0:
                dt = 1.0 / 30.0
                ff["v_y"] += 450.0 * dt
                new_y1 = by1 + ff["v_y"] * dt
                new_y2 = by2 + ff["v_y"] * dt

                if new_y2 >= ff["y_floor"]:
                    new_y2 = ff["y_floor"]
                    new_y1 = max(new_y2 - ff.get("height", by2 - by1), 0)
                    ff["bbox"] = [bx1, new_y1, bx2, new_y2]
                    ff["v_y"] = 0
                    ff["settled"] = True
                    state = "DROPPED"
                else:
                    ff["bbox"] = [bx1, new_y1, bx2, new_y2]
                    state = "FREE_FALL"
            else:
                state = "DROPPED"

            cur_x1, cur_y1, cur_x2, cur_y2 = ff["bbox"]
            final_boxes.append({
                "track_id": tid,
                "class": ff["class"],
                "bbox": [round(cur_x1, 1), round(cur_y1, 1), round(cur_x2, 1), round(cur_y2, 1)],
                "confidence": round(max(0.20, ff["conf"] * (0.95 ** ff["missing"])), 2),
                "held_by": None,
                "state": state,
                "event": "FREE_FALL" if state == "FREE_FALL" else "SETTLED"
            })

        for tid in to_delete_freefall:
            if tid in self.freefall_objects: del self.freefall_objects[tid]

        return final_boxes


class PersonAPipeline:
    def __init__(self, box_model_path="weights/box_11s.pt", pose_model_path="yolov8n-pose.pt", box_conf=0.10, person_conf=0.35):
        self.box_conf = box_conf
        self.person_conf = person_conf
        self.tracker_config = "custom_bytetrack.yaml"
        
        with open(self.tracker_config, "w") as f:
            f.write(
                "tracker_type: bytetrack\n"
                "track_high_thresh: 0.10\n"
                "track_low_thresh: 0.05\n"
                "new_track_thresh: 0.10\n"
                "track_buffer: 90\n"
                "match_thresh: 0.60\n"
                "fuse_score: True\n"
            )
        
        self.box_model = YOLO(box_model_path)
        self.pose_model = YOLO(pose_model_path)
        self.interaction_tracker = DynamicHOITracker(max_missing_frames=45)

    def run(self, video_path: str, output_dir: str = "outputs_hoi_v2", max_frames: Optional[int] = None, save_video: bool = True, enable_risk_engine: bool = True):
        os.makedirs(output_dir, exist_ok=True)
        video_filename = os.path.basename(video_path)
        base_name = os.path.splitext(video_filename)[0]
        json_out_path = os.path.join(output_dir, f"{base_name}_tracking_results.json")
        video_out_path = os.path.join(output_dir, f"{base_name}_perception.mp4")
        events_out_path = os.path.join(output_dir, f"{base_name}_warehouse_events.json")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        limit_frames = min(total_video_frames, max_frames) if max_frames else total_video_frames
        writer = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)) if save_video else None

        # ── Inline Person B Risk Engine ──────────────────────────────────────
        risk_engine = None
        active_alert = None          # last fired event dict (shown for N frames)
        alert_frames_left = 0
        if enable_risk_engine and _RISK_ENGINE_AVAILABLE:
            risk_engine = _RiskEngine(video_id=video_filename, fps=fps, resolution=[width, height])
            print("[Person B] Risk Engine attached inline — live risk scoring ENABLED.")
        elif enable_risk_engine:
            print("[Person B] pipeline_person_b.py not found — risk scoring DISABLED.")

        tracking_data = {"video_metadata": {"filename": video_filename, "fps": round(fps, 2), "resolution": [width, height], "total_frames_processed": 0}, "frames": []}
        frame_idx, t_start = 0, time.time()

        try:
            while cap.isOpened() and (max_frames is None or frame_idx < max_frames):
                ret, frame = cap.read()
                if not ret: break
                frame_idx += 1
                timestamp_sec = round(frame_idx / fps, 3)

                person_tracks, raw_box_tracks = [], []
                pose_results = self.pose_model.track(source=frame, persist=True, tracker=self.tracker_config, conf=self.person_conf, verbose=False)
                
                if pose_results and len(pose_results) > 0 and pose_results[0].boxes:
                    r = pose_results[0]
                    kps_data = r.keypoints.data.cpu().numpy() if r.keypoints else None
                    for i, box in enumerate(r.boxes):
                        track_id = int(box.id[0]) if box.id else (i + 1)
                        x1, y1, x2, y2 = [round(float(v), 1) for v in box.xyxy[0].tolist()]
                        norm_box = [round(x1 / width, 3), round(y1 / height, 3), round(x2 / width, 3), round(y2 / height, 3)]
                        person_entry = {
                            "track_id": track_id,
                            "class": "person",
                            "bbox": [x1, y1, x2, y2],
                            "bbox_normalized": norm_box,
                            "confidence": round(float(box.conf[0]), 3)
                        }
                        if kps_data is not None and i < len(kps_data):
                            person_entry["keypoints"] = {KEYPOINT_NAMES[k_idx]: [round(float(kx), 1), round(float(ky), 1), round(float(kc), 3)] for k_idx, (kx, ky, kc) in enumerate(kps_data[i])}
                        person_tracks.append(person_entry)

                box_results = self.box_model.track(source=frame, persist=True, tracker=self.tracker_config, conf=self.box_conf, verbose=False)
                if box_results and len(box_results) > 0 and box_results[0].boxes:
                    for j, bbox_obj in enumerate(box_results[0].boxes):
                        bx1, by1, bx2, by2 = [round(float(v), 1) for v in bbox_obj.xyxy[0].tolist()]
                        bw = bx2 - bx1
                        bh = by2 - by1
                        # Filter oversized camera-edge artifacts (allow large cartons, flatpacks and pallets up to 92%)
                        if bw > (width * 0.92) or bh > (height * 0.92) or (bw * bh) > (width * height * 0.85):
                            continue

                        # Robust ID Assignment without proliferating ghost tracks
                        if bbox_obj.id is not None:
                            box_track_id = int(bbox_obj.id[0]) + 1000
                        else:
                            bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            matched_id = None
                            for existing_tid, hist in self.interaction_tracker.box_history.items():
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

                final_box_tracks = self.interaction_tracker.update(raw_box_tracks, person_tracks, timestamp_sec)
                
                # Ensure all final box tracks include bbox_normalized
                for fb in final_box_tracks:
                    if "bbox_normalized" not in fb:
                        fx1, fy1, fx2, fy2 = fb["bbox"]
                        fb["bbox_normalized"] = [round(fx1 / width, 3), round(fy1 / height, 3), round(fx2 / width, 3), round(fy2 / height, 3)]

                all_frame_tracks = person_tracks + final_box_tracks
                tracking_data["frames"].append({
                    "frame_idx": frame_idx,
                    "frame_id": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "tracks": all_frame_tracks
                })

                # ── Inline Risk Engine: process current frame ────────────────
                if risk_engine is not None:
                    frame_payload = {
                        "frame_idx": frame_idx,
                        "frame_id": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "tracks": all_frame_tracks
                    }
                    new_events = risk_engine.process_frame(frame_payload)
                    _SEV_RANK = {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
                    _TYPE_PRIORITY = {
                        "DROP_HIGH_IMPACT": 10,
                        "DROP_LOW_SLIP": 9,
                        "NEAR_MISS_UNSAFE_CARRY": 8,
                        "ROUGH_THROW_SLIDE": 7,
                        "ROUGH_CARTON_ROLLING": 6,
                        "OPERATOR_STEPPING_CARTON": 5,
                        "EQUIPMENT_STRAP_LIFT": 4,
                        "UNSAFE_FLOOR_DRAG": 3,
                        "IMPROPER_MISORIENTATION": 2,
                        "STACK_UNSTABLE_WOBBLE": 2,
                        "STACK_INVERTED_PYRAMID": 2,
                    }

                    if new_events:
                        hazard_events = [
                            e for e in new_events
                            if e.get("behaviour_type") != "BENCHMARK_SAFE_HANDLING"
                            and e.get("behaviour_code") != "BENCHMARK_SAFE_HANDLING"
                        ]
                        if hazard_events:
                            cand_evt = max(hazard_events, key=lambda e: (
                                _TYPE_PRIORITY.get(e.get("behaviour_code", ""), 0),
                                _SEV_RANK.get(e.get("risk_level", "Low"), 0)
                            ))
                            cand_score = (
                                _TYPE_PRIORITY.get(cand_evt.get("behaviour_code", ""), 0),
                                _SEV_RANK.get(cand_evt.get("risk_level", "Low"), 0)
                            )
                            curr_score = (
                                _TYPE_PRIORITY.get(active_alert.get("behaviour_code", ""), 0),
                                _SEV_RANK.get(active_alert.get("risk_level", "Low"), 0)
                            ) if (active_alert and alert_frames_left > 0) else (0, 0)

                            is_new = (active_alert is None or alert_frames_left <= 0 or
                                      cand_evt.get("event_id") != active_alert.get("event_id"))

                            if is_new or cand_score > curr_score:
                                hold_time = 2.5 if cand_evt.get("is_near_miss", False) or cand_score[1] >= 3 else 2.0
                                active_alert = cand_evt
                                alert_frames_left = int(fps * hold_time)

                    if alert_frames_left > 0:
                        alert_frames_left -= 1
                    else:
                        active_alert = None

                if writer:
                    vis_frame = frame.copy()
                    for t in all_frame_tracks:
                        x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                        cls_name = t["class"]
                        if cls_name == "person":
                            color, label = (255, 200, 0), "WORKER"
                        else:
                            state, held_by = t.get("state", "RESTING"), t.get("held_by", None)
                            if state == "ROLLING": color, label = (0, 215, 255), "ROLLING"
                            elif state == "DROPPED": color, label = (0, 165, 255), "DROPPED"
                            elif held_by: color, label = (0, 215, 255), "HELD"
                            else: color, label = (0, 140, 255), "CARTON"

                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        badge_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                        cv2.rectangle(vis_frame, (x1, max(0, y1 - 18)), (x1 + badge_size[0] + 6, max(18, y1)), color, -1)
                        cv2.putText(vis_frame, label, (x1 + 3, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                        if "keypoints" in t:
                            kps = t["keypoints"]
                            for p1, p2 in SKELETON_PAIRS:
                                n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                                if n1 in kps and n2 in kps and kps[n1][2] > 0.3 and kps[n2][2] > 0.3:
                                    cv2.line(vis_frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)
                            for k_name, (kx, ky, kc) in kps.items():
                                if kc > 0.3:
                                    kp_col = (0, 0, 255) if "wrist" in k_name else (255, 0, 255) if "ankle" in k_name else (0, 255, 0)
                                    cv2.circle(vis_frame, (int(kx), int(ky)), 6 if "wrist" in k_name or "ankle" in k_name else 3, kp_col, -1)

                    # ── Top HUD bar ──────────────────────────────────────────
                    cv2.rectangle(vis_frame, (0, 0), (width, 32), (18, 18, 18), -1)
                    cv2.line(vis_frame, (0, 32), (width, 32), (55, 55, 55), 1)
                    cv2.putText(vis_frame, "GODREJ AI | FIELD INTELLIGENCE", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

                    # ── Risk Alert Banner with Recommendations ──
                    if active_alert and alert_frames_left > 0:
                        level = active_alert.get("risk_level", "Medium")
                        btype = active_alert.get("behaviour_type", "Safety Alert")
                        bcode = active_alert.get("behaviour_code", "")
                        prob = active_alert.get("near_miss_probability", 0.0)
                        is_near_miss = active_alert.get("is_near_miss", False) or prob > 0.0

                        if is_near_miss:
                            bg_col, txt_col, tag_str = (0, 0, 210), (255, 255, 255), "NEAR MISS"
                        elif level == "Critical":
                            bg_col, txt_col, tag_str = (0, 0, 210), (255, 255, 255), "CRITICAL"
                        elif level == "High":
                            bg_col, txt_col, tag_str = (0, 125, 245), (255, 255, 255), "HIGH RISK"
                        elif level == "Medium":
                            bg_col, txt_col, tag_str = (0, 195, 240), (20, 20, 20), "WARNING"
                        else:
                            bg_col, txt_col, tag_str = (40, 165, 60), (255, 255, 255), "SAFE"

                        SHORT_MAP = {
                            "DROP_HIGH_IMPACT": "HIGH DROP", "DROP_LOW_SLIP": "CARTON SLIP",
                            "NEAR_MISS_UNSAFE_CARRY": "UNSAFE CARRY", "UNSAFE_FLOOR_DRAG": "FLOOR DRAG",
                            "ROUGH_THROW_SLIDE": "CARTON THROW", "ROUGH_CARTON_ROLLING": "CARTON ROLLING",
                            "STACK_INVERTED_PYRAMID": "BAD STACK", "STACK_UNSTABLE_WOBBLE": "UNSTABLE STACK",
                            "EQUIPMENT_STRAP_LIFT": "STRAP LIFT", "OPERATOR_STEPPING_CARTON": "STEPPING HAZARD",
                            "IMPROPER_MISORIENTATION": "WRONG ORIENTATION", "BENCHMARK_SAFE_HANDLING": "SAFE HANDLING"
                        }
                        issue = SHORT_MAP.get(bcode, SHORT_MAP.get(btype, " ".join(btype.replace("_", " ").split()[:2]).upper()))
                        action = active_alert.get("recommended_action", active_alert.get("action", "Follow standard material handling guidelines."))
                        if len(action) > 105:
                            action = action[:102] + "..."

                        bh = 68
                        by1 = height - bh - 15
                        by2 = height - 15
                        cv2.rectangle(vis_frame, (25, by1), (width - 25, by2), bg_col, -1)
                        cv2.rectangle(vis_frame, (25, by1), (width - 25, by2), (255, 255, 255), 2)

                        tag_sz = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_DUPLEX, 0.60, 2)[0]
                        badge_w = tag_sz[0] + 18
                        cv2.rectangle(vis_frame, (35, by1 + 8), (35 + badge_w, by1 + 33), (255, 255, 255), -1)
                        cv2.putText(vis_frame, tag_str, (44, by1 + 27), cv2.FONT_HERSHEY_DUPLEX, 0.58, (10, 10, 10), 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, issue, (35 + badge_w + 18, by1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_col, 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, f"RECOMMENDATION: {action}", (35, by1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
                    else:
                        # Safe default banner
                        bg_col = (35, 145, 55)
                        txt_col = (255, 255, 255)
                        tag_str = "SAFE"
                        issue = "NORMAL OPERATIONS"
                        action = "All handling practices operating within safe parameters."

                        bh = 68
                        by1 = height - bh - 15
                        by2 = height - 15
                        cv2.rectangle(vis_frame, (25, by1), (width - 25, by2), bg_col, -1)
                        cv2.rectangle(vis_frame, (25, by1), (width - 25, by2), (100, 220, 130), 2)

                        tag_sz = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_DUPLEX, 0.60, 2)[0]
                        badge_w = tag_sz[0] + 18
                        cv2.rectangle(vis_frame, (35, by1 + 8), (35 + badge_w, by1 + 33), (255, 255, 255), -1)
                        cv2.putText(vis_frame, tag_str, (44, by1 + 27), cv2.FONT_HERSHEY_DUPLEX, 0.58, (10, 80, 20), 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, issue, (35 + badge_w + 18, banner_y1 + 28) if 'banner_y1' in locals() else (35 + badge_w + 18, by1 + 28), cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_col, 2, cv2.LINE_AA)
                        cv2.putText(vis_frame, f"STATUS: {action}", (35, by1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 245, 220), 1, cv2.LINE_AA)

                    writer.write(vis_frame)

        finally:
            cap.release()
            if writer: writer.release()

        elapsed_total = time.time() - t_start
        tracking_data["video_metadata"]["total_frames_processed"] = frame_idx
        tracking_data["video_metadata"]["processing_time_sec"] = round(elapsed_total, 2)
        with open(json_out_path, "w") as f: json.dump(tracking_data, f, indent=2)

        # ── Save Person B events JSON ────────────────────────────────────────
        if risk_engine is not None:
            events_output = risk_engine.finalize()
            with open(events_out_path, "w") as f: json.dump(events_output, f, indent=2)
            n_events = len(events_output.get("events", []))
            print(f"[Person B] {n_events} risk event(s) saved → {events_out_path}")

        return json_out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Person A Perception + Person B Risk Engine (Method 1)")
    parser.add_argument("--input", type=str, required=True, help="Path to video file")
    parser.add_argument("--output_dir", type=str, default="outputs_person_a")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--no_video", action="store_true")
    parser.add_argument("--no_risk", action="store_true", help="Disable inline Person B risk engine")
    args = parser.parse_args()
    pipeline = PersonAPipeline()
    pipeline.run(args.input, args.output_dir, args.max_frames, not args.no_video, not args.no_risk)

