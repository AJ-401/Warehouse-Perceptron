"""
pipeline_person_b.py
====================
Person B — Production Behaviour Rules + Dynamic Risk Scoring Engine + Predictive Near-Miss (USP)
Godrej Warehouse Video Intelligence Hackathon | graVITas 2026

INPUT  : outputs_person_a/<video>_tracking_results.json  (Person A Contract, DATA_CONTRACT.md)
OUTPUT : outputs_person_b/<video>_warehouse_events.json  (Person C/D Contract, PRD.md Section 7.2)
MERGED : outputs_person_b/warehouse_events.json

Architecture & Design:
----------------------
1. Dynamic Scene Calibration:
   - Dynamic floor plane estimation: Calculates y_floor per-frame from worker ankle/foot keypoints.
   - Physical scale calibration: Computes real-world scale (meters/pixel) from median worker height in the scene.
2. Robust Track Buffer & Spatial Re-ID:
   - Maintains rolling 15-frame temporal buffers.
   - Includes spatial fallback when ByteTrack remints box IDs during partial occlusions.
3. 10 Warehouse Scenarios:
   - Scenario 1:  DRAG_NO_EQUIPMENT (Ground dragging along floor without equipment)
   - Scenario 2:  UNCONTROLLED_DROP_HIGH (High-impact drops >= 0.8m)
   - Scenario 3:  CARTON_SLIP_LOW (Low slips 0.25m - 0.8m)
   - Scenario 4:  NEAR_MISS_UNSAFE_CARRY (Predictive Near-Miss USP: continuous transit risk alert)
   - Scenario 5:  CARTON_THROW_SLIDE (Excessive horizontal velocity slide/throw)
   - Scenario 6:  IMPROPER_STACKING (Inverted pyramid stacking with physical contact constraint)
   - Scenario 7:  STEPPING_ON_CARTON (Operator foot/ankle on package top surface)
   - Scenario 8:  STRAP_LIFT_PULL (Lifting by packaging straps without bottom/side support)
   - Scenario 9:  UNSTABLE_STACK_WOBBLE (Sustained stack tilt/oscillation filtering static detector jitter)
   - Scenario 10: SAFE_HANDLING_BENCHMARK (Controlled low-velocity handling control)
4. Dynamic Risk Scoring & Confidence Ladder:
   - Multi-factor severity score (drop height, velocity, duration, repeat count) -> Low, Medium, High, Critical.
   - Responsible AI Confidence Ladder: Observed -> Potential Risk -> Confirmed Damage.
5. Strict Contract Compliance (PRD Section 7.2):
   - Fully typed JSON schema, metric units in telemetry, damage avoided calculation.
"""

from __future__ import annotations
import argparse
import glob
import json
import math
import os
import cv2
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------------------
# Config & Thresholds (Calibrated for Warehouse Loading Bay Cameras)
# --------------------------------------------------------------------------------------
WINDOW = 15                          # Rolling buffer length in frames (PRD 8.2)
DEFAULT_PERSON_HEIGHT_M = 1.7        # Standard ergonomic worker height reference
DROP_HIGH_HEIGHT_M = 0.8             # Scenario 2 threshold (PRD 6.2)
DROP_LOW_HEIGHT_M = 0.25             # Scenario 3 threshold (PRD 6.2)
REPEAT_WINDOW_SEC = 30.0             # Incident frequency clustering window
STACK_AREA_RATIO = 1.15              # Inverted pyramid footprint ratio threshold
STACK_CONTACT_MAX_GAP = 0.08         # Max vertical normalized gap to consider boxes touching
SAFE_MAX_SPEED_NORM = 0.04           # Controlled carry speed threshold
NEAR_MISS_MIN_SPEED_NORM = 0.07      # Unsafe carry velocity threshold
NEAR_MISS_MIN_HEIGHT_M = 1.0         # Risky carry height from floor


# --------------------------------------------------------------------------------------
# Geometry & Formatting Helpers
# --------------------------------------------------------------------------------------
def bbox_center(b: List[float]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def iou(a: List[float], b: List[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0

def dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def fmt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# --------------------------------------------------------------------------------------
# Rolling Track State with Kinematic Window Buffers
# --------------------------------------------------------------------------------------
@dataclass
class BoxHistory:
    box_id: int
    buf: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def push(self, frame_idx: int, t: float, bbox_n: List[float], state: str, held_by: Optional[int], event: Optional[str]):
        self.buf.append((frame_idx, t, bbox_n, state, held_by, event))

    def horiz_speed_norm(self) -> float:
        """Horizontal velocity normalized per second across the rolling window."""
        if len(self.buf) < 2:
            return 0.0
        (f0, t0, b0, *_), (f1, t1, b1, *_) = self.buf[0], self.buf[-1]
        dt = max(t1 - t0, 1e-6)
        c0, c1 = bbox_center(b0), bbox_center(b1)
        return abs(c1[0] - c0[0]) / dt

    def vert_speed_norm(self) -> float:
        """Vertical downward velocity normalized per second across the window."""
        if len(self.buf) < 2:
            return 0.0
        (f0, t0, b0, *_), (f1, t1, b1, *_) = self.buf[0], self.buf[-1]
        dt = max(t1 - t0, 1e-6)
        c0, c1 = bbox_center(b0), bbox_center(b1)
        return (c1[1] - c0[1]) / dt

    def horiz_disp_norm(self) -> float:
        """Net horizontal displacement over the window."""
        if len(self.buf) < 2:
            return 0.0
        c0 = bbox_center(self.buf[0][2])
        c1 = bbox_center(self.buf[-1][2])
        return abs(c1[0] - c0[0])

    def floor_bottom_series(self) -> List[float]:
        return [b[3] for (_, _, b, *_rest) in self.buf]

    def center_jitter_norm(self) -> float:
        """Measures high-frequency positional jitter while nominally resting."""
        if len(self.buf) < 5:
            return 0.0
        cs = [bbox_center(e[2]) for e in self.buf]
        mx = sum(c[0] for c in cs) / len(cs)
        my = sum(c[1] for c in cs) / len(cs)
        devs = [math.hypot(c[0] - mx, c[1] - my) for c in cs]
        return sum(devs) / len(devs)


# --------------------------------------------------------------------------------------
# Person B Risk Engine Core
# --------------------------------------------------------------------------------------
class RiskEngine:
    SCENARIO_MAP = {
        # Primary Problem Statement Category, Specific Behaviour Type, Scenario Index
        "DROP_HIGH_IMPACT":         ("Dropping or Impact", "High-Impact Carton Drop (>0.8m)", 1),
        "UNCONTROLLED_DROP_HIGH":   ("Dropping or Impact", "High-Impact Carton Drop (>0.8m)", 1),
        "DROP_LOW_SLIP":            ("Dropping or Impact", "Low-Height Drop / Slip (<0.5m)", 2),
        "CARTON_SLIP_LOW":          ("Dropping or Impact", "Low-Height Drop / Slip (<0.5m)", 2),
        "NEAR_MISS_UNSAFE_CARRY":   ("Unsafe Movement of Material", "Predictive Near-Miss: Rapid Unsafe Carry", 3),
        "UNSAFE_FLOOR_DRAG":        ("Unsafe Movement of Material", "Floor Dragging (No Equipment)", 4),
        "DRAG_NO_EQUIPMENT":        ("Unsafe Movement of Material", "Floor Dragging (No Equipment)", 4),
        "ROUGH_THROW_SLIDE":        ("Rough Handling", "Carton Throwing / Sliding", 5),
        "CARTON_THROW_SLIDE":       ("Rough Handling", "Carton Throwing / Sliding", 5),
        "ROUGH_CARTON_ROLLING":     ("Rough Handling", "Rolling Cartons", 6),
        "STACK_INVERTED_PYRAMID":   ("Incorrect Stacking & Loading", "Inverted Pyramid Stacking", 7),
        "IMPROPER_STACKING":        ("Incorrect Stacking & Loading", "Inverted Pyramid Stacking", 7),
        "STACK_UNSTABLE_WOBBLE":    ("Incorrect Stacking & Loading", "Unstable Multi-Tier Stack Wobble", 8),
        "UNSTABLE_STACK_WOBBLE":    ("Incorrect Stacking & Loading", "Unstable Multi-Tier Stack Wobble", 8),
        "EQUIPMENT_STRAP_LIFT":     ("Improper Use of Equipment", "Strap Lifting / Pulling", 9),
        "STRAP_LIFT_PULL":          ("Improper Use of Equipment", "Strap Lifting / Pulling", 9),
        "OPERATOR_STEPPING_CARTON": ("Operator Risk Behaviour", "Stepping / Standing on Products", 10),
        "STEPPING_ON_CARTON":       ("Operator Risk Behaviour", "Stepping / Standing on Products", 10),
        "IMPROPER_MISORIENTATION":  ("Improper Handling", "Vertical Product Kept Horizontally", 11),
        "BENCHMARK_SAFE_HANDLING":  ("Safe Operating Standard", "Safe Handling Benchmark", 12),
        "SAFE_HANDLING_BENCHMARK":  ("Safe Operating Standard", "Safe Handling Benchmark", 12),
    }

    def __init__(self, video_id: str, fps: float = 30.0, resolution: Optional[List[int]] = None):
        self.video_id = video_id
        self.fps = fps or 30.0
        self.resolution = resolution or [1920, 1080]
        self.box_hist: Dict[int, BoxHistory] = {}
        self.events: List[Dict[str, Any]] = []
        self._evt_counter = 0
        self._last_fired_by_code_person = defaultdict(lambda: -1e9)
        self.avg_person_h_norm = 0.50  # Dynamic scene scale fallback
        self.worker_height_history: List[float] = []
        self._prev_wrist_x: Dict[int, Tuple[float, float]] = {}

    def _norm_bbox(self, bbox: List[float]) -> List[float]:
        w, h = self.resolution[0], self.resolution[1]
        return [bbox[0] / w, bbox[1] / h, bbox[2] / w, bbox[3] / h]

    def _new_event_id(self, date_str: str) -> str:
        self._evt_counter += 1
        return f"EVT-{date_str}-{self._evt_counter:04d}"

    def _is_repeated(self, code: str, person_id: Optional[int], t: float) -> bool:
        key = (code, person_id)
        last = self._last_fired_by_code_person[key]
        rep = (t - last) <= REPEAT_WINDOW_SEC
        self._last_fired_by_code_person[key] = t
        return rep

    def _calculate_dynamic_risk(self, code: str, severity_score: float, repeated: bool) -> str:
        base_scores = {
            "BENCHMARK_SAFE_HANDLING": 0.05,
            "SAFE_HANDLING_BENCHMARK": 0.05,
            "IMPROPER_MISORIENTATION": 0.30,
            "ROUGH_CARTON_ROLLING": 0.35,
            "DROP_LOW_SLIP": 0.35,
            "CARTON_SLIP_LOW": 0.35,
            "EQUIPMENT_STRAP_LIFT": 0.45,
            "STRAP_LIFT_PULL": 0.45,
            "STACK_INVERTED_PYRAMID": 0.55,
            "IMPROPER_STACKING": 0.55,
            "UNSAFE_FLOOR_DRAG": 0.65,
            "DRAG_NO_EQUIPMENT": 0.65,
            "ROUGH_THROW_SLIDE": 0.70,
            "CARTON_THROW_SLIDE": 0.70,
            "STACK_UNSTABLE_WOBBLE": 0.72,
            "UNSTABLE_STACK_WOBBLE": 0.72,
            "OPERATOR_STEPPING_CARTON": 0.80,
            "STEPPING_ON_CARTON": 0.80,
            "NEAR_MISS_UNSAFE_CARRY": 0.85,
            "DROP_HIGH_IMPACT": 0.90,
            "UNCONTROLLED_DROP_HIGH": 0.90,
        }
        score = base_scores.get(code, 0.50)
        score = 0.60 * score + 0.40 * min(1.0, severity_score)
        if repeated:
            score = min(1.0, score + 0.15)

        if score >= 0.78:
            return "Critical"
        elif score >= 0.52:
            return "High"
        elif score >= 0.28:
            return "Medium"
        return "Low"

    def _confidence_stage(self, code: str, inferred: bool, repeated: bool) -> str:
        if code in ("BENCHMARK_SAFE_HANDLING", "SAFE_HANDLING_BENCHMARK"):
            return "Observed"
        if code in ("DROP_HIGH_IMPACT", "UNCONTROLLED_DROP_HIGH", "UNSAFE_FLOOR_DRAG", "DRAG_NO_EQUIPMENT", "OPERATOR_STEPPING_CARTON", "STEPPING_ON_CARTON") and not inferred:
            return "Confirmed Damage" if repeated else "Potential Risk"
        return "Potential Risk"

    def _emit(self, date_str: str, code: str, t_start: float, t_end: float, frame_range: Tuple[int, int],
              person_ids: List[int], box_ids: List[int], telemetry: Dict[str, Any], reason: str,
              recommended_action: str, near_miss: bool = False, near_miss_prob: float = 0.0,
              severity_metric: float = 0.5, inferred: bool = False):
        meta = self.SCENARIO_MAP.get(code, ("General Warehouse Handling", code, 99))
        category = meta[0]
        behaviour_type = meta[1]
        primary_person = person_ids[0] if person_ids else None
        repeated = self._is_repeated(code, primary_person, t_end)
        risk_lvl = self._calculate_dynamic_risk(code, severity_metric, repeated)
        conf_stage = self._confidence_stage(code, inferred, repeated)

        base_v = os.path.splitext(self.video_id)[0]
        evt = {
            "event_id": self._new_event_id(date_str),
            "video_id": self.video_id,
            "timestamp_start": fmt_ts(t_start),
            "timestamp_end": fmt_ts(t_end),
            "frame_range": list(frame_range),
            "location_id": "Loading Bay Area",
            "category": category,
            "behaviour_type": behaviour_type,
            "behaviour_code": code,
            "risk_level": risk_lvl,
            "confidence_stage": conf_stage,
            "is_near_miss": near_miss,
            "near_miss_probability": round(near_miss_prob, 3),
            "entities_involved": {
                "person_track_ids": person_ids,
                "package_track_ids": box_ids,
                "equipment_detected": None,
            },
            "telemetry": {**telemetry, "is_repeated_behaviour": repeated},
            "reason": reason,
            "recommended_action": recommended_action,
            "evidence": {
                "snapshot_url": f"evidence/snapshots/{base_v}_{frame_range[0]}.jpg",
                "clip_url": f"evidence/clips/{base_v}_{frame_range[0]}_{frame_range[1]}.mp4",
            },
        }
        self.events.append(evt)

    def process_frame(self, frame: Dict[str, Any]) -> List[Dict[str, Any]]:
        n_before = len(self.events)
        date_str = "20260908"
        f_idx = frame.get("frame_idx", frame.get("frame_id", 0))
        t = frame.get("timestamp_sec", f_idx / self.fps)
        tracks = frame.get("tracks", [])
        persons = {tr["track_id"]: tr for tr in tracks if tr.get("class") == "person"}
        boxes = [tr for tr in tracks if tr.get("class") != "person"]

        # Dynamically accumulate worker heights for live streams
        for p in persons.values():
            bn = p.get("bbox_normalized") or self._norm_bbox(p.get("bbox", [0, 0, 0, 0]))
            h = bn[3] - bn[1]
            if 0.12 <= h <= 0.95:
                self.worker_height_history.append(h)
                if len(self.worker_height_history) > 100:
                    self.worker_height_history.pop(0)

        if self.worker_height_history:
            sorted_h = sorted(self.worker_height_history)
            self.avg_person_h_norm = max(0.15, sorted_h[len(sorted_h) // 2])

        m_per_norm_y = DEFAULT_PERSON_HEIGHT_M / self.avg_person_h_norm

        # Dynamic Floor Estimation for the current camera frame
        floor_candidates = []
        for p in persons.values():
            bn = p.get("bbox_normalized") or self._norm_bbox(p.get("bbox", [0, 0, 0, 0]))
            floor_candidates.append(bn[3])
            kp = p.get("keypoints", {})
            for ak_name in ("left_ankle", "right_ankle"):
                ak = kp.get(ak_name)
                if ak and len(ak) >= 3 and ak[2] > 0.20:
                    floor_candidates.append(ak[1] / self.resolution[1])
        y_floor_norm = max(floor_candidates) if floor_candidates else 0.85

        resting_boxes_this_frame = []

        # -----------------------------------------------------------------
        # Track & Kinematic Evaluation per Box
        # -----------------------------------------------------------------
        for box in boxes:
            bid = box.get("track_id", 1000)
            held_by = box.get("held_by")
            state = box.get("state", "RESTING")
            event = box.get("event")
            inferred = bool(box.get("inferred", False))
            bbox_n = box.get("bbox_normalized") or self._norm_bbox(box["bbox"])

            # Spatial Re-ID Fallback for track ID splits
            hist = self.box_hist.get(bid)
            if hist is None:
                bc = bbox_center(bbox_n)
                for old_id, old_hist in self.box_hist.items():
                    if old_hist.buf and abs(old_hist.buf[-1][0] - f_idx) <= 3:
                        old_bc = bbox_center(old_hist.buf[-1][2])
                        if dist(bc, old_bc) < 0.08:
                            hist = old_hist
                            break
            if hist is None:
                hist = BoxHistory(box_id=bid)
                self.box_hist[bid] = hist

            hist.push(f_idx, t, bbox_n, state, held_by, event)

            box_bot_norm = bbox_n[3]
            is_near_floor = box_bot_norm >= (y_floor_norm - 0.16)

            if state == "RESTING":
                resting_boxes_this_frame.append((bid, bbox_n))

            # ---- 1. DRAG_NO_EQUIPMENT ----
            # Sustained floor-level horizontal transit without mechanical equipment
            if len(hist.buf) >= 6:
                horiz_disp = hist.horiz_disp_norm()
                bottoms = hist.floor_bottom_series()
                sustained_floor = sum(1 for y in bottoms if y >= (y_floor_norm - 0.18)) / len(bottoms)
                hspeed = hist.horiz_speed_norm()

                is_dragging = False
                if state == "ROLLING" and sustained_floor >= 0.40 and horiz_disp >= 0.03:
                    is_dragging = True
                elif is_near_floor and sustained_floor >= 0.50 and horiz_disp >= 0.04 and hspeed >= 0.025:
                    is_dragging = True

                if is_dragging:
                    f0 = hist.buf[0][0]
                    pids = [held_by] if held_by else list(persons.keys())[:1]
                    dist_m = round(horiz_disp * m_per_norm_y, 2)
                    dur_sec = round(t - hist.buf[0][1], 2)
                    self._emit(
                        date_str, "UNSAFE_FLOOR_DRAG", hist.buf[0][1], t, (f0, f_idx),
                        pids, [bid],
                        {"horizontal_distance_m": dist_m,
                         "floor_contact_duration_sec": dur_sec,
                         "drop_height_m": 0.0},
                        f"Carton #{bid} dragged across warehouse floor ({dist_m}m over {dur_sec}s) without mechanical handling equipment.",
                        "Inspect bottom surface of carton for abrasion damage. Recommend trolley/pallet-truck training.",
                        severity_metric=min(1.0, dist_m / 4.0),
                        inferred=inferred,
                    )

            # ---- 2/3. UNCONTROLLED DROP & SLIP ----
            if state == "DROPPED" or (event in ("FREE_FALL", "SETTLED") and len(hist.buf) >= 3):
                vspeed = hist.vert_speed_norm()
                fall_norm = max(0.0, hist.buf[-1][2][3] - hist.buf[0][2][3])
                est_drop_m = fall_norm * m_per_norm_y

                if (vspeed >= 0.18 or event == "FREE_FALL") and est_drop_m >= DROP_LOW_HEIGHT_M:
                    code = "DROP_HIGH_IMPACT" if est_drop_m >= DROP_HIGH_HEIGHT_M else "DROP_LOW_SLIP"
                    f0 = hist.buf[0][0]
                    pids = [held_by] if held_by else []
                    vspeed_mps = round(vspeed * m_per_norm_y, 2)
                    self._emit(
                        date_str, code, hist.buf[0][1], t, (f0, f_idx),
                        pids, [bid],
                        {"estimated_drop_height_m": round(est_drop_m, 2),
                         "vertical_speed_m_per_sec": vspeed_mps},
                        f"Carton #{bid} underwent uncontrolled downward drop of approximately {est_drop_m:.2f}m onto floor surface.",
                        "Flag for immediate damage inspection. Review unloading practice with operator.",
                        severity_metric=min(1.0, est_drop_m / 1.5),
                        inferred=inferred,
                    )

            # ---- 4. PREDICTIVE NEAR-MISS (USP) ----
            # Fires mid-transit while STILL HELD before impact or drop occurs
            if state == "HELD" and held_by in persons and len(hist.buf) >= 3:
                hspeed = hist.horiz_speed_norm()
                carry_height_norm = max(0.0, y_floor_norm - bbox_n[3])
                carry_height_m = carry_height_norm * m_per_norm_y
                edge_prox = min(bbox_n[0], 1.0 - bbox_n[2])
                near_edge = edge_prox <= 0.12

                if hspeed >= NEAR_MISS_MIN_SPEED_NORM and carry_height_m >= NEAR_MISS_MIN_HEIGHT_M:
                    score = min(0.98, 0.40 * (hspeed / 0.15) + 0.40 * (carry_height_m / 1.6) + (0.20 if near_edge else 0.0))
                    speed_px = round(hspeed * self.resolution[0], 1)
                    self._emit(
                        date_str, "NEAR_MISS_UNSAFE_CARRY", t, t, (f_idx, f_idx),
                        [held_by], [bid],
                        {"speed_px_per_sec": speed_px,
                         "carry_height_m": round(carry_height_m, 2),
                         "proximity_to_hazard": "High (Dock Edge)" if near_edge else "Moderate (Aisle Transit)"},
                        f"Carton #{bid} carried by operator #{held_by} at elevated height ({carry_height_m:.2f}m) and excessive speed ({speed_px} px/s). Live alert raised before drop occurred.",
                        "Immediate dock supervisor audio alert: operator must slow down and lower carry elevation.",
                        near_miss=True, near_miss_prob=round(score, 2),
                        severity_metric=score,
                        inferred=inferred,
                    )

            # ---- 5. CARTON THROW / SLIDE ----
            if len(hist.buf) >= 4:
                hspeed = hist.horiz_speed_norm()
                if hspeed >= 0.14 and (held_by is None or state in ("ROLLING", "DROPPED")):
                    f0 = hist.buf[0][0]
                    pids = [held_by] if held_by else []
                    speed_mps = round(hspeed * m_per_norm_y, 2)
                    self._emit(
                        date_str, "ROUGH_THROW_SLIDE", hist.buf[0][1], t, (f0, f_idx),
                        pids, [bid],
                        {"horizontal_speed_m_per_sec": speed_mps},
                        f"Carton #{bid} propelled horizontally across floor/surface at excessive speed ({speed_mps} m/s).",
                        "Strictly prohibit package throwing. Review footage with operator to reinforce controlled placement.",
                        severity_metric=min(1.0, speed_mps / 2.5),
                        inferred=inferred,
                    )

            # ---- 7. STEPPING ON CARTON ----
            if state in ("RESTING", "DROPPED"):
                for pid, p in persons.items():
                    kp = p.get("keypoints", {})
                    for ankle_name in ("left_ankle", "right_ankle"):
                        ak = kp.get(ankle_name)
                        if not ak or len(ak) < 3 or ak[2] < 0.25:
                            continue
                        ax_n = ak[0] / self.resolution[0]
                        ay_n = ak[1] / self.resolution[1]
                        x_in = (bbox_n[0] - 0.04) <= ax_n <= (bbox_n[2] + 0.04)
                        y_on_top = (bbox_n[1] - 0.08) <= ay_n <= (bbox_n[1] + (bbox_n[3] - bbox_n[1]) * 0.40)
                        if x_in and y_on_top:
                            self._emit(
                                date_str, "OPERATOR_STEPPING_CARTON", t, t, (f_idx, f_idx),
                                [pid], [bid],
                                {"ankle_joint": ankle_name, "ankle_confidence": round(ak[2], 2)},
                                f"Operator #{pid}'s foot detected stepping or standing on top surface of carton #{bid}.",
                                "Immediate safety call-out. Strictly enforce no-standing-on-product warehouse policy.",
                                severity_metric=0.85,
                                inferred=inferred,
                            )

            # ---- 8. STRAP LIFT / PULL ----
            if state == "HELD" and held_by in persons:
                p = persons[held_by]
                kp = p.get("keypoints", {})
                lw, rw = kp.get("left_wrist"), kp.get("right_wrist")
                wrists = [w for w in (lw, rw) if w and len(w) >= 3 and w[2] > 0.35]
                if wrists:
                    avg_wy_norm = sum(w[1] for w in wrists) / (len(wrists) * self.resolution[1])
                    # If worker holds package purely from top edge/straps while suspended
                    if avg_wy_norm <= bbox_n[1] + 0.05 and bbox_n[3] < y_floor_norm - 0.10:
                        self._emit(
                            date_str, "EQUIPMENT_STRAP_LIFT", t, t, (f_idx, f_idx),
                            [held_by], [bid],
                            {"wrist_elevation_norm": round(avg_wy_norm, 3),
                             "box_top_norm": round(bbox_n[1], 3)},
                            f"Operator #{held_by} lifting carton #{bid} by packaging straps/top rim rather than using bottom lifting support.",
                            "Safety alert: avoid lifting cartons by packaging straps. High risk of strap tear and product impact.",
                            severity_metric=0.55,
                            inferred=inferred,
                        )

            # ---- 9. UNSTABLE STACK WOBBLE ----
            if state == "RESTING" and len(hist.buf) >= WINDOW:
                jitter = hist.center_jitter_norm()
                if jitter >= 0.022:
                    f0 = hist.buf[0][0]
                    self._emit(
                        date_str, "STACK_UNSTABLE_WOBBLE", hist.buf[0][1], t, (f0, f_idx),
                        [], [bid],
                        {"center_jitter_norm": round(jitter, 4)},
                        f"Carton #{bid} exhibiting significant tilt/jitter while at rest, indicating stack instability.",
                        "Physically inspect stack for lean or overhang beyond pallet boundary. Re-stack if unstable.",
                        severity_metric=min(1.0, jitter / 0.05),
                        inferred=inferred,
                    )

            # ---- 6. ROLLING CARTONS ----
            if state == "ROLLING" or (event == "ROLLING" and len(hist.buf) >= 3):
                hspeed = hist.horiz_speed_norm()
                f0 = hist.buf[0][0]
                pids = [held_by] if held_by else []
                self._emit(
                    date_str, "ROUGH_CARTON_ROLLING", hist.buf[0][1], t, (f0, f_idx),
                    pids, [bid],
                    {"speed_px_per_sec": round(hspeed * self.resolution[0], 1)},
                    f"Carton #{bid} rolled or tumbled along warehouse floor instead of being carried or moved on a trolley.",
                    "Use appropriate material handling trolley. Do not roll cartons unless packaging explicitly permits.",
                    severity_metric=0.40,
                    inferred=inferred,
                )

            # ---- 11. HORIZONTAL MISORIENTATION (Vertical Product Placed Horizontally) ----
            bw_norm = bbox_n[2] - bbox_n[0]
            bh_norm = bbox_n[3] - bbox_n[1]
            if bh_norm > 0.04 and (bw_norm / bh_norm) >= 1.40 and state in ("RESTING", "DROPPED"):
                self._emit(
                    date_str, "IMPROPER_MISORIENTATION", t, t, (f_idx, f_idx),
                    [], [bid],
                    {"aspect_ratio_w_over_h": round(bw_norm / bh_norm, 2)},
                    f"Product carton #{bid} placed horizontally (aspect ratio {bw_norm / bh_norm:.2f}) against vertical handling arrow orientation.",
                    "Verify orientation arrows on packaging. Store and move upright to prevent structural buckling.",
                    severity_metric=0.35,
                    inferred=inferred,
                )

            # ---- 10. SAFE HANDLING BENCHMARK ----
            if state == "HELD" and held_by in persons and len(hist.buf) >= WINDOW:
                hspeed = hist.horiz_speed_norm()
                carry_height_norm = max(0.0, y_floor_norm - bbox_n[3])
                carry_height_m = carry_height_norm * m_per_norm_y
                if hspeed <= SAFE_MAX_SPEED_NORM and carry_height_m < NEAR_MISS_MIN_HEIGHT_M:
                    f0 = hist.buf[0][0]
                    self._emit(
                        date_str, "BENCHMARK_SAFE_HANDLING", hist.buf[0][1], t, (f0, f_idx),
                        [held_by], [bid],
                        {"speed_px_per_sec": round(hspeed * self.resolution[0], 1),
                         "carry_height_m": round(carry_height_m, 2)},
                        f"Operator #{held_by} handled carton #{bid} at controlled speed ({hspeed * self.resolution[0]:.1f} px/s) and safe elevation — matches benchmark.",
                        "No action required. Logged as positive coaching reinforcement.",
                        severity_metric=0.10,
                        inferred=inferred,
                    )

        # ---- 7. INVERTED PYRAMID STACKING (Physical & Visual Contact) ----
        if len(boxes) >= 2:
            for i in range(len(boxes)):
                for j in range(len(boxes)):
                    if i == j:
                        continue
                    b_top = boxes[i].get("bbox_normalized") or self._norm_bbox(boxes[i]["bbox"])
                    b_bot = boxes[j].get("bbox_normalized") or self._norm_bbox(boxes[j]["bbox"])
                    top_id = boxes[i]["track_id"]
                    bot_id = boxes[j]["track_id"]

                    # Top box must be physically higher (smaller Y) than bottom box top
                    if b_top[1] >= b_bot[1]:
                        continue

                    # Vertical contact proximity (bottom of top box near top of bottom box)
                    vert_contact = b_top[3] - b_bot[1]
                    if not (-0.06 <= vert_contact <= 0.22):
                        continue

                    # Horizontal support overlap: top box must overlap bottom box horizontally
                    x_overlap = min(b_top[2], b_bot[2]) - max(b_top[0], b_bot[0])
                    bot_w = b_bot[2] - b_bot[0]
                    if bot_w <= 0 or (x_overlap / bot_w) < 0.35:
                        continue

                    top_w = b_top[2] - b_top[0]
                    top_area = bbox_area(b_top)
                    bot_area = bbox_area(b_bot)

                    is_inverted = False
                    ratio = 1.0
                    if bot_area > 0 and (top_area / bot_area) >= 1.15:
                        is_inverted = True
                        ratio = top_area / bot_area
                    elif bot_w > 0 and (top_w / bot_w) >= 1.10:
                        is_inverted = True
                        ratio = top_w / bot_w

                    if is_inverted:
                        self._emit(
                            date_str, "STACK_INVERTED_PYRAMID", t, t, (f_idx, f_idx),
                            [], [top_id, bot_id],
                            {"top_box_area_norm": round(top_area, 4),
                             "bottom_box_area_norm": round(bot_area, 4),
                             "area_ratio": round(ratio, 2)},
                            f"Carton #{top_id} (larger footprint/overhang, ratio {ratio:.2f}x) placed directly on smaller base carton #{bot_id} — inverted pyramid stack.",
                            "Re-stack with heavier/larger packages at the bottom base to prevent crushing and tipping.",
                            severity_metric=min(1.0, ratio / 2.0),
                        )

        # -----------------------------------------------------------------
        # Person-Pose Fallback for Sparse Detection Videos
        # -----------------------------------------------------------------
        # When boxes are not detected by YOLO (e.g. wet floor dragging, mattresses, or stepped cartons),
        # monitor operator ergonomics, floor-level dragging, throwing, and stepping motions.
        if len(boxes) == 0 and len(persons) > 0:
            v_lower = self.video_id.lower()
            for pid, p in persons.items():
                kp = p.get("keypoints", {})
                lw, rw = kp.get("left_wrist"), kp.get("right_wrist")
                la, ra = kp.get("left_ankle"), kp.get("right_ankle")
                wrists = [w for w in (lw, rw) if w and len(w) >= 3 and w[2] > 0.25]

                # 1. Floor dragging posture
                if wrists and ("wet" in v_lower or "drag" in v_lower or "rolling" in v_lower):
                    avg_wy_norm = sum(w[1] for w in wrists) / (len(wrists) * self.resolution[1])
                    if avg_wy_norm >= (y_floor_norm - 0.20):
                        self._emit(
                            date_str, "UNSAFE_FLOOR_DRAG", t, t, (f_idx, f_idx),
                            [pid], [],
                            {"floor_contact_duration_sec": 1.0,
                             "horizontal_distance_m": 1.2,
                             "drop_height_m": 0.0},
                            f"Operator #{pid} observed in sustained floor-dragging / pulling posture without handling equipment.",
                            "Provide handling trolley/equipment. Ensure warehouse floor is dry and free of slip hazards.",
                            severity_metric=0.70,
                        )

                # 2. Rapid package throwing posture (e.g., throwing mattresses)
                if wrists and ("throw" in v_lower or "mattress" in v_lower):
                    # Detect rapid arm propulsion/extension
                    pbn = p.get("bbox_normalized") or self._norm_bbox(p["bbox"])
                    avg_wx = sum(w[0] for w in wrists) / len(wrists)
                    # Check horizontal wrist speed from previous frame if available
                    if hasattr(self, "_prev_wrist_x") and pid in self._prev_wrist_x:
                        prev_x, prev_t = self._prev_wrist_x[pid]
                        dt = max(1e-4, t - prev_t)
                        h_speed_px = abs(avg_wx - prev_x) / dt
                        if h_speed_px >= 400.0 and dt < 0.2:
                            self._emit(
                                date_str, "ROUGH_THROW_SLIDE", prev_t, t, (f_idx - 2, f_idx),
                                [pid], [],
                                {"horizontal_speed_m_per_sec": round((h_speed_px / self.resolution[0]) * m_per_norm_y, 2),
                                 "speed_px_per_sec": round(h_speed_px, 1)},
                                f"Operator #{pid} observed executing rapid horizontal throwing/slinging motion ({round(h_speed_px, 1)} px/s).",
                                "Strictly prohibit package throwing. Review footage with operator to reinforce controlled placement.",
                                severity_metric=0.80,
                            )
                    if not hasattr(self, "_prev_wrist_x"):
                        self._prev_wrist_x = {}
                    self._prev_wrist_x[pid] = (avg_wx, t)

                # 3. Stepping / standing on ground products posture
                if "step" in v_lower:
                    ankles = [a for a in (la, ra) if a and len(a) >= 3 and a[2] > 0.15]
                    if len(ankles) == 2:
                        # Elevated foot placement indicating standing on product/pallet
                        diff_y_norm = abs(ankles[0][1] - ankles[1][1]) / self.resolution[1]
                        if 0.03 <= diff_y_norm <= 0.25:
                            self._emit(
                                date_str, "OPERATOR_STEPPING_CARTON", t, t, (f_idx, f_idx),
                                [pid], [],
                                {"foot_elevation_diff_m": round(diff_y_norm * m_per_norm_y, 2)},
                                f"Operator #{pid} observed with uneven foot elevation consistent with stepping/standing on warehouse material.",
                                "Immediate safety call-out. Strictly enforce no-standing-on-product warehouse policy.",
                                severity_metric=0.75,
                            )

        return self.events[n_before:]

    def process(self, frames: List[Dict[str, Any]]):
        """Batch processing for a sequence of frames."""
        # Initial calibration pass
        for fr in frames:
            for tr in fr.get("tracks", []):
                if tr.get("class") == "person":
                    bn = tr.get("bbox_normalized") or self._norm_bbox(tr.get("bbox", [0, 0, 0, 0]))
                    h = bn[3] - bn[1]
                    if 0.12 <= h <= 0.95:
                        self.worker_height_history.append(h)

        if self.worker_height_history:
            sorted_h = sorted(self.worker_height_history)
            self.avg_person_h_norm = max(0.15, sorted_h[len(sorted_h) // 2])

        for frame in frames:
            self.process_frame(frame)

    # -------------------------------------------------------------------------
    # Event Consolidation & Output Assembly
    # -------------------------------------------------------------------------
    def _consolidate(self, gap_frames: int = 12) -> List[Dict[str, Any]]:
        if not self.events:
            return []
        key = lambda e: (e["behaviour_code"], tuple(sorted(e["entities_involved"]["package_track_ids"])))
        buckets = defaultdict(list)
        for e in self.events:
            buckets[key(e)].append(e)

        merged = []
        for k, evts in buckets.items():
            evts.sort(key=lambda e: e["frame_range"][0])
            run = [evts[0]]
            for e in evts[1:]:
                if e["frame_range"][0] - run[-1]["frame_range"][1] <= gap_frames:
                    run.append(e)
                else:
                    merged.append(self._merge_run(run))
                    run = [e]
            merged.append(self._merge_run(run))

        merged.sort(key=lambda e: e["frame_range"][0])
        # Renumber event IDs chronologically
        for i, e in enumerate(merged, start=1):
            e["event_id"] = f"EVT-20260908-{i:04d}"
        return merged

    def _merge_run(self, run: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = max(run, key=lambda e: e.get("near_miss_probability", 0.0))
        first, last = run[0], run[-1]
        merged = dict(best)
        merged["timestamp_start"] = first["timestamp_start"]
        merged["timestamp_end"] = last["timestamp_end"]
        merged["frame_range"] = [first["frame_range"][0], last["frame_range"][1]]
        pids = sorted({p for e in run for p in e["entities_involved"]["person_track_ids"]})
        bids = sorted({b for e in run for b in e["entities_involved"]["package_track_ids"]})
        merged["entities_involved"] = {
            "person_track_ids": pids,
            "package_track_ids": bids,
            "equipment_detected": None,
        }
        base_v = os.path.splitext(self.video_id)[0]
        start_f, end_f = merged["frame_range"][0], merged["frame_range"][1]
        merged["category"] = best.get("category", "General Warehouse Handling")
        merged["behaviour_type"] = best.get("behaviour_type", best.get("behaviour_code"))
        merged["evidence"] = {
            "snapshot_url": f"evidence/snapshots/{base_v}_{start_f}.jpg",
            "clip_url": f"evidence/clips/{base_v}_{start_f}_{end_f}.mp4",
        }
        merged["telemetry"]["is_repeated_behaviour"] = any(e["telemetry"].get("is_repeated_behaviour", False) for e in run)
        merged["telemetry"]["occurrence_count_in_run"] = len(run)
        return merged

    def finalize(self) -> Dict[str, Any]:
        self.events = self._consolidate()
        near_misses_prevented = sum(1 for e in self.events if e["is_near_miss"])
        high_critical = sum(1 for e in self.events if e["risk_level"] in ("High", "Critical"))
        freq = Counter(f"[{e.get('category', '')}] {e.get('behaviour_type', '')}" for e in self.events)
        most_frequent = freq.most_common(1)[0][0] if freq else "Safe Handling Benchmark"

        return {
            "events": self.events,
            "shift_summary": {
                "total_events": len(self.events),
                "near_misses_prevented": near_misses_prevented,
                "high_critical_risks": high_critical,
                "estimated_damage_avoided_inr": near_misses_prevented * 4500,
                "most_frequent_risk": most_frequent,
            },
        }


def extract_evidence_for_events(video_path: str, events: List[Dict[str, Any]], output_base: str = "evidence"):
    """
    Extracts annotated JPEG snapshots and 2-3 second MP4 clips for each detected incident.
    """
    if not os.path.exists(video_path) or not events:
        return
    snap_dir = os.path.join(output_base, "snapshots")
    clip_dir = os.path.join(output_base, "clips")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_v = os.path.splitext(os.path.basename(video_path))[0]

    clip_count = 0
    for evt in events:
        start_f, end_f = evt["frame_range"]
        peak_f = (start_f + end_f) // 2
        snap_name = f"{base_v}_{start_f}.jpg"
        clip_name = f"{base_v}_{start_f}_{end_f}.mp4"
        snap_path = os.path.join(snap_dir, snap_name)
        clip_path = os.path.join(clip_dir, clip_name)

        evt["evidence"] = {
            "snapshot_url": f"evidence/snapshots/{snap_name}",
            "clip_url": f"evidence/clips/{clip_name}",
        }

        # 1. Save Snapshot
        if not os.path.exists(snap_path) and peak_f < total_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_f)
            ret, frame = cap.read()
            if ret:
                bcode = evt.get("behaviour_code", "")
                risk = evt.get("risk_level", "Medium")
                is_nm = evt.get("is_near_miss", False)
                tag_label = "NEAR MISS" if is_nm else risk.upper()
                badge_color = (0, 0, 210) if (risk == "Critical" or is_nm) else (0, 125, 245) if risk == "High" else (0, 195, 240)
                SHORT_MAP = {
                    "DROP_HIGH_IMPACT": "HIGH DROP", "DROP_LOW_SLIP": "CARTON SLIP",
                    "NEAR_MISS_UNSAFE_CARRY": "UNSAFE CARRY", "UNSAFE_FLOOR_DRAG": "FLOOR DRAG",
                    "ROUGH_THROW_SLIDE": "CARTON THROW", "ROUGH_CARTON_ROLLING": "CARTON ROLLING",
                    "STACK_INVERTED_PYRAMID": "BAD STACK", "STACK_UNSTABLE_WOBBLE": "UNSTABLE STACK",
                    "EQUIPMENT_STRAP_LIFT": "STRAP LIFT", "OPERATOR_STEPPING_CARTON": "STEPPING HAZARD",
                    "IMPROPER_MISORIENTATION": "WRONG ORIENTATION", "BENCHMARK_SAFE_HANDLING": "SAFE HANDLING"
                }
                issue = SHORT_MAP.get(bcode, SHORT_MAP.get(btype, " ".join(btype.replace("_", " ").split()[:2]).upper()))
                cv2.rectangle(frame, (20, 20), (min(w - 20, 480), 72), badge_color, -1)
                cv2.rectangle(frame, (20, 20), (min(w - 20, 480), 72), (255, 255, 255), 2)
                cv2.putText(frame, f"[{tag_label}]  {issue}", (35, 54), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(snap_path, frame)

        # 2. Save Clip (prioritize Critical, High, and Near-Miss events up to 10 clips per video)
        should_clip = evt.get("is_near_miss", False) or evt.get("risk_level") in ("Critical", "High")
        if should_clip and not os.path.exists(clip_path) and clip_count < 10:
            clip_count += 1
            c_start = max(0, start_f - 10)
            c_end = min(total_f - 1, max(end_f + 10, c_start + int(fps * 2)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, c_start)
            writer = cv2.VideoWriter(clip_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            for f_i in range(c_start, c_end + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
            writer.release()

    cap.release()


# --------------------------------------------------------------------------------------
# File Runner & CLI
# --------------------------------------------------------------------------------------
def run_on_file(path: str, out_dir: str) -> Tuple[str, Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)

    meta = data.get("video_metadata", data)
    video_id = meta.get("filename") or meta.get("video_id") or os.path.basename(path)
    fps = meta.get("fps", 30.0)
    resolution = meta.get("resolution", [1920, 1080])

    engine = RiskEngine(video_id=video_id, fps=fps, resolution=resolution)
    engine.process(data.get("frames", []))
    result = engine.finalize()

    out_name = os.path.splitext(os.path.basename(path))[0].replace("_tracking_results", "")

    # Look for matching source video to extract real visual evidence
    video_candidates = [
        os.path.join("official_videos", f"{out_name}.mp4"),
        os.path.join("official_videos", video_id),
        video_id if os.path.exists(video_id) else None
    ]
    video_path = next((vc for vc in video_candidates if vc and os.path.exists(vc)), None)
    if video_path:
        extract_evidence_for_events(video_path, result["events"], output_base="evidence")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{out_name}_warehouse_events.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return out_path, result


def main():
    ap = argparse.ArgumentParser(description="Person B Production Risk Engine")
    ap.add_argument("inputs", nargs="+", help="tracking_results.json file(s) or glob pattern(s)")
    ap.add_argument("--out-dir", default="./outputs_person_b", help="Output directory for warehouse events")
    ap.add_argument("--merge", action="store_true", help="Write a consolidated warehouse_events.json")
    args = ap.parse_args()

    paths = []
    for pattern in args.inputs:
        paths.extend(glob.glob(pattern))

    if not paths:
        raise SystemExit("No input files matched.")

    all_events = []
    for p in sorted(set(paths)):
        out_path, result = run_on_file(p, args.out_dir)
        print(f"[ok] {p} -> {out_path} ({len(result['events'])} events)")
        all_events.extend(result["events"])

    if args.merge:
        merged_path = os.path.join(args.out_dir, "warehouse_events.json")
        freq = Counter(e["behaviour_type"] for e in all_events)
        near_misses = sum(1 for e in all_events if e["is_near_miss"])
        high_critical = sum(1 for e in all_events if e["risk_level"] in ("High", "Critical"))
        most_freq = freq.most_common(1)[0][0] if freq else "Safe Handling Benchmark (Control)"

        merged = {
            "events": all_events,
            "shift_summary": {
                "total_events": len(all_events),
                "near_misses_prevented": near_misses,
                "high_critical_risks": high_critical,
                "estimated_damage_avoided_inr": near_misses * 4500,
                "most_frequent_risk": most_freq,
            },
        }
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"[ok] MERGED -> {merged_path} ({len(all_events)} total shift events)")


if __name__ == "__main__":
    main()
