"""
Person B — Behaviour Rules + Risk Scoring Engine + Predictive Near-Miss (USP)
Godrej Warehouse Video Intelligence Hackathon | graVITas 2026
IMPROVED & DEFENSIVE IMPLEMENTATION
"""

from __future__ import annotations
import json
import math
import argparse
import glob
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field

WINDOW = 15
DEFAULT_PERSON_HEIGHT_M = 1.7
DROP_HIGH_HEIGHT_M = 0.8
DROP_LOW_HEIGHT_M = 0.3
REPEAT_WINDOW_SEC = 30.0

def bbox_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def bbox_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def fmt_ts(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

@dataclass
class BoxHistory:
    box_id: int
    buf: deque = field(default_factory=lambda: deque(maxlen=WINDOW))

    def push(self, frame_idx, t, bbox_n, state, held_by, event):
        self.buf.append((frame_idx, t, bbox_n, state, held_by, event))

    def horiz_speed_norm(self):
        if len(self.buf) < 2:
            return 0.0
        (f0, t0, b0, *_), (f1, t1, b1, *_) = self.buf[0], self.buf[-1]
        dt = max(t1 - t0, 1e-6)
        c0, c1 = bbox_center(b0), bbox_center(b1)
        return abs(c1[0] - c0[0]) / dt

    def vert_speed_norm(self):
        if len(self.buf) < 2:
            return 0.0
        (f0, t0, b0, *_), (f1, t1, b1, *_) = self.buf[0], self.buf[-1]
        dt = max(t1 - t0, 1e-6)
        c0, c1 = bbox_center(b0), bbox_center(b1)
        return (c1[1] - c0[1]) / dt

    def horiz_disp_norm(self):
        if len(self.buf) < 2:
            return 0.0
        c0 = bbox_center(self.buf[0][2])
        c1 = bbox_center(self.buf[-1][2])
        return abs(c1[0] - c0[0])

    def floor_bottom_series(self):
        return [b[3] for (_, _, b, *_rest) in self.buf]

    def center_jitter_norm(self):
        if len(self.buf) < 5:
            return 0.0
        cs = [bbox_center(e[2]) for e in self.buf]
        mx = sum(c[0] for c in cs) / len(cs)
        my = sum(c[1] for c in cs) / len(cs)
        devs = [math.hypot(c[0] - mx, c[1] - my) for c in cs]
        return sum(devs) / len(devs)

class ImprovedRiskEngine:
    SCENARIO_MAP = {
        "DRAG_NO_EQUIPMENT":        ("Carton / KD Floor Dragging",                1),
        "UNCONTROLLED_DROP_HIGH":   ("High-Impact Carton Drop (>0.8m)",           2),
        "CARTON_SLIP_LOW":          ("Low-Height Drop / Carton Slip (<0.5m)",     3),
        "NEAR_MISS_UNSAFE_CARRY":   ("Predictive Near-Miss: Rapid Unsafe Carry",  4),
        "CARTON_THROW_SLIDE":       ("Carton Throwing / Sliding",                 5),
        "IMPROPER_STACKING":        ("Improper Stacking (Inverted Pyramid)",      6),
        "STEPPING_ON_CARTON":       ("Stepping / Standing on Carton",             7),
        "STRAP_LIFT_PULL":          ("Strap Lifting / Pulling",                   8),
        "UNSTABLE_STACK_WOBBLE":    ("Unstable Multi-Tier Stack Wobble",          9),
        "SAFE_HANDLING_BENCHMARK":  ("Safe Handling Benchmark (Control)",         10),
    }

    def __init__(self, video_id, fps, resolution):
        self.video_id = video_id
        self.fps = fps or 30.0
        self.resolution = resolution or [1920, 1080]
        self.box_hist: dict[int, BoxHistory] = {}
        self.events = []
        self._evt_counter = 0
        self._last_fired_by_code_person = defaultdict(lambda: -1e9)
        self.avg_person_h_norm = 0.5  # dynamic fallback calibration

    def _norm_bbox(self, bbox, res):
        return [bbox[0] / res[0], bbox[1] / res[1], bbox[2] / res[0], bbox[3] / res[1]]

    def _new_event_id(self, date_str):
        self._evt_counter += 1
        return f"EVT-{date_str}-{self._evt_counter:04d}"

    def _is_repeated(self, code, person_id, t):
        key = (code, person_id)
        last = self._last_fired_by_code_person[key]
        rep = (t - last) <= REPEAT_WINDOW_SEC
        self._last_fired_by_code_person[key] = t
        return rep

    def _risk_level(self, code):
        return {
            "DRAG_NO_EQUIPMENT": "High",
            "UNCONTROLLED_DROP_HIGH": "Critical",
            "CARTON_SLIP_LOW": "Medium",
            "NEAR_MISS_UNSAFE_CARRY": "Critical",
            "CARTON_THROW_SLIDE": "High",
            "IMPROPER_STACKING": "Medium",
            "STEPPING_ON_CARTON": "High",
            "STRAP_LIFT_PULL": "Medium",
            "UNSTABLE_STACK_WOBBLE": "High",
            "SAFE_HANDLING_BENCHMARK": "Low",
        }[code]

    def _confidence_stage(self, code, inferred, repeated):
        if code == "SAFE_HANDLING_BENCHMARK":
            return "Observed"
        if code in ("UNCONTROLLED_DROP_HIGH", "CARTON_SLIP_LOW") and not inferred:
            return "Confirmed Damage" if repeated else "Potential Risk"
        return "Potential Risk"

    def _emit(self, date_str, code, t_start, t_end, frame_range, person_ids, box_ids,
              telemetry, reason, recommended_action, near_miss=False, near_miss_prob=0.0,
              inferred=False):
        behaviour_type, _ = self.SCENARIO_MAP[code]
        primary_person = person_ids[0] if person_ids else None
        repeated = self._is_repeated(code, primary_person, t_end)
        evt = {
            "event_id": self._new_event_id(date_str),
            "video_id": self.video_id,
            "timestamp_start": fmt_ts(t_start),
            "timestamp_end": fmt_ts(t_end),
            "frame_range": list(frame_range),
            "location_id": "Loading Bay Area",
            "behaviour_type": behaviour_type,
            "behaviour_code": code,
            "risk_level": self._risk_level(code),
            "confidence_stage": self._confidence_stage(code, inferred, repeated),
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
                "snapshot_url": f"/evidence/snapshots/{self.video_id}_{frame_range[0]}.jpg",
                "clip_url": f"/evidence/clips/{self.video_id}_{frame_range[0]}_{frame_range[1]}.mp4",
            },
        }
        self.events.append(evt)

    def process(self, frames):
        date_str = "20260908"

        # Pre-pass: estimate average person height for scene calibration
        p_heights = []
        for fr in frames:
            for tr in fr.get("tracks", []):
                if tr.get("class") == "person":
                    bn = tr.get("bbox_normalized")
                    if bn:
                        p_heights.append(bn[3] - bn[1])
        if p_heights:
            p_heights.sort()
            # median height
            self.avg_person_h_norm = max(0.15, p_heights[len(p_heights) // 2])

        # meters per normalized vertical unit
        m_per_norm_y = DEFAULT_PERSON_HEIGHT_M / self.avg_person_h_norm

        for frame in frames:
            f_idx = frame.get("frame_idx", frame.get("frame_id"))
            t = frame["timestamp_sec"]
            tracks = frame["tracks"]
            persons = {tr["track_id"]: tr for tr in tracks if tr["class"] == "person"}
            boxes = [tr for tr in tracks if tr["class"] != "person"]

            # Dynamic Floor Estimation for this frame
            floor_candidates = []
            for p in persons.values():
                bn = p.get("bbox_normalized")
                if bn: floor_candidates.append(bn[3])
                kp = p.get("keypoints", {})
                for ak_name in ("left_ankle", "right_ankle"):
                    ak = kp.get(ak_name)
                    if ak and ak[2] > 0.25:
                        floor_candidates.append(ak[1] / self.resolution[1])
            y_floor_norm = max(floor_candidates) if floor_candidates else 0.85

            resting_boxes_this_frame = []

            for box in boxes:
                bid = box["track_id"]
                held_by = box.get("held_by")
                state = box.get("state")
                event = box.get("event")
                inferred = bool(box.get("inferred", False))
                bbox_n = box.get("bbox_normalized") or self._norm_bbox(box["bbox"], self.resolution)

                hist = self.box_hist.setdefault(bid, BoxHistory(box_id=bid))
                hist.push(f_idx, t, bbox_n, state, held_by, event)

                box_bot_norm = bbox_n[3]
                is_near_floor = box_bot_norm >= (y_floor_norm - 0.15)

                if state == "RESTING":
                    resting_boxes_this_frame.append((bid, bbox_n))

                # ---- 1. DRAG_NO_EQUIPMENT ----
                # Detects if box is dragged along floor (either ROLLING, or moving while near floor)
                if len(hist.buf) >= 6:
                    horiz_disp = hist.horiz_disp_norm()
                    bottoms = hist.floor_bottom_series()
                    sustained_floor = sum(1 for y in bottoms if y >= (y_floor_norm - 0.18)) / len(bottoms)
                    hspeed = hist.horiz_speed_norm()
                    
                    is_dragging = False
                    if state == "ROLLING" and sustained_floor >= 0.5 and horiz_disp >= 0.03:
                        is_dragging = True
                    elif is_near_floor and sustained_floor >= 0.6 and horiz_disp >= 0.04 and hspeed >= 0.03:
                        # Moving horizontally along floor without lift equipment
                        is_dragging = True

                    if is_dragging:
                        f0 = hist.buf[0][0]
                        person_ids = [held_by] if held_by else list(persons.keys())[:1]
                        dist_m = round(horiz_disp * m_per_norm_y, 2)
                        dur_sec = round(t - hist.buf[0][1], 2)
                        self._emit(
                            date_str, "DRAG_NO_EQUIPMENT", hist.buf[0][1], t, (f0, f_idx),
                            person_ids, [bid],
                            {"horizontal_distance_m": dist_m,
                             "floor_contact_duration_sec": dur_sec,
                             "drop_height_m": 0.0},
                            f"Carton #{bid} dragged across warehouse floor ({dist_m}m over {dur_sec}s) without mechanical handling equipment.",
                            "Inspect bottom surface of carton for abrasion damage. Recommend trolley/pallet-truck training.",
                            inferred=inferred,
                        )

                # ---- 2/3. DROP / SLIP (Calibrated Drop Height) ----
                if state == "DROPPED" or (event in ("FREE_FALL", "SETTLED") and len(hist.buf) >= 3):
                    vspeed = hist.vert_speed_norm()
                    fall_norm = max(0.0, hist.buf[-1][2][3] - hist.buf[0][2][3])
                    est_drop_m = fall_norm * m_per_norm_y

                    if (vspeed >= 0.20 or event == "FREE_FALL") and est_drop_m >= DROP_LOW_HEIGHT_M:
                        code = "UNCONTROLLED_DROP_HIGH" if est_drop_m >= DROP_HIGH_HEIGHT_M else "CARTON_SLIP_LOW"
                        f0 = hist.buf[0][0]
                        person_ids = [held_by] if held_by else []
                        self._emit(
                            date_str, code, hist.buf[0][1], t, (f0, f_idx),
                            person_ids, [bid],
                            {"estimated_drop_height_m": round(est_drop_m, 2),
                             "vertical_speed_m_per_sec": round(vspeed * m_per_norm_y, 2)},
                            f"Carton #{bid} underwent uncontrolled downward drop of approximately {est_drop_m:.2f}m onto floor surface.",
                            "Flag for damage inspection. Review handling technique with operator.",
                            inferred=inferred,
                        )

                # ---- 5. THROW / SLIDE ----
                if len(hist.buf) >= 4:
                    hspeed = hist.horiz_speed_norm()
                    # High horizontal speed while unheld or rolling
                    if hspeed >= 0.15 and (held_by is None or state in ("ROLLING", "DROPPED")):
                        f0 = hist.buf[0][0]
                        person_ids = [held_by] if held_by else []
                        speed_mps = round(hspeed * m_per_norm_y, 2)
                        self._emit(
                            date_str, "CARTON_THROW_SLIDE", hist.buf[0][1], t, (f0, f_idx),
                            person_ids, [bid],
                            {"horizontal_speed_m_per_sec": speed_mps},
                            f"Carton #{bid} propelled horizontally across floor/surface at excessive speed ({speed_mps} m/s).",
                            "Review footage to coach on safe transfer technique. Strictly prohibit package throwing.",
                            inferred=inferred,
                        )

                # ---- 8. STRAP LIFT / PULL (Scenario 8) ----
                if state == "HELD" and held_by in persons:
                    p = persons[held_by]
                    kp = p.get("keypoints", {})
                    lw = kp.get("left_wrist")
                    rw = kp.get("right_wrist")
                    wrists = [w for w in (lw, rw) if w and w[2] > 0.35]
                    if wrists:
                        avg_wy_norm = sum(w[1] for w in wrists) / (len(wrists) * self.resolution[1])
                        # If worker is holding carton purely from the very top edge or straps
                        if avg_wy_norm <= bbox_n[1] + 0.05 and bbox_n[3] < y_floor_norm - 0.10:
                            self._emit(
                                date_str, "STRAP_LIFT_PULL", t, t, (f_idx, f_idx),
                                [held_by], [bid],
                                {"wrist_elevation_norm": round(avg_wy_norm, 3),
                                 "box_top_norm": round(bbox_n[1], 3)},
                                f"Operator #{held_by} lifting carton #{bid} by packaging straps/top rim rather than using proper lifting points or bottom support.",
                                "Safety alert: avoid lifting cartons by packaging straps. High risk of strap breakage and product damage.",
                                inferred=inferred,
                            )

                # ---- 7. STEPPING ON CARTON ----
                if state in ("RESTING", "DROPPED"):
                    for pid, p in persons.items():
                        kp = p.get("keypoints", {})
                        for ankle_name in ("left_ankle", "right_ankle"):
                            ak = kp.get(ankle_name)
                            if not ak or ak[2] < 0.25:
                                continue
                            ax_n = ak[0] / self.resolution[0]
                            ay_n = ak[1] / self.resolution[1]
                            # Tolerance on box top surface
                            x_in = (bbox_n[0] - 0.04) <= ax_n <= (bbox_n[2] + 0.04)
                            y_on_top = (bbox_n[1] - 0.08) <= ay_n <= (bbox_n[1] + (bbox_n[3] - bbox_n[1]) * 0.4)
                            if x_in and y_on_top:
                                self._emit(
                                    date_str, "STEPPING_ON_CARTON", t, t, (f_idx, f_idx),
                                    [pid], [bid],
                                    {"ankle_joint": ankle_name, "ankle_confidence": round(ak[2], 2)},
                                    f"Operator #{pid}'s foot detected stepping or standing on top surface of carton #{bid}.",
                                    "Immediate safety call-out. Strictly enforce no-standing-on-product warehouse policy.",
                                    inferred=inferred,
                                )

                # ---- 9. UNSTABLE STACK WOBBLE ----
                if state == "RESTING" and len(hist.buf) >= WINDOW:
                    jitter = hist.center_jitter_norm()
                    if jitter >= 0.025:  # filter out detector noise
                        f0 = hist.buf[0][0]
                        self._emit(
                            date_str, "UNSTABLE_STACK_WOBBLE", hist.buf[0][1], t, (f0, f_idx),
                            [], [bid],
                            {"center_jitter_norm": round(jitter, 4)},
                            f"Carton #{bid} exhibiting significant tilt/jitter while at rest, indicating stack instability.",
                            "Physically inspect stack for lean/overhang beyond pallet boundary. Re-stack if unstable.",
                            inferred=inferred,
                        )

                # ---- 4. PREDICTIVE NEAR-MISS (USP) ----
                if state == "HELD" and held_by in persons and len(hist.buf) >= 3:
                    hspeed = hist.horiz_speed_norm()
                    carry_height_norm = max(0.0, y_floor_norm - bbox_n[3])
                    edge_prox = min(bbox_n[0], 1.0 - bbox_n[2])
                    near_edge = edge_prox <= 0.12
                    
                    if hspeed >= 0.08 and carry_height_norm >= 0.35:
                        score = min(0.98, 0.4 * (hspeed / 0.15) + 0.4 * (carry_height_norm / 0.5) + (0.2 if near_edge else 0.0))
                        self._emit(
                            date_str, "NEAR_MISS_UNSAFE_CARRY", t, t, (f_idx, f_idx),
                            [held_by], [bid],
                            {"speed_px_per_sec": round(hspeed * self.resolution[0], 1),
                             "carry_height_m": round(carry_height_norm * m_per_norm_y, 2),
                             "proximity_to_hazard": "Dock Edge" if near_edge else "Aisle"},
                            f"Carton #{bid} carried by operator #{held_by} at elevated height ({carry_height_norm * m_per_norm_y:.2f}m) and excessive speed. Live alert raised before drop occurred.",
                            "Supervisor audio alert: operator must slow down and lower carry elevation.",
                            near_miss=True, near_miss_prob=round(score, 2),
                            inferred=inferred,
                        )

                # ---- 10. SAFE HANDLING BENCHMARK ----
                if state == "HELD" and held_by in persons and len(hist.buf) >= WINDOW:
                    hspeed = hist.horiz_speed_norm()
                    carry_height_norm = max(0.0, y_floor_norm - bbox_n[3])
                    if hspeed <= 0.04 and carry_height_norm < 0.35:
                        f0 = hist.buf[0][0]
                        self._emit(
                            date_str, "SAFE_HANDLING_BENCHMARK", hist.buf[0][1], t, (f0, f_idx),
                            [held_by], [bid],
                            {"speed_px_per_sec": round(hspeed * self.resolution[0], 1),
                             "carry_height_m": round(carry_height_norm * m_per_norm_y, 2)},
                            f"Operator #{held_by} carried carton #{bid} at controlled speed and low elevation — aligns with safe handling standard.",
                            "No action required. Benchmark logged for positive reinforcement.",
                            inferred=inferred,
                        )

            # ---- 6. IMPROPER STACKING (Physical contact required) ----
            for i in range(len(resting_boxes_this_frame)):
                for j in range(len(resting_boxes_this_frame)):
                    if i == j: continue
                    top_id, top_bn = resting_boxes_this_frame[i]
                    bot_id, bot_bn = resting_boxes_this_frame[j]

                    # Top box must physically sit on top of bottom box:
                    # 1. bottom of top box touches top of bottom box within tolerance
                    vertical_gap = abs(top_bn[3] - bot_bn[1])
                    if vertical_gap > 0.08:
                        continue
                    # 2. Horizontal overlap must be significant
                    x_overlap = min(top_bn[2], bot_bn[2]) - max(top_bn[0], bot_bn[0])
                    bot_w = bot_bn[2] - bot_bn[0]
                    if bot_w <= 0 or (x_overlap / bot_w) < 0.4:
                        continue

                    top_area = bbox_area(top_bn)
                    bot_area = bbox_area(bot_bn)
                    if bot_area > 0 and (top_area / bot_area) >= 1.15:
                        self._emit(
                            date_str, "IMPROPER_STACKING", t, t, (f_idx, f_idx),
                            [], [top_id, bot_id],
                            {"top_box_area_norm": round(top_area, 4),
                             "bottom_box_area_norm": round(bot_area, 4),
                             "area_ratio": round(top_area / bot_area, 2)},
                            f"Carton #{top_id} (larger footprint) stacked directly on top of smaller carton #{bot_id} — inverted pyramid stack.",
                            "Re-stack with larger/heavier packages at the bottom to prevent crushing and tipping.",
                        )

    def _consolidate(self, gap_frames=12):
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
        for i, e in enumerate(merged, start=1):
            e["event_id"] = f"EVT-20260908-{i:04d}"
        return merged

    def _merge_run(self, run):
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
        merged["telemetry"]["is_repeated_behaviour"] = any(e["telemetry"]["is_repeated_behaviour"] for e in run)
        merged["telemetry"]["occurrence_count_in_run"] = len(run)
        return merged

    def finalize(self):
        self.events = self._consolidate()
        near_misses_prevented = sum(1 for e in self.events if e["is_near_miss"])
        high_critical = sum(1 for e in self.events if e["risk_level"] in ("High", "Critical"))
        from collections import Counter
        freq = Counter(e["behaviour_type"] for e in self.events)
        most_frequent = freq.most_common(1)[0][0] if freq else None
        return {
            "events": self.events,
            "shift_summary": {
                "total_events": len(self.events),
                "near_misses_prevented": near_misses_prevented,
                "high_critical_risks": high_critical,
                "estimated_damage_avoided_inr": near_misses_prevented * 4500,
                "estimated_damage_avoided_inr_projection": near_misses_prevented * 4500,
                "most_frequent_risk": most_frequent,
            },
        }

def run_improved_engine(path):
    with open(path) as f:
        data = json.load(f)
    meta = data.get("video_metadata", data)
    video_id = meta.get("filename") or meta.get("video_id") or os.path.basename(path)
    fps = meta.get("fps", 30.0)
    resolution = meta.get("resolution", [1920, 1080])

    engine = ImprovedRiskEngine(video_id=video_id, fps=fps, resolution=resolution)
    engine.process(data["frames"])
    return engine.finalize()

if __name__ == "__main__":
    files = sorted(glob.glob("outputs_person_a/*_tracking_results.json"))
    for f in files:
        res = run_improved_engine(f)
        evts = res["events"]
        from collections import Counter
        c = Counter(e["behaviour_code"] for e in evts)
        print(f"=== {os.path.basename(f)} === ({len(evts)} events)")
        for code, count in c.items():
            print(f"   {code}: {count}")
        if not evts:
            print("   NO EVENTS DETECTED")
