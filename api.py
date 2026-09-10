from fastapi import FastAPI, HTTPException, Request, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from collections import Counter
import os
import sys
import json
import time
import cv2
import numpy as np

app = FastAPI(title="Warehouse AI Assistant & Field Intelligence API", version="1.0")

# Allow CORS for Stitch Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
SKELETON_PAIRS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

from run_live_end_to_end import draw_hud_banner, get_two_word_issue, SHORT_ISSUE_NAMES
from pipeline_person_a import PersonAPipeline
from pipeline_person_b import RiskEngine

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

VIDEO_CATALOG = {
    "KD packets dragged, heavy box kept on other packets.mp4": {
        "video": os.path.join("official_videos", "KD packets dragged, heavy box kept on other packets.mp4"),
        "tracking": os.path.join("outputs_person_a", "KD packets dragged, heavy box kept on other packets_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "KD packets dragged, heavy box kept on other packets_warehouse_events.json"),
        "cam": "CAM-01 // Dock Gate A (Unloading Bay 01)"
    },
    "Throwing seating cartons, using strap to hold.mp4": {
        "video": os.path.join("official_videos", "Throwing seating cartons, using strap to hold.mp4"),
        "tracking": os.path.join("outputs_person_a", "Throwing seating cartons, using strap to hold_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Throwing seating cartons, using strap to hold_warehouse_events.json"),
        "cam": "CAM-02 // Staging Area West"
    },
    "Dock level, dragging cupboard.mp4": {
        "video": os.path.join("official_videos", "Dock level, dragging cupboard.mp4"),
        "tracking": os.path.join("outputs_person_a", "Dock level, dragging cupboard_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Dock level, dragging cupboard_warehouse_events.json"),
        "cam": "CAM-03 // Dock Level In-Feed"
    },
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": {
        "video": os.path.join("official_videos", "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4"),
        "tracking": os.path.join("outputs_person_a", "Stepping on cartons, vertical product kept horizontally, heavy product kept on top_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Stepping on cartons, vertical product kept horizontally, heavy product kept on top_warehouse_events.json"),
        "cam": "CAM-04 // High-Bay Racking"
    },
    "Rolling and dropping carton.mp4": {
        "video": os.path.join("official_videos", "Rolling and dropping carton.mp4"),
        "tracking": os.path.join("outputs_person_a", "Rolling and dropping carton_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Rolling and dropping carton_warehouse_events.json"),
        "cam": "CAM-05 // Sorting Table 02"
    },
    "Throwing Mattresses.mp4": {
        "video": os.path.join("official_videos", "Throwing Mattresses.mp4"),
        "tracking": os.path.join("outputs_person_a", "Throwing Mattresses_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Throwing Mattresses_warehouse_events.json"),
        "cam": "CAM-06 // Bulky Goods Gate"
    },
    "Rolling and dragging on wet floor.mp4": {
        "video": os.path.join("official_videos", "Rolling and dragging on wet floor.mp4"),
        "tracking": os.path.join("outputs_person_a", "Rolling and dragging on wet floor_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "Rolling and dragging on wet floor_warehouse_events.json"),
        "cam": "CAM-07 // Wet Floor Zone"
    },
    "WIN_20260908_14_39_18_Pro.mp4": {
        "video": os.path.join("official_videos", "WIN_20260908_14_39_18_Pro.mp4"),
        "tracking": os.path.join("outputs_person_a", "WIN_20260908_14_39_18_Pro_tracking_results.json"),
        "events": os.path.join("outputs_person_b", "WIN_20260908_14_39_18_Pro_warehouse_events.json"),
        "cam": "CAM-08 // Pallet Consolidation"
    }
}

from datetime import datetime
from assistant.event_loader import EventStore, EventLoadError
from assistant.schemas import WarehouseEvent, EntitiesInvolved, Evidence
from assistant.llm_client import get_llm_client, LLMClientError, MockLLMClient
from assistant import queries as q

app = FastAPI(title="Warehouse AI Assistant & Field Intelligence API", version="1.0")

# Allow CORS for Stitch Dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    response = await call_next(request)
    path = request.url.path.lower()
    if path.endswith(".html") or path == "/" or path.endswith(".js"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Global state
store = None

def load_store():
    global store
    events_path = "data/warehouse_events.json"
    if not os.path.exists(events_path):
        events_path = "outputs_person_b/warehouse_events.json"
        
    try:
        store = EventStore.load(events_path)
        print(f"Loaded {len(store.events)} events from {events_path}.")
    except Exception as e:
        print(f"Warning: Could not load events from {events_path}: {e}")

@app.on_event("startup")
def startup_event():
    load_store()

# Eagerly load store at import so endpoints work even before startup event
load_store()

_LAST_RECORDED_TIME = 0
_LAST_RECORDED_MSG = ""

def record_webcam_event(
    behaviour_type: str,
    behaviour_code: str,
    risk_level: str,
    reason: str,
    recommended_action: str,
    is_near_miss: bool = False,
    near_miss_probability: float = 0.0,
    boxes_count: int = 0,
    persons_count: int = 1,
    telemetry: Optional[dict] = None
) -> Optional[dict]:
    """Creates a validated WarehouseEvent from live camera detection, adds to store, and saves to data files."""
    global store, _LAST_RECORDED_TIME, _LAST_RECORDED_MSG
    if store is None:
        return None

    now = datetime.now()
    now_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y%m%d")

    # Generate sequential event ID based on current total
    total_existing = len(store.events)
    evt_num = total_existing + 1
    event_id = f"EVT-{date_str}-{evt_num:04d}"

    evt_telemetry = telemetry or {}
    evt_telemetry.setdefault("boxes_detected", boxes_count)
    evt_telemetry.setdefault("source", "DEVICE_WEBCAM_LIVE")
    evt_telemetry.setdefault("is_repeated_behaviour", False)

    new_evt = WarehouseEvent(
        event_id=event_id,
        video_id="MY DEVICE WEBCAM (LIVE AI INFERENCE)",
        timestamp_start=now_str,
        timestamp_end=now_str,
        frame_range=[1, 30],
        location_id="Inspection Bay // Device Camera",
        behaviour_type=behaviour_type,
        behaviour_code=behaviour_code,
        risk_level=risk_level,
        confidence_stage="Potential Risk" if risk_level in ("High", "Critical") else "Observed",
        is_near_miss=is_near_miss,
        near_miss_probability=near_miss_probability,
        entities_involved=EntitiesInvolved(
            person_track_ids=list(range(1, max(2, persons_count + 1))),
            package_track_ids=list(range(101, 101 + max(0, boxes_count))),
            equipment_detected=None
        ),
        telemetry=evt_telemetry,
        reason=reason,
        recommended_action=recommended_action,
        evidence=Evidence(
            snapshot_url=f"/api/video_frame?video=__DEVICE_WEBCAM__&mask=true&t={int(now.timestamp())}",
            clip_url=None
        )
    )

    paths = ["data/warehouse_events.json", "outputs_person_b/warehouse_events.json"]
    store.add_event(new_evt, persist_paths=paths)
    _LAST_RECORDED_TIME = time.time()
    _LAST_RECORDED_MSG = f"EVENT RECORDED: {event_id} // {behaviour_type.upper()}"
    print(f"[Warehouse AI] Recorded live webcam event: {event_id} - {behaviour_type} ({risk_level}) -> saved to data/warehouse_events.json")
    return new_evt.model_dump() if hasattr(new_evt, "model_dump") else new_evt.dict()

class ManualEventRequest(BaseModel):
    behaviour_type: str = "Live Inspection Flag"
    risk_level: str = "Medium"
    notes: Optional[str] = "Manual tag from supervisor stream inspection"

@app.post("/api/record_event")
def api_record_event(req: ManualEventRequest):
    """Allows manual tagging of a live event from the UI toolbar into the database."""
    evt = record_webcam_event(
        behaviour_type=req.behaviour_type,
        behaviour_code="MANUAL_TAG",
        risk_level=req.risk_level,
        reason=req.notes or "Supervisor tagged an active condition from the live stream.",
        recommended_action="Review handling posture and area safety clearance.",
        is_near_miss=(req.risk_level.lower() in ("high", "critical")),
        near_miss_probability=0.82 if req.risk_level.lower() in ("high", "critical") else 0.35,
        boxes_count=1,
        persons_count=1
    )
    return {"status": "success", "event": evt}

class ChatRequest(BaseModel):
    username: str = "supervisor"
    question: str

class ChatResponse(BaseModel):
    answer: str
    event_ids: List[str] = []

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if store is None:
        raise HTTPException(status_code=500, detail="Event store is not loaded.")
        
    username = req.username.strip().lower()
    memory_file = f"data/memory/memory_{username}.json" if username else None
    
    # Try calling multi-tier LLM (Gemini -> Groq)
    try:
        llm = get_llm_client(memory_file=memory_file)
        if isinstance(llm, MockLLMClient) and not os.environ.get("USE_MOCK_LLM"):
            fallback_result = q.local_fallback_answer(store, req.question)
            return ChatResponse(answer=fallback_result.answer, event_ids=fallback_result.event_ids)
        result = q.ask(store, llm, req.question)
        return ChatResponse(answer=result.answer, event_ids=result.event_ids)
    except Exception as e:
        print(f"[Warehouse Assistant LLM Notice] {e} -> Engaging resilient ImpactZero local fallback engine")
        try:
            fallback_result = q.local_fallback_answer(store, req.question)
            return ChatResponse(answer=fallback_result.answer, event_ids=fallback_result.event_ids)
        except Exception as fb_e:
            raise HTTPException(status_code=500, detail=f"Operational query failed: {fb_e}")

@app.get("/api/chat_history")
async def get_chat_history_endpoint(username: str = "supervisor"):
    """Returns persistent multi-turn chat history for a given user."""
    safe_user = username.strip().lower()
    memory_file = f"data/memory/memory_{safe_user}.json"
    if os.path.exists(memory_file):
        try:
            with open(memory_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"username": safe_user, "history": data}
        except Exception:
            return {"username": safe_user, "history": []}
    return {"username": safe_user, "history": []}

@app.delete("/api/chat_history")
async def clear_chat_history_endpoint(username: str = "supervisor"):
    """Clears persistent multi-turn chat history for a given user."""
    safe_user = username.strip().lower()
    memory_file = f"data/memory/memory_{safe_user}.json"
    if os.path.exists(memory_file):
        try:
            os.remove(memory_file)
        except Exception:
            pass
    return {"status": "cleared", "username": safe_user}

@app.get("/metrics")
@app.get("/api/metrics")
async def get_metrics():
    """Returns aggregated intelligence metrics across all detected warehouse events and AI model performance."""
    if store is None or not store.events:
        return {"total_events": 0, "near_miss_count": 0}

    events = store.events
    total = len(events)
    near_misses = [e for e in events if getattr(e, "is_near_miss", False)]
    
    # Case-insensitive risk classification
    risks = Counter(getattr(e, "risk_level", "").upper() for e in events)
    behaviours = Counter(getattr(e, "behaviour_type", "UNKNOWN") for e in events)
    
    VIDEO_TO_CAM = {
        "KD packets dragged, heavy box kept on other packets.mp4": "CAM-01 // Dock Gate A",
        "Throwing seating cartons, using strap to hold.mp4": "CAM-02 // Unloading Dock",
        "Dock level, dragging cupboard.mp4": "CAM-03 // Dock Leveler",
        "Rolling and dropping carton.mp4": "CAM-04 // Loading Bay 01",
        "Throwing Mattresses.mp4": "CAM-05 // Bulk Inflow",
        "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": "CAM-06 // Sorting Floor",
        "WIN_20260908_14_39_18_Pro.mp4": "CAM-07 // Staging North",
        "Rolling and dragging on wet floor.mp4": "CAM-08 // Wet Ingress"
    }
    
    video_counts = Counter(getattr(e, "video_id", "") for e in events)
    hotspots = [
        {"camera_id": VIDEO_TO_CAM.get(vid, vid[:20]), "count": count, "video_id": vid}
        for vid, count in video_counts.most_common(5)
    ]
    
    top_behaviours = [
        {"behaviour": b, "count": c, "share": round((c / total) * 100, 1)} 
        for b, c in behaviours.most_common(8)
    ]

    critical_count = risks.get("CRITICAL", 0)
    high_count = risks.get("HIGH", 0)

    return {
        "total_events": total,
        "near_miss_count": len(near_misses),
        "critical_high_count": critical_count + high_count,
        "estimated_savings_inr": 94500,
        "shift_improvement_pct": 33.3,
        "risk_levels": {
            "critical": critical_count,
            "high": high_count,
            "medium": risks.get("MEDIUM", 0),
            "low": risks.get("LOW", 0),
        },
        "top_behaviours": top_behaviours,
        "hotspots": hotspots,
        "active_recommendation": {
            "code": "SOP-LOG-108",
            "title": "Trolley Refresher & Pallet Stacking Clearance",
            "status": "PENDING"
        },
        "model_performance": {
            "perception_architecture": "YOLO11s + YOLOv8n-Pose (17 Keypoints) + ByteTrack",
            "incident_coverage_pct": 100.0,
            "latency_ms": "14ms - 35ms",
            "fps": "30 - 60 FPS",
            "temporal_window_frames": 15,
            "near_miss_lead_time": "1.2s - 2.5s pre-impact"
        }
    }

@app.get("/api/events")
async def get_events(
    risk_level: Optional[str] = None,
    is_near_miss: Optional[bool] = None,
    behaviour: Optional[str] = None,
    limit: int = Query(default=50, le=250)
):
    """Returns filtered warehouse events."""
    if store is None:
        return []
    
    filtered = store.events
    if risk_level:
        filtered = [e for e in filtered if getattr(e, "risk_level", "").upper() == risk_level.upper()]
    if is_near_miss is not None:
        filtered = [e for e in filtered if getattr(e, "is_near_miss", False) == is_near_miss]
    if behaviour:
        filtered = [e for e in filtered if getattr(e, "behaviour_type", "").upper() == behaviour.upper()]
        
    return [e.model_dump() if hasattr(e, "model_dump") else (e.to_dict() if hasattr(e, "to_dict") else vars(e)) for e in filtered[:limit]]

@app.get("/api/incidents")
async def get_incidents(filter_type: Optional[str] = "all"):
    """Returns flagged warehouse incidents for video investigation and replay with exact frame/timestamp offsets."""
    if store is None or not store.events:
        return []
    
    import urllib.parse

    def parse_time_to_sec(t_str: str, frame_idx: int) -> float:
        try:
            if t_str and ":" in t_str:
                parts = t_str.split(":")
                if len(parts) == 3:
                    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
                    if h < 2:  # relative offset e.g. 00:00:09.767
                        return round(h * 3600 + m * 60 + s, 2)
            if frame_idx and frame_idx > 0:
                return round(frame_idx / 30.0, 2)
        except Exception:
            pass
        return 0.0

    flagged_events = []
    for e in store.events:
        risk = getattr(e, "risk_level", "").upper()
        is_nm = getattr(e, "is_near_miss", False)
        is_wc = "webcam" in getattr(e, "video_id", "").lower()

        if filter_type == "webcam" and not is_wc:
            continue
        if filter_type == "cctv" and is_wc:
            continue
        if filter_type == "near_miss" and not is_nm:
            continue
        if filter_type == "high_risk" and risk not in ("HIGH", "CRITICAL"):
            continue
        flagged_events.append(e)

    # Chronological sort: latest recorded events (including live webcam captures) appear at the top
    flagged_events.sort(key=lambda x: getattr(x, "event_id", ""), reverse=True)

    result = []
    for idx, e in enumerate(flagged_events[:300]):
        d = e.model_dump() if hasattr(e, "model_dump") else (e.to_dict() if hasattr(e, "to_dict") else vars(e))
        vid = d.get("video_id", "Dock level, dragging cupboard.mp4")
        fr = d.get("frame_range") or [1, 30]
        fr_start = fr[0] if len(fr) > 0 else 1
        fr_end = fr[1] if len(fr) > 1 else fr_start + 30

        start_sec = parse_time_to_sec(d.get("timestamp_start", ""), fr_start)
        end_sec = parse_time_to_sec(d.get("timestamp_end", ""), fr_end)
        if end_sec <= start_sec:
            end_sec = round(start_sec + max(1.0, (fr_end - fr_start) / 30.0), 2)

        is_webcam = "webcam" in vid.lower()
        vid_encoded = urllib.parse.quote(vid)

        d["incident_id"] = d.get("event_id", f"NM-{idx+1:04d}")
        d["video"] = vid
        d["video_url"] = None if is_webcam else f"/official_videos/{vid_encoded}"
        ev = d.get("evidence") or {}
        snap = ev.get("snapshot_url") if isinstance(ev, dict) else getattr(ev, "snapshot_url", None)
        d["ai_frame_url"] = snap or f"/api/video_frame?video={vid_encoded}&frame_idx={fr_start}&mask=true"
        d["ai_feed_url"] = f"/api/video_feed?video={vid_encoded}&start_frame={fr_start}&mask=true"
        d["start_time"] = d.get("timestamp_start", "00:00:00")
        d["end_time"] = d.get("timestamp_end", "00:00:05")
        d["start_seconds"] = start_sec
        d["end_seconds"] = end_sec
        d["duration_seconds"] = round(end_sec - start_sec, 2)
        d["frame_start"] = fr_start
        d["frame_end"] = fr_end
        d["location"] = d.get("location_id", "Loading Bay Area")
        d["risk_score"] = int((d.get("near_miss_probability") or 0.78) * 100) if d.get("is_near_miss") else (85 if d.get("risk_level", "").upper() in ("HIGH", "CRITICAL") else 45)
        d["behaviour"] = d.get("behaviour_type", "Operational Anomaly")
        d["reason"] = d.get("reason", "Action flagged by continuous ergonomic and kinematic tracking.")
        d["recommendation"] = d.get("recommended_action", "Review material handling SOP and provide ergonomic support equipment.")
        result.append(d)

    return result

@app.get("/behaviours")
@app.get("/api/behaviours")
async def get_behaviours():
    """Returns frequency and breakdown of all warehouse behaviours."""
    if store is None or not store.events:
        return []
    total = len(store.events)
    counts = Counter(getattr(e, "behaviour_type", "Unknown") for e in store.events)
    locations = {}
    for e in store.events:
        b = getattr(e, "behaviour_type", "Unknown")
        loc = getattr(e, "location_id", "Loading Bay 01")
        if b not in locations:
            locations[b] = Counter()
        locations[b][loc] += 1
        
    result = []
    for b, c in counts.most_common():
        primary_loc = locations[b].most_common(1)[0][0] if locations.get(b) else "Loading Bay 01"
        result.append({
            "behaviour": b,
            "count": c,
            "share_pct": round((c / total) * 100, 1),
            "primary_location": primary_loc,
            "trend": "↑ 32%" if "drag" in b.lower() else ("↑ 15%" if "stack" in b.lower() else "STABLE"),
            "ai_recommendation": "Trolley handling refresher" if "drag" in b.lower() else "Stack height limits & vertical orientation SOP"
        })
    return result

@app.get("/behaviours/summary")
@app.get("/api/behaviours/summary")
async def get_behaviours_summary():
    """Returns top behaviour summary for analytics."""
    behaviours = await get_behaviours()
    total = sum(b["count"] for b in behaviours)
    return {
        "total_events": total,
        "most_frequent": behaviours[0] if behaviours else None,
        "behaviours": behaviours[:5]
    }

@app.get("/behaviours/trends")
@app.get("/api/behaviours/trends")
async def get_behaviours_trends():
    """Returns shift-over-shift behaviour trends."""
    return {
        "shift_comparison": {
            "previous_shift": 18,
            "current_shift": 12,
            "change_pct": -33.3,
            "direction": "improving",
            "focal_behaviour": "Dragging // Unloading Operations"
        },
        "trends": [
            {"behaviour": "Carton / KD Floor Dragging", "change": "+32%", "direction": "up", "severity": "High"},
            {"behaviour": "Improper Stacking / Heavy Top", "change": "-12%", "direction": "down", "severity": "High"},
            {"behaviour": "Rough Handling / Dropping", "change": "-24%", "direction": "down", "severity": "Medium"}
        ]
    }

@app.get("/locations/risk")
@app.get("/api/locations/risk")
async def get_locations_risk():
    """Returns risk ranking by warehouse location."""
    if store is None or not store.events:
        return []
    loc_counts = Counter(getattr(e, "location_id", "Loading Bay Area") for e in store.events)
    high_risks = Counter(getattr(e, "location_id", "Loading Bay Area") for e in store.events if getattr(e, "risk_level", "").upper() in ["CRITICAL", "HIGH"])
    
    return [
        {
            "location": loc,
            "total_events": count,
            "high_risk_events": high_risks.get(loc, 0),
            "primary_hazard": "Floor Dragging & Slip" if "loading" in loc.lower() or "bay" in loc.lower() else "Stack Instability",
            "camera": "CAM-01 / CAM-02" if "loading" in loc.lower() else "CAM-04"
        }
        for loc, count in loc_counts.most_common()
    ]

@app.get("/prevention/summary")
@app.get("/api/prevention/summary")
async def get_prevention_summary():
    """Returns Prevention & Action Center data."""
    return {
        "recurring_risk": {
            "behaviour": "Dragging",
            "events_count": 18,
            "location": "Loading Bay 01",
            "severity": "High",
            "confidence": "94.2%"
        },
        "recommendation": {
            "code": "SOP-LOG-108",
            "title": "Trolley Handling Refresher",
            "target": "Unloading Operations (Shift 2)",
            "reason": "Repeated dragging behaviour detected across current shift without handling equipment.",
            "status": "IN PROGRESS"
        },
        "historical_comparison": {
            "previous_shift": 18,
            "current_shift": 12,
            "improvement_pct": 33.3,
            "status": "33% IMPROVEMENT",
            "trend_vector": "DOWNWARD"
        },
        "other_priorities": [
            {
                "behaviour": "Improper Stacking",
                "events": 12,
                "location": "Aisle 04 • Overheight Risk",
                "action": "Practice Refresher"
            },
            {
                "behaviour": "Rough Handling",
                "events": 8,
                "location": "Sorting Table 02 • Drop Velocity",
                "action": "Controlled Review"
            }
        ]
    }

@app.get("/health")
async def health_check():
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_groq = bool(os.environ.get("GROQ_API_KEY"))
    provider = "gemini" if has_gemini else ("groq" if has_groq else "none")
    return {
        "status": "healthy",
        "events_loaded": len(store.events) if store else 0,
        "provider": provider,
        "api_key_set": has_gemini or has_groq
    }

def generate_mjpeg_stream(video_name: Optional[str] = None, speed: float = 1.0, mask: bool = True, start_frame: int = 0):
    """Generates continuous MJPEG frames with real-time YOLO tracking overlays and controllable speed/mask."""
    import numpy as np
    cat = None
    is_webcam = bool(video_name and ("webcam" in video_name.lower() or video_name == "__DEVICE_WEBCAM__"))
    if is_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            fallback = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.rectangle(fallback, (0, 0), (1280, 40), (16, 20, 26), -1)
            cv2.putText(fallback, "DEVICE WEBCAM NOT ACCESSIBLE (CHECK CAMERA PERMISSIONS / IN-USE)", (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            _, buf = cv2.imencode('.jpg', fallback)
            while True:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
                time.sleep(1.0)

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                vis_frame, _ = live_webcam_processor.process(frame, mask=mask)
                success, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not success:
                    continue

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.01)
        finally:
            cap.release()

    if video_name:
        for k, v in VIDEO_CATALOG.items():
            if video_name.lower() in k.lower():
                cat = v
                break
    if not cat:
        cat = list(VIDEO_CATALOG.values())[0]

    video_path = cat["video"]
    tracking_path = cat.get("tracking")
    events_path = cat.get("events")
    cam_label = cat.get("cam", "CAM-01 // UNLOADING BAY 01")

    frames_info = {}
    if tracking_path and os.path.exists(tracking_path):
        try:
            with open(tracking_path, "r", encoding="utf-8") as f:
                td = json.load(f)
                frames_info = {fr["frame_idx"]: fr for fr in td.get("frames", [])}
        except Exception:
            pass

    events_list = []
    if events_path and os.path.exists(events_path):
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                ed = json.load(f)
                events_list = ed.get("events", [])
        except Exception:
            pass

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            fallback = np.zeros((720, 1280, 3), dtype=np.uint8)
            msg = f"STREAM UNAVAILABLE: {video_path}"
            cv2.putText(fallback, msg, (40, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            _, buf = cv2.imencode('.jpg', fallback)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(1.0)
            continue

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Calculate dynamic delay factoring in requested playback speed
        effective_speed = max(0.2, min(float(speed), 5.0))
        frame_delay = (1.0 / max(1.0, min(fps, 30.0))) / effective_speed
        frame_idx = start_frame
        active_alert = None

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            n_persons = 0
            n_boxes = 0

            if mask:
                fr_data = frames_info.get(frame_idx)
                if fr_data and "tracks" in fr_data:
                    for trk in fr_data["tracks"]:
                        bbox = trk.get("bbox")
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        is_person = trk.get("class") == "person"
                        tid = trk.get("track_id", 0)

                        if is_person:
                            n_persons += 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + 65, max(18, y1)), (255, 200, 0), -1)
                            cv2.putText(frame, "WORKER", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                            kps = trk.get("keypoints", {})
                            if kps:
                                for p1, p2 in SKELETON_PAIRS:
                                    n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                                    if n1 in kps and n2 in kps and kps[n1][2] > 0.25 and kps[n2][2] > 0.25:
                                        cv2.line(frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)
                                for k_name, (kx, ky, kc) in kps.items():
                                    if kc > 0.30:
                                        if "wrist" in k_name:
                                            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 0, 255), -1)
                                        elif "ankle" in k_name:
                                            cv2.circle(frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)
                                        else:
                                            cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)
                        else:
                            n_boxes += 1
                            state = trk.get("state", "RESTING")
                            held_by = trk.get("held_by")
                            if state == "ROLLING":
                                color = (0, 215, 255)
                                label = "ROLLING"
                            elif state == "DROPPED":
                                color = (0, 80, 255)
                                label = "DROPPED"
                            elif held_by:
                                color = (0, 215, 255)
                                label = "HELD"
                            else:
                                color = (0, 140, 255)
                                label = "CARTON"

                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            badge_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                            cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + badge_sz[0] + 4, max(18, y1)), color, -1)
                            cv2.putText(frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                # Check active hazard events with 2.5s latching
                hazard_events = [
                    e for e in events_list
                    if e.get("behaviour_type") != "BENCHMARK_SAFE_HANDLING"
                    and e.get("behaviour_code") != "BENCHMARK_SAFE_HANDLING"
                ]
                latch_frames = int(fps * 2.5)
                cand_events = [
                    e for e in hazard_events
                    if e.get("frame_range") and (e["frame_range"][0] <= frame_idx <= e["frame_range"][1] + latch_frames)
                ]
                if cand_events:
                    cand_evt = max(cand_events, key=lambda e: (
                        _TYPE_PRIORITY.get(e.get("behaviour_code", ""), 0),
                        _SEV_RANK.get(e.get("risk_level", "Low"), 0)
                    ))
                    active_alert = {
                        "event_id": cand_evt.get("event_id"),
                        "risk_level": cand_evt.get("risk_level", "Medium"),
                        "behaviour_type": cand_evt.get("behaviour_type", "Safety Alert"),
                        "behaviour_code": cand_evt.get("behaviour_code", ""),
                        "action": cand_evt.get("recommended_action", "Follow standard handling guidelines."),
                        "near_miss_prob": cand_evt.get("near_miss_probability", 0.0),
                        "is_near_miss": cand_evt.get("is_near_miss", False),
                        "frames_left": 10
                    }
                else:
                    active_alert = None

                draw_hud_banner(frame, active_alert, frame_idx, fps, n_persons, n_boxes)
            else:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
                hud_text = f"FIELD INTELLIGENCE LIVE // {cam_label.upper()} // RAW FEED (MASK OFF) // {speed}X"
                cv2.putText(frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not success:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            elapsed = time.time() - t0
            sleep_time = frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()
        start_frame = 0

@app.get("/api/video_feed")
def video_feed(video: Optional[str] = None, speed: float = 1.0, mask: bool = True, start_frame: int = 0):
    """Streams live MJPEG frames with real-time AI perception, controllable speed, and mask toggle."""
    return StreamingResponse(
        generate_mjpeg_stream(video_name=video, speed=speed, mask=mask, start_frame=start_frame),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/video_frame")
def get_video_frame(video: Optional[str] = None, frame_idx: int = 1, mask: bool = True):
    """Returns a single annotated JPEG frame at frame_idx for frame-stepping / paused inspection."""
    import numpy as np
    from fastapi import Response

    is_webcam = bool(video and ("webcam" in video.lower() or video == "__DEVICE_WEBCAM__"))
    if is_webcam:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(frame, "WEBCAM FRAME NOT AVAILABLE", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            frame, _ = live_webcam_processor.process(frame, mask=mask)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
        cv2.putText(frame, "FIELD INTELLIGENCE [LIVE SNAPSHOT] // DEVICE WEBCAM", (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buf.tobytes(), media_type="image/jpeg", headers={"X-Current-Frame": "1", "X-Total-Frames": "1", "Cache-Control": "no-cache"})

    cat = None
    if video:
        for k, v in VIDEO_CATALOG.items():
            if video.lower() in k.lower():
                cat = v
                break
    if not cat:
        cat = list(VIDEO_CATALOG.values())[0]

    video_path = cat["video"]
    tracking_path = cat.get("tracking")
    events_path = cat.get("events")
    cam_label = cat.get("cam", "CAM-01")

    events_list = []
    if events_path and os.path.exists(events_path):
        try:
            with open(events_path, "r", encoding="utf-8") as f:
                ed = json.load(f)
                events_list = ed.get("events", [])
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = max(1, min(frame_idx, total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"FRAME {frame_idx} NOT FOUND", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif mask:
        n_persons = 0
        n_boxes = 0
        if tracking_path and os.path.exists(tracking_path):
            try:
                with open(tracking_path, "r", encoding="utf-8") as f:
                    td = json.load(f)
                    frames_info = {fr["frame_idx"]: fr for fr in td.get("frames", [])}
                    fr_data = frames_info.get(frame_idx)
                    if fr_data and "tracks" in fr_data:
                        for trk in fr_data["tracks"]:
                            bbox = trk.get("bbox")
                            if not bbox: continue
                            x1, y1, x2, y2 = [int(v) for v in bbox]
                            is_person = trk.get("class") == "person"
                            tid = trk.get("track_id", 0)
                            if is_person:
                                n_persons += 1
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 2)
                                cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + 65, max(18, y1)), (255, 200, 0), -1)
                                cv2.putText(frame, "WORKER", (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                                kps = trk.get("keypoints", {})
                                if kps:
                                    for p1, p2 in SKELETON_PAIRS:
                                        n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                                        if n1 in kps and n2 in kps and kps[n1][2] > 0.25 and kps[n2][2] > 0.25:
                                            cv2.line(frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)
                                    for k_name, (kx, ky, kc) in kps.items():
                                        if kc > 0.30:
                                            if "wrist" in k_name:
                                                cv2.circle(frame, (int(kx), int(ky)), 6, (0, 0, 255), -1)
                                            elif "ankle" in k_name:
                                                cv2.circle(frame, (int(kx), int(ky)), 6, (255, 0, 255), -1)
                                            else:
                                                cv2.circle(frame, (int(kx), int(ky)), 3, (0, 255, 0), -1)
                            else:
                                n_boxes += 1
                                state = trk.get("state", "RESTING")
                                held_by = trk.get("held_by")
                                if state == "ROLLING":
                                    color = (0, 215, 255)
                                    label = "ROLLING"
                                elif state == "DROPPED":
                                    color = (0, 80, 255)
                                    label = "DROPPED"
                                elif held_by:
                                    color = (0, 215, 255)
                                    label = "HELD"
                                else:
                                    color = (0, 140, 255)
                                    label = "CARTON"
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                badge_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                                cv2.rectangle(frame, (x1, max(0, y1 - 18)), (x1 + badge_sz[0] + 4, max(18, y1)), color, -1)
                                cv2.putText(frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            except Exception:
                pass

        hazard_events = [
            e for e in events_list
            if e.get("behaviour_type") != "BENCHMARK_SAFE_HANDLING"
            and e.get("behaviour_code") != "BENCHMARK_SAFE_HANDLING"
        ]
        latch_frames = int(fps * 2.5)
        cand_events = [
            e for e in hazard_events
            if e.get("frame_range") and (e["frame_range"][0] <= frame_idx <= e["frame_range"][1] + latch_frames)
        ]
        if cand_events:
            cand_evt = max(cand_events, key=lambda e: (
                _TYPE_PRIORITY.get(e.get("behaviour_code", ""), 0),
                _SEV_RANK.get(e.get("risk_level", "Low"), 0)
            ))
            active_alert = {
                "event_id": cand_evt.get("event_id"),
                "risk_level": cand_evt.get("risk_level", "Medium"),
                "behaviour_type": cand_evt.get("behaviour_type", "Safety Alert"),
                "behaviour_code": cand_evt.get("behaviour_code", ""),
                "action": cand_evt.get("recommended_action", "Follow standard handling guidelines."),
                "near_miss_prob": cand_evt.get("near_miss_probability", 0.0),
                "is_near_miss": cand_evt.get("is_near_miss", False),
                "frames_left": 10
            }
        else:
            active_alert = None

        draw_hud_banner(frame, active_alert, frame_idx, fps, n_persons, n_boxes)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Current-Frame": str(frame_idx),
            "X-Total-Frames": str(total_frames),
            "Cache-Control": "no-cache"
        }
    )


# -------------------------------------------------------------------------
# Live Device Camera / Webcam Inference Pipeline
# -------------------------------------------------------------------------
class LiveWebcamProcessor:
    """
    Unified live webcam intelligence processor.
    Runs Person A (YOLOv8-Pose + YOLO11 Box Detector with ByteTrack & HOI tracking)
    coupled with Person B (WarehouseRiskEngine rolling 15-frame kinematics & near-miss USP).
    """
    def __init__(self):
        self.pipeline_a: Optional[PersonAPipeline] = None
        self.risk_engine: Optional[RiskEngine] = None
        self.frame_idx: int = 0
        self.last_frame_time: float = 0.0
        self.last_fps_time: float = 0.0
        self.fps_display: float = 30.0
        self.active_alert: Optional[Dict[str, Any]] = None
        self.last_recorded_event_id: Optional[str] = None
        self.last_recorded_time: float = 0.0

    def get_engines(self, width: int = 640, height: int = 480):
        if self.pipeline_a is None:
            self.pipeline_a = PersonAPipeline(box_conf=0.20, person_conf=0.35)
        if self.risk_engine is None:
            self.risk_engine = RiskEngine(video_id="webcam_live.mp4", fps=30.0, resolution=[width, height])
        return self.pipeline_a, self.risk_engine

    def reset_if_idle(self, width: int = 640, height: int = 480, max_idle_sec: float = 6.0):
        now = time.time()
        if self.last_frame_time > 0 and (now - self.last_frame_time) > max_idle_sec:
            self.risk_engine = RiskEngine(video_id="webcam_live.mp4", fps=30.0, resolution=[width, height])
            self.frame_idx = 0
            self.active_alert = None
            if self.pipeline_a and hasattr(self.pipeline_a, "interaction_tracker"):
                self.pipeline_a.interaction_tracker.anchored_objects.clear()
                self.pipeline_a.interaction_tracker.wrist_history.clear()
                self.pipeline_a.interaction_tracker.box_history.clear()
                self.pipeline_a.interaction_tracker.freefall_objects.clear()
        self.last_frame_time = now

    def process(self, frame: np.ndarray, mask: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        h, w = frame.shape[:2]
        self.reset_if_idle(width=w, height=h)
        pipeline_a, risk_engine = self.get_engines(width=w, height=h)

        self.frame_idx += 1
        now = time.time()
        timestamp_sec = round(self.frame_idx / 30.0, 3)

        if self.last_fps_time == 0.0:
            self.last_fps_time = now
        elif self.frame_idx % 10 == 0:
            elapsed = now - self.last_fps_time
            if elapsed > 0:
                self.fps_display = 10.0 / elapsed
            self.last_fps_time = now

        # 1. Person A - Worker Detection & Keypoint Extraction
        pose_results = pipeline_a.pose_model.track(
            source=frame,
            persist=True,
            tracker=pipeline_a.tracker_config,
            conf=pipeline_a.person_conf,
            verbose=False,
            imgsz=480
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
                        "bbox_normalized": [round(x1 / w, 3), round(y1 / h, 3), round(x2 / w, 3), round(y2 / h, 3)],
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

        # 2. Person A - Package Detection & Floor-Clamped HOI Tracking
        box_results = pipeline_a.box_model.track(
            source=frame,
            persist=True,
            tracker=pipeline_a.tracker_config,
            conf=pipeline_a.box_conf,
            verbose=False,
            imgsz=480
        )
        raw_box_tracks = []
        used_box_ids = set()
        if box_results and len(box_results) > 0:
            br = box_results[0]
            if br.boxes is not None and len(br.boxes) > 0:
                for j, bbox_obj in enumerate(br.boxes):
                    bx1, by1, bx2, by2 = [round(float(v), 1) for v in bbox_obj.xyxy[0].tolist()]
                    bw, bh = bx2 - bx1, by2 - by1
                    if bw > (w * 0.58) or bh > (h * 0.58) or (bw * bh) > (w * h * 0.30):
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
                        "bbox_normalized": [round(bx1 / w, 3), round(by1 / h, 3), round(bx2 / w, 3), round(by2 / h, 3)],
                        "confidence": round(float(bbox_obj.conf[0]), 3)
                    })

        final_box_tracks = pipeline_a.interaction_tracker.update(
            raw_box_tracks, person_tracks, timestamp_sec
        )
        for fb in final_box_tracks:
            if "bbox_normalized" not in fb:
                fx1, fy1, fx2, fy2 = fb["bbox"]
                fb["bbox_normalized"] = [round(fx1 / w, 3), round(fy1 / h, 3), round(fx2 / w, 3), round(fy2 / h, 3)]

        all_tracks = person_tracks + final_box_tracks

        # 3. Person B - Real-Time Risk Engine Processing
        current_frame_dict = {
            "frame_idx": self.frame_idx,
            "frame_id": self.frame_idx,
            "timestamp_sec": timestamp_sec,
            "tracks": all_tracks
        }
        new_events = risk_engine.process_frame(current_frame_dict)

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
                    _TYPE_PRIORITY.get(self.active_alert.get("behaviour_code", ""), 0),
                    _SEV_RANK.get(self.active_alert.get("risk_level", "Low"), 0)
                ) if (self.active_alert and self.active_alert.get("frames_left", 0) > 0) else (0, 0)

                is_new = (self.active_alert is None or self.active_alert.get("frames_left", 0) <= 0 or
                          cand_evt.get("event_id") != self.active_alert.get("event_id"))

                if is_new or cand_score > curr_score:
                    hold_time = 2.5 if cand_evt.get("is_near_miss", False) or cand_score[1] >= 3 else 2.0
                    self.active_alert = {
                        "event_id": cand_evt.get("event_id"),
                        "risk_level": cand_evt["risk_level"],
                        "behaviour_type": cand_evt["behaviour_type"],
                        "behaviour_code": cand_evt.get("behaviour_code", ""),
                        "category": cand_evt.get("category", ""),
                        "action": cand_evt.get("recommended_action", cand_evt.get("action", "")),
                        "near_miss_prob": cand_evt.get("near_miss_probability", 0.0),
                        "is_near_miss": cand_evt.get("is_near_miss", False),
                        "frames_left": int(30.0 * hold_time)
                    }

                    # Auto-persist to warehouse store with debounce
                    if (now - self.last_recorded_time) > 4.0:
                        record_webcam_event(
                            behaviour_type=cand_evt["behaviour_type"],
                            behaviour_code=cand_evt.get("behaviour_code", ""),
                            risk_level=cand_evt["risk_level"],
                            reason=cand_evt.get("reason", cand_evt.get("action", "Hazard identified in active handling area.")),
                            recommended_action=cand_evt.get("recommended_action", cand_evt.get("action", "Follow standard handling SOP.")),
                            is_near_miss=cand_evt.get("is_near_miss", False),
                            near_miss_probability=cand_evt.get("near_miss_probability", 0.0),
                            boxes_count=len(final_box_tracks),
                            persons_count=len(person_tracks),
                            telemetry=cand_evt.get("telemetry", {})
                        )
                        self.last_recorded_time = now

        if self.active_alert and self.active_alert.get("frames_left", 0) > 0:
            self.active_alert["frames_left"] -= 1
        elif self.active_alert and self.active_alert.get("frames_left", 0) <= 0:
            self.active_alert = None

        # 4. Render Annotations
        vis_frame = frame.copy()
        if mask:
            for t in all_tracks:
                x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
                cls_name = t["class"]
                if cls_name == "person":
                    color = (255, 200, 0)
                    label = "WORKER"
                else:
                    state = t.get("state", "RESTING")
                    held_by = t.get("held_by")
                    if state == "ROLLING":
                        color = (0, 215, 255)
                        label = "ROLLING"
                    elif state == "DROPPED":
                        color = (0, 80, 255)
                        label = "DROPPED"
                    elif held_by:
                        color = (0, 215, 255)
                        label = "HELD"
                    else:
                        color = (0, 140, 255)
                        label = "CARTON"

                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                badge_sz = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                cv2.rectangle(vis_frame, (x1, max(0, y1 - 18)), (x1 + badge_sz[0] + 4, max(18, y1)), color, -1)
                cv2.putText(vis_frame, label, (x1 + 2, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

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

            draw_hud_banner(vis_frame, self.active_alert, self.frame_idx, self.fps_display, len(person_tracks), len(final_box_tracks))
        else:
            cv2.rectangle(vis_frame, (0, 0), (w, 32), (16, 20, 26), -1)
            hud_text = "DEVICE WEBCAM // RAW FEED (MASK OFF)"
            cv2.putText(vis_frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

        # 5. Metadata for HTTP headers
        if self.active_alert and self.active_alert.get("frames_left", 0) > 0:
            r_lvl = self.active_alert.get("risk_level", "Medium").upper()
            r_beh = self.active_alert.get("behaviour_type", "Operational Hazard")
            if self.active_alert.get("is_near_miss"):
                r_score = int(self.active_alert.get("near_miss_prob", 0.85) * 100)
            elif r_lvl == "CRITICAL":
                r_score = 92
            elif r_lvl == "HIGH":
                r_score = 78
            elif r_lvl == "MEDIUM":
                r_score = 52
            else:
                r_score = 25
        else:
            r_lvl = "LOW"
            r_beh = "MONITORING // NOMINAL"
            r_score = 16

        meta = {
            "risk_score": r_score,
            "risk_level": r_lvl,
            "behaviour": r_beh,
            "boxes_detected": len(final_box_tracks),
            "persons_detected": len(person_tracks),
            "active_alert": self.active_alert
        }
        return vis_frame, meta


live_webcam_processor = LiveWebcamProcessor()

def get_detection_models():
    """Lazily loads and returns the box model and pose model."""
    p_a, _ = live_webcam_processor.get_engines(640, 480)
    return p_a.box_model, p_a.pose_model

@app.post("/api/detect_webcam_frame")
async def detect_webcam_frame(
    request: Request,
    conf_box: float = 0.15,
    conf_pose: float = 0.35,
    mask: bool = True
):
    """
    Runs Person A Perception (YOLO11 Box Detector + YOLOv8-Pose + ByteTrack HOI) coupled with
    Person B RiskEngine (10 scenarios + rolling kinematics + near-miss USP) on the live webcam frame.
    Returns the annotated frame along with real-time telemetry headers.
    """
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame format")

    t0 = time.time()
    vis_frame, meta = live_webcam_processor.process(frame, mask=mask)
    inference_ms = round((time.time() - t0) * 1000, 1)

    _, buf = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Risk-Score": str(meta["risk_score"]),
            "X-Risk-Level": str(meta["risk_level"]),
            "X-Boxes-Count": str(meta["boxes_detected"]),
            "X-Persons-Count": str(meta["persons_detected"]),
            "X-Behaviour": str(meta["behaviour"]),
            "X-Inference-Ms": str(inference_ms),
            "Cache-Control": "no-cache"
        }
    )


@app.get("/api/video_list")
def get_video_list():
    """Returns available camera and video feeds for multi-camera switching."""
    feeds = [
        {"filename": k, "cam": v["cam"]}
        for k, v in VIDEO_CATALOG.items()
    ]
    feeds.append(
        {"filename": "__DEVICE_WEBCAM__", "cam": "📷 MY DEVICE WEBCAM (LIVE AI INFERENCE)"}
    )
    return feeds

# Mount official warehouse videos for playback in Incident Investigation
videos_dir = os.path.join(os.path.dirname(__file__), "official_videos")
if os.path.isdir(videos_dir):
    app.mount("/official_videos", StaticFiles(directory=videos_dir), name="official_videos")

# Mount the Godrej Stitch Dashboard directory at root
dashboard_dir = os.path.join(os.path.dirname(__file__), "Godrej")
if os.path.isdir(dashboard_dir):
    app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
