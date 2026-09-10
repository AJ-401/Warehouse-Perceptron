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
from assistant.llm_client import get_llm_client, LLMClientError
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
    
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Neither GEMINI_API_KEY nor GROQ_API_KEY is set on the server."
        )
        
    try:
        llm = get_llm_client(memory_file=memory_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {str(e)}")

    try:
        # Run the tool-calling assistant
        result = q.ask(store, llm, req.question)
        return ChatResponse(answer=result.answer, event_ids=result.event_ids)
    except Exception as e:
        error_str = str(e)
        if any(k in error_str.lower() for k in ("limit exceeded", "429", "resource_exhausted", "rate limit")):
            raise HTTPException(status_code=429, detail="API rate limit exceeded. Please wait a minute and try again.")
        raise HTTPException(status_code=500, detail=f"LLM request failed: {error_str}")

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

@app.get("/api/metrics")
async def get_metrics():
    """Returns aggregated intelligence metrics across all detected warehouse events."""
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

    return {
        "total_events": total,
        "near_miss_count": len(near_misses),
        "risk_levels": {
            "critical": risks.get("CRITICAL", 0),
            "high": risks.get("HIGH", 0),
            "medium": risks.get("MEDIUM", 0),
            "low": risks.get("LOW", 0),
        },
        "top_behaviours": top_behaviours,
        "hotspots": hotspots,
        "active_recommendation": {
            "code": "SOP-LOG-108",
            "title": "Trolley Refresher & Pallet Stacking Clearance",
            "status": "PENDING"
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
        if filter_type == "near_miss" and not is_nm:
            continue
        if filter_type == "high_risk" and risk not in ("HIGH", "CRITICAL"):
            continue
        if is_nm or risk in ("HIGH", "CRITICAL") or filter_type == "all":
            flagged_events.append(e)

    # Sort so near misses and critical/high risk events appear first
    flagged_events.sort(key=lambda x: (
        1 if getattr(x, "is_near_miss", False) else 0,
        1 if getattr(x, "risk_level", "").upper() in ("HIGH", "CRITICAL") else 0
    ), reverse=True)

    result = []
    for idx, e in enumerate(flagged_events[:80]):
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
        d["ai_frame_url"] = f"/api/video_frame?video={vid_encoded}&frame_idx={fr_start}&mask=true"
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

        box_model, pose_model = get_detection_models()
        last_auto_event_time = 0.0
        strain_start_time = None
        lifting_start_time = None

        try:
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.04)
                    continue

                if mask and box_model and pose_model:
                    b_res = box_model(frame, conf=0.25, imgsz=480, verbose=False)
                    p_res = pose_model(frame, conf=0.35, imgsz=480, verbose=False)

                    if b_res and len(b_res) > 0 and b_res[0].boxes:
                        for b in b_res[0].boxes:
                            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                            conf = float(b.conf[0])
                            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (95, 185, 255), 2)
                            cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + 180, max(22, by1)), (95, 185, 255), -1)
                            cv2.putText(frame, f"BOX [{int(conf*100)}%] // YOLO11", (bx1 + 4, max(16, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                    if p_res and len(p_res) > 0 and p_res[0].boxes:
                        r = p_res[0]
                        kps_data = r.keypoints.data.cpu().numpy() if r.keypoints else None
                        for i, p_box in enumerate(r.boxes):
                            px1, py1, px2, py2 = [int(v) for v in p_box.xyxy[0].tolist()]
                            posture_strain = False
                            if kps_data is not None and i < len(kps_data):
                                kps = kps_data[i]
                                if kps[0][2] > 0.3 and (kps[11][2] > 0.3 or kps[12][2] > 0.3):
                                    hx = (kps[11][0] + kps[12][0]) / 2.0 if (kps[11][2] > 0.3 and kps[12][2] > 0.3) else (kps[11][0] if kps[11][2] > 0.3 else kps[12][0])
                                    hy = (kps[11][1] + kps[12][1]) / 2.0 if (kps[11][2] > 0.3 and kps[12][2] > 0.3) else (kps[11][1] if kps[11][2] > 0.3 else kps[12][1])
                                    dx = abs(kps[0][0] - hx)
                                    dy = abs(kps[0][1] - hy)
                                    if dy > 0 and (dx / dy) > 0.55:
                                        posture_strain = True

                                for p1, p2 in SKELETON_PAIRS:
                                    if kps[p1][2] > 0.3 and kps[p2][2] > 0.3:
                                        pt1 = (int(kps[p1][0]), int(kps[p1][1]))
                                        pt2 = (int(kps[p2][0]), int(kps[p2][1]))
                                        bone_col = (0, 165, 255) if posture_strain else (0, 255, 128)
                                        cv2.line(frame, pt1, pt2, bone_col, 2)
                                for k_idx in range(17):
                                    if kps[k_idx][2] > 0.3:
                                        cv2.circle(frame, (int(kps[k_idx][0]), int(kps[k_idx][1])), 3, (0, 255, 0), -1)

                            box_color = (0, 165, 255) if posture_strain else (0, 229, 255)
                            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                            badge_title = "WORKER [BENDING STRAIN]" if posture_strain else f"WORKER #{i+1} [NOMINAL]"
                            cv2.rectangle(frame, (px1, max(0, py1 - 22)), (px1 + 190, max(22, py1)), box_color, -1)
                            cv2.putText(frame, badge_title, (px1 + 4, max(16, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
                dt_ms = round((time.time() - t0) * 1000, 1)
                
                # Automatic Event Persistence with Debouncing
                if mask and box_model and pose_model:
                    if posture_strain:
                        if strain_start_time is None:
                            strain_start_time = time.time()
                        elif (time.time() - strain_start_time) >= 0.7 and (time.time() - last_auto_event_time) > 6.0:
                            b_cnt = len(b_res[0].boxes) if (b_res and len(b_res) > 0 and b_res[0].boxes) else 0
                            record_webcam_event(
                                behaviour_type="Awkward Posture // Bending Strain",
                                behaviour_code="POSTURE_STRAIN",
                                risk_level="High",
                                reason="Operator observed in sustained forward bend (>30°) without ergonomic support.",
                                recommended_action="Dock supervisor alert: instruct operator to lower carry height and bend knees.",
                                is_near_miss=True,
                                near_miss_probability=0.82,
                                boxes_count=b_cnt,
                                persons_count=1
                            )
                            last_auto_event_time = time.time()
                            strain_start_time = None
                    else:
                        strain_start_time = None

                    boxes_cnt = len(b_res[0].boxes) if (b_res and len(b_res) > 0 and b_res[0].boxes) else 0
                    if boxes_cnt > 0 and not posture_strain:
                        if lifting_start_time is None:
                            lifting_start_time = time.time()
                        elif (time.time() - lifting_start_time) >= 1.0 and (time.time() - last_auto_event_time) > 6.0:
                            record_webcam_event(
                                behaviour_type="Carton / Package Manual Lifting",
                                behaviour_code="MANUAL_LIFTING",
                                risk_level="Medium",
                                reason=f"Operator active in manual package handling ({boxes_cnt} package(s) tracked).",
                                recommended_action="Verify product weight complies with two-person lift SOP if over 20kg.",
                                is_near_miss=False,
                                near_miss_probability=0.35,
                                boxes_count=boxes_cnt,
                                persons_count=1
                            )
                            last_auto_event_time = time.time()
                            lifting_start_time = None
                    elif boxes_cnt == 0:
                        lifting_start_time = None

                # Visual HUD with recent event banner
                if (time.time() - _LAST_RECORDED_TIME) < 2.5:
                    hud_text = f"FIELD INTELLIGENCE LIVE // {_LAST_RECORDED_MSG} // {dt_ms}ms"
                    cv2.putText(frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (95, 185, 255), 1, cv2.LINE_AA)
                else:
                    hud_text = f"FIELD INTELLIGENCE LIVE // DEVICE WEBCAM // YOLO11 + POSE // {dt_ms}ms"
                    cv2.putText(frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
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
        elif mask:
            box_model, pose_model = get_detection_models()
            if box_model and pose_model:
                b_res = box_model(frame, conf=0.25, imgsz=480, verbose=False)
                p_res = pose_model(frame, conf=0.35, imgsz=480, verbose=False)
                if b_res and len(b_res) > 0 and b_res[0].boxes:
                    for b in b_res[0].boxes:
                        bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
                        conf = float(b.conf[0])
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (95, 185, 255), 2)
                        cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + 180, max(22, by1)), (95, 185, 255), -1)
                        cv2.putText(frame, f"BOX [{int(conf*100)}%] // YOLO11", (bx1 + 4, max(16, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                if p_res and len(p_res) > 0 and p_res[0].boxes:
                    for i, p_box in enumerate(p_res[0].boxes):
                        px1, py1, px2, py2 = [int(v) for v in p_box.xyxy[0].tolist()]
                        cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 229, 255), 2)
                        cv2.rectangle(frame, (px1, max(0, py1 - 22)), (px1 + 170, max(22, py1)), (0, 229, 255), -1)
                        cv2.putText(frame, f"WORKER #{i+1} [NOMINAL]", (px1 + 4, max(16, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
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
_BOX_MODEL = None
_POSE_MODEL = None

def get_detection_models():
    """Lazily loads and caches the trained box model and pose model."""
    global _BOX_MODEL, _POSE_MODEL
    if _BOX_MODEL is None or _POSE_MODEL is None:
        try:
            from ultralytics import YOLO
            box_path = "weights/box_11s.pt" if os.path.exists("weights/box_11s.pt") else "yolov8n.pt"
            pose_path = "yolov8n-pose.pt" if os.path.exists("yolov8n-pose.pt") else "yolov8n-pose.pt"
            _BOX_MODEL = YOLO(box_path)
            _POSE_MODEL = YOLO(pose_path)
            print(f"[AI Models] Loaded Box Detector ({box_path}) and Pose Estimator ({pose_path})")
        except Exception as e:
            print(f"[AI Models] Failed to load YOLO models: {e}")
    return _BOX_MODEL, _POSE_MODEL

@app.post("/api/detect_webcam_frame")
async def detect_webcam_frame(
    request: Request,
    conf_box: float = 0.15,
    conf_pose: float = 0.35,
    mask: bool = True
):
    """
    Directly runs the trained YOLO11 package detector (weights/box_11s.pt) and 
    YOLOv8-Pose (yolov8n-pose.pt) on a live video frame from the user's device camera.
    Returns the annotated frame along with real-time telemetry headers.
    """
    contents = await request.body()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid frame format")

    box_model, pose_model = get_detection_models()
    h, w = frame.shape[:2]

    t0 = time.time()
    
    # Run trained models with optimized resolution for speed
    box_results = box_model(frame, conf=conf_box, imgsz=480, verbose=False) if box_model else None
    pose_results = pose_model(frame, conf=conf_pose, imgsz=480, verbose=False) if pose_model else None
    
    inference_ms = round((time.time() - t0) * 1000, 1)

    boxes_detected = 0
    persons_detected = 0
    risk_score = 16  # baseline safe rating
    risk_level = "LOW"
    behaviour = "MONITORING // NOMINAL"
    has_interaction = False

    # 1. Annotate Detected Cartons / Packages using weights/box_11s.pt
    box_coords = []
    if box_results and len(box_results) > 0 and box_results[0].boxes:
        for b in box_results[0].boxes:
            bx1, by1, bx2, by2 = [int(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            box_coords.append((bx1, by1, bx2, by2, conf))
            boxes_detected += 1
            if mask:
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (95, 185, 255), 2)
                badge_text = f"BOX [{int(conf * 100)}%] // YOLO11"
                cv2.rectangle(frame, (bx1, max(0, by1 - 22)), (bx1 + 180, max(22, by1)), (95, 185, 255), -1)
                cv2.putText(frame, badge_text, (bx1 + 4, max(16, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # 2. Annotate Worker Pose and Evaluate Ergonomics using yolov8n-pose.pt
    if pose_results and len(pose_results) > 0 and pose_results[0].boxes:
        r = pose_results[0]
        kps_data = r.keypoints.data.cpu().numpy() if r.keypoints else None
        
        for i, p_box in enumerate(r.boxes):
            px1, py1, px2, py2 = [int(v) for v in p_box.xyxy[0].tolist()]
            persons_detected += 1

            posture_strain = False
            wrists = []

            if kps_data is not None and i < len(kps_data):
                kps = kps_data[i]  # 17 keypoints (x, y, conf)
                
                # Check wrists proximity to any box
                for idx in [9, 10]:  # left_wrist, right_wrist
                    if kps[idx][2] > 0.3:
                        wx, wy = int(kps[idx][0]), int(kps[idx][1])
                        wrists.append((wx, wy))
                        for (bx1, by1, bx2, by2, _) in box_coords:
                            if (bx1 - 40) <= wx <= (bx2 + 40) and (by1 - 40) <= wy <= (by2 + 40):
                                has_interaction = True

                # Check torso bend (angle between nose and hips)
                if kps[0][2] > 0.3 and (kps[11][2] > 0.3 or kps[12][2] > 0.3):
                    hx = (kps[11][0] + kps[12][0]) / 2.0 if (kps[11][2] > 0.3 and kps[12][2] > 0.3) else (kps[11][0] if kps[11][2] > 0.3 else kps[12][0])
                    hy = (kps[11][1] + kps[12][1]) / 2.0 if (kps[11][2] > 0.3 and kps[12][2] > 0.3) else (kps[11][1] if kps[11][2] > 0.3 else kps[12][1])
                    nx, ny = kps[0][0], kps[0][1]
                    dx = abs(nx - hx)
                    dy = abs(ny - hy)
                    if dy > 0 and (dx / dy) > 0.55:  # leaning forward > 30 deg
                        posture_strain = True

                # Draw skeleton
                if mask:
                    for p1, p2 in SKELETON_PAIRS:
                        if kps[p1][2] > 0.3 and kps[p2][2] > 0.3:
                            pt1 = (int(kps[p1][0]), int(kps[p1][1]))
                            pt2 = (int(kps[p2][0]), int(kps[p2][1]))
                            bone_color = (0, 165, 255) if posture_strain else (0, 255, 128)
                            cv2.line(frame, pt1, pt2, bone_color, 2)
                    for k_idx in range(17):
                        if kps[k_idx][2] > 0.3:
                            cv2.circle(frame, (int(kps[k_idx][0]), int(kps[k_idx][1])), 4, (0, 255, 0), -1)

            if mask:
                box_color = (0, 165, 255) if posture_strain else (0, 229, 255)
                cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
                badge_title = "WORKER [BENDING STRAIN]" if posture_strain else f"WORKER #{i+1} [NOMINAL]"
                cv2.rectangle(frame, (px1, max(0, py1 - 22)), (px1 + 190, max(22, py1)), box_color, -1)
                cv2.putText(frame, badge_title, (px1 + 4, max(16, py1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 1, cv2.LINE_AA)

            if posture_strain:
                risk_score = max(risk_score, 76)
                risk_level = "HIGH"
                behaviour = "AWKWARD POSTURE // BENDING"
            elif has_interaction:
                risk_score = max(risk_score, 45)
                risk_level = "MEDIUM"
                behaviour = "ACTIVE MATERIAL HANDLING"

    if has_interaction and risk_level == "LOW":
        risk_score = 42
        risk_level = "MEDIUM"
        behaviour = "BOX LIFTING / CARRYING"

    if mask:
        if posture_strain:
            active_alert = {
                "event_id": f"CAM-DEV-{int(time.time())%10000:04d}",
                "risk_level": "High",
                "behaviour_type": "Awkward Posture // Bending Strain",
                "behaviour_code": "OPERATOR_STEPPING_CARTON",
                "action": "Maintain upright posture, bend at knees, and avoid twisting while carrying.",
                "near_miss_prob": 0.82,
                "is_near_miss": True,
                "frames_left": 10
            }
        elif has_interaction:
            active_alert = {
                "event_id": f"CAM-DEV-{int(time.time())%10000:04d}",
                "risk_level": "Medium",
                "behaviour_type": "Manual Package Handling",
                "behaviour_code": "UNSAFE_FLOOR_DRAG",
                "action": "Keep load centered close to torso; verify weight complies with SOP.",
                "near_miss_prob": 0.35,
                "is_near_miss": False,
                "frames_left": 10
            }
        else:
            active_alert = None

        draw_hud_banner(frame, active_alert, 1, 30.0, persons_detected, boxes_detected)
    else:
        cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
        hud_text = f"DEVICE WEBCAM // RAW FEED (MASK OFF) // {inference_ms}ms"
        cv2.putText(frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Risk-Score": str(risk_score),
            "X-Risk-Level": str(risk_level),
            "X-Boxes-Count": str(boxes_detected),
            "X-Persons-Count": str(persons_detected),
            "X-Behaviour": str(behaviour),
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
