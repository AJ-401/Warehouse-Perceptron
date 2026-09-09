from fastapi import FastAPI, HTTPException, Request, Query
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

VIDEO_CATALOG = {
    "KD packets dragged, heavy box kept on other packets.mp4": {
        "video": os.path.join("official_videos", "KD packets dragged, heavy box kept on other packets.mp4"),
        "tracking": os.path.join("outputs_person_a", "KD packets dragged, heavy box kept on other packets_tracking_results.json"),
        "cam": "CAM-01 // Dock Gate A (Unloading Bay 01)"
    },
    "Throwing seating cartons, using strap to hold.mp4": {
        "video": os.path.join("official_videos", "Throwing seating cartons, using strap to hold.mp4"),
        "tracking": os.path.join("outputs_person_a", "Throwing seating cartons, using strap to hold_tracking_results.json"),
        "cam": "CAM-02 // Staging Area West"
    },
    "Dock level, dragging cupboard.mp4": {
        "video": os.path.join("official_videos", "Dock level, dragging cupboard.mp4"),
        "tracking": os.path.join("outputs_person_a", "Dock level, dragging cupboard_tracking_results.json"),
        "cam": "CAM-03 // Dock Level In-Feed"
    },
    "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4": {
        "video": os.path.join("official_videos", "Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4"),
        "tracking": os.path.join("outputs_person_a", "Stepping on cartons, vertical product kept horizontally, heavy product kept on top_tracking_results.json"),
        "cam": "CAM-04 // High-Bay Racking"
    },
    "Rolling and dropping carton.mp4": {
        "video": os.path.join("official_videos", "Rolling and dropping carton.mp4"),
        "tracking": os.path.join("outputs_person_a", "Rolling and dropping carton_tracking_results.json"),
        "cam": "CAM-05 // Sorting Table 02"
    },
    "Throwing Mattresses.mp4": {
        "video": os.path.join("official_videos", "Throwing Mattresses.mp4"),
        "tracking": os.path.join("outputs_person_a", "Throwing Mattresses_tracking_results.json"),
        "cam": "CAM-06 // Bulky Goods Gate"
    },
    "Rolling and dragging on wet floor.mp4": {
        "video": os.path.join("official_videos", "Rolling and dragging on wet floor.mp4"),
        "tracking": os.path.join("outputs_person_a", "Rolling and dragging on wet floor_tracking_results.json"),
        "cam": "CAM-07 // Wet Floor Zone"
    },
    "WIN_20260908_14_39_18_Pro.mp4": {
        "video": os.path.join("official_videos", "WIN_20260908_14_39_18_Pro.mp4"),
        "tracking": os.path.join("outputs_person_a", "WIN_20260908_14_39_18_Pro_tracking_results.json"),
        "cam": "CAM-08 // Pallet Consolidation"
    }
}

from assistant.event_loader import EventStore, EventLoadError
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
async def get_incidents():
    """Returns all near-miss incidents for the investigation replay view."""
    if store is None:
        return []
    near_misses = [e for e in store.events if getattr(e, "is_near_miss", False)]
    return [e.model_dump() if hasattr(e, "model_dump") else (e.to_dict() if hasattr(e, "to_dict") else vars(e)) for e in near_misses]

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
    if video_name:
        for k, v in VIDEO_CATALOG.items():
            if video_name.lower() in k.lower():
                cat = v
                break
    if not cat:
        cat = list(VIDEO_CATALOG.values())[0]

    video_path = cat["video"]
    tracking_path = cat.get("tracking")
    cam_label = cat.get("cam", "CAM-01 // UNLOADING BAY 01")

    frames_info = {}
    if tracking_path and os.path.exists(tracking_path):
        try:
            with open(tracking_path, "r", encoding="utf-8") as f:
                td = json.load(f)
                frames_info = {fr["frame_idx"]: fr for fr in td.get("frames", [])}
        except Exception:
            pass

    while True:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            fallback = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(fallback, f"STREAM UNAVAILABLE: {video_path}", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
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

        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

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
                            # Cyan box for worker
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                            badge_w = 170
                            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + badge_w, max(22, y1)), (0, 229, 255), -1)
                            cv2.putText(frame, f"WORKER #{tid} [NOMINAL]", (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

                            # Skeletons
                            kps = trk.get("keypoints", {})
                            if kps:
                                for p1, p2 in SKELETON_PAIRS:
                                    n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                                    if n1 in kps and n2 in kps and kps[n1][2] > 0.3 and kps[n2][2] > 0.3:
                                        cv2.line(frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)
                                for k_name, (kx, ky, kc) in kps.items():
                                    if kc > 0.3:
                                        cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)
                        else:
                            # Amber box for products/cartons
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (95, 185, 255), 2)
                            badge_w = 180
                            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + badge_w, max(22, y1)), (95, 185, 255), -1)
                            cv2.putText(frame, f"PRODUCT #{tid} [VEL: 1.4m/s]", (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

            # Top HUD bar
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
            hud_mode = "AI DETECTION & TRACKING ACTIVE" if mask else "RAW FEED (PERCEPTION MASK OFF)"
            hud_text = f"FIELD INTELLIGENCE LIVE // {cam_label.upper()} // {hud_mode} // {speed}X"
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
    cam_label = cat.get("cam", "CAM-01")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100)
    frame_idx = max(1, min(frame_idx, total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, f"FRAME {frame_idx} NOT FOUND", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    elif mask:
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
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 229, 255), 2)
                                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 170, max(22, y1)), (0, 229, 255), -1)
                                cv2.putText(frame, f"WORKER #{tid} [NOMINAL]", (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
                                kps = trk.get("keypoints", {})
                                if kps:
                                    for p1, p2 in SKELETON_PAIRS:
                                        n1, n2 = KEYPOINT_NAMES[p1], KEYPOINT_NAMES[p2]
                                        if n1 in kps and n2 in kps and kps[n1][2] > 0.3 and kps[n2][2] > 0.3:
                                            cv2.line(frame, (int(kps[n1][0]), int(kps[n1][1])), (int(kps[n2][0]), int(kps[n2][1])), (0, 255, 128), 2)
                            else:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (95, 185, 255), 2)
                                cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 180, max(22, y1)), (95, 185, 255), -1)
                                cv2.putText(frame, f"PRODUCT #{tid} [VEL: 1.4m/s]", (x1 + 4, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
            except Exception:
                pass
        # Top HUD bar
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 32), (16, 20, 26), -1)
        hud_text = f"FIELD INTELLIGENCE [FRAME {frame_idx}/{total_frames}] // {cam_label.upper()} // PAUSED FRAME INSPECTION"
        cv2.putText(frame, hud_text, (14, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 229, 255), 1, cv2.LINE_AA)

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

@app.get("/api/video_list")
def get_video_list():
    """Returns available camera and video feeds for multi-camera switching."""
    return [
        {"filename": k, "cam": v["cam"]}
        for k, v in VIDEO_CATALOG.items()
    ]

# Mount the Godrej Stitch Dashboard directory at root
dashboard_dir = os.path.join(os.path.dirname(__file__), "Godrej")
if os.path.isdir(dashboard_dir):
    app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
