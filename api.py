from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from collections import Counter
import os
import sys

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
    
    risks = Counter(getattr(e, "risk_level", "LOW") for e in events)
    behaviours = Counter(getattr(e, "behaviour_type", "UNKNOWN") for e in events)
    cameras = Counter(getattr(e, "camera_id", "UNKNOWN") for e in events)
    
    top_behaviours = [
        {"behaviour": b, "count": c, "share": round((c / total) * 100, 1)} 
        for b, c in behaviours.most_common(5)
    ]
    
    hotspots = [{"camera_id": cam, "count": count} for cam, count in cameras.most_common(4)]

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
        
    return [e.to_dict() if hasattr(e, "to_dict") else vars(e) for e in filtered[:limit]]

@app.get("/api/incidents")
async def get_incidents():
    """Returns all near-miss incidents for the investigation replay view."""
    if store is None:
        return []
    near_misses = [e for e in store.events if getattr(e, "is_near_miss", False)]
    return [e.to_dict() if hasattr(e, "to_dict") else vars(e) for e in near_misses]

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

# Mount the Godrej Stitch Dashboard directory at root
dashboard_dir = os.path.join(os.path.dirname(__file__), "Godrej")
if os.path.isdir(dashboard_dir):
    app.mount("/", StaticFiles(directory=dashboard_dir, html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
