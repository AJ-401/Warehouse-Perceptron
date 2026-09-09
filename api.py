from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

from assistant.event_loader import EventStore, EventLoadError
from assistant.llm_client import GroqLLMClient, LLMClientError
from assistant import queries as q

app = FastAPI(title="Warehouse AI Assistant API", version="1.0")

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

@app.on_event("startup")
async def startup_event():
    global store
    try:
        store = EventStore.load()
        print(f"Loaded {len(store.events)} events on startup.")
    except EventLoadError as e:
        print(f"Failed to load event store: {e}")
        # Not exiting so the API still comes up, but requests will fail clearly
        store = None

class ChatRequest(BaseModel):
    username: str
    question: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if store is None:
        raise HTTPException(status_code=500, detail="Event store is not loaded.")
        
    username = req.username.strip().lower()
    memory_file = f"data/memory/memory_{username}.json" if username else None
    
    if not os.environ.get("GROQ_API_KEY"):
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not set on the server.")
        
    try:
        llm = GroqLLMClient(memory_file=memory_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize LLM: {str(e)}")

    try:
        # Run the tool-calling assistant
        result = q.ask(store, llm, req.question)
        return ChatResponse(answer=result.answer)
    except Exception as e:
        error_str = str(e)
        if "rate_limit_exceeded" in error_str or "413" in error_str:
            raise HTTPException(status_code=429, detail="API rate limit exceeded. Please wait a minute and try again.")
        raise HTTPException(status_code=500, detail=f"LLM request failed: {error_str}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "events_loaded": len(store.events) if store else 0,
        "api_key_set": bool(os.environ.get("GROQ_API_KEY"))
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
