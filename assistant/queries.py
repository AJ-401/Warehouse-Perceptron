"""
queries.py

The supervisor-facing functions. Each one:
  1. filters EventStore down to only the relevant events,
  2. builds a grounded context block (system_prompt.build_context_block),
  3. asks the LLM a narrowly-scoped question,
  4. returns both the LLM's answer AND the raw event_ids used —
     so the dashboard can show "based on these N events" alongside
     the answer, and so it's testable without trusting the LLM's word
     for what it looked at.

Every function takes an `llm` argument (an LLMClient or MockLLMClient)
rather than constructing one internally — makes these easy to test
and easy to swap models later.
"""

from __future__ import annotations
from collections import Counter

from .event_loader import EventStore
from .schemas import WarehouseEvent
from .system_prompt import SYSTEM_INSTRUCTIONS, build_context_block


class QueryResult:
    """Simple return wrapper: the answer text plus which events grounded it."""

    def __init__(self, answer: str, event_ids: list[str]):
        self.answer = answer
        self.event_ids = event_ids

    def to_dict(self) -> dict:
        return {"answer": self.answer, "event_ids_used": self.event_ids}

    def __repr__(self) -> str:
        return f"QueryResult(event_ids={self.event_ids})"


# ---------------------------------------------------------------------------
# 1. High-risk events
# ---------------------------------------------------------------------------

def get_high_risk_events(store: EventStore, llm, min_risk: str = "High") -> QueryResult:
    order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    threshold = order.get(min_risk.lower(), 2)
    events = sorted(
        [e for e in store.events if order.get(e.risk_level.lower(), 0) >= threshold],
        key=lambda x: order.get(x.risk_level.lower(), 0), 
        reverse=True
    )[:5]

    context = build_context_block(events, label=f"TOP 5 EVENTS AT OR ABOVE {min_risk.upper()} RISK")
    user_message = (
        f"List and briefly explain each event at or above {min_risk} risk this shift. "
        f"Group by location if there's more than one per bay."
    )
    answer = llm.ask(SYSTEM_INSTRUCTIONS, user_message, context=context)
    return QueryResult(answer, [e.event_id for e in events])


# ---------------------------------------------------------------------------
# 2. Shift summary
# ---------------------------------------------------------------------------

def shift_summary_data(store: EventStore) -> dict:
    """Structured (non-LLM) shift stats — useful for the dashboard directly."""
    behaviour_counts = Counter(e.behaviour_type for e in store.events)
    risk_counts = Counter(e.risk_level for e in store.events)
    return {
        "total_events": len(store.events),
        "by_behaviour": dict(behaviour_counts),
        "by_risk_level": dict(risk_counts),
        "near_misses_count": sum(1 for e in store.events if e.is_near_miss),
        "repeated_behaviours_count": len(store.repeated_behaviours()),
        "locations": store.locations(),
        "reported_summary": store.shift_summary.model_dump(),
    }


def shift_summary_narrative(store: EventStore, llm) -> QueryResult:
    events = sorted(store.events, key=lambda x: (x.risk_level == "Critical", x.is_near_miss), reverse=True)[:5]
    context = build_context_block(events, label="TOP 5 CRITICAL/NEAR-MISS EVENTS")
    data = shift_summary_data(store)
    user_message = (
        "Write a concise shift summary for a supervisor: total events, the most "
        "common risky behaviour, any near misses, and top locations.\n\nData: {data}"
    )
    answer = llm.ask(SYSTEM_INSTRUCTIONS, user_message, context=context)
    return QueryResult(answer, [e.event_id for e in events])


# ---------------------------------------------------------------------------
# 3. Explain a specific event
# ---------------------------------------------------------------------------

def explain_event(store: EventStore, event_id: str, llm) -> QueryResult:
    event = store.get(event_id)
    if event is None:
        return QueryResult(
            f"I don't have an event with ID {event_id} in the current data.", []
        )
    context = build_context_block([event], label="EVENT")
    user_message = (
        f"Explain why {event_id} was flagged, in plain language a supervisor can "
        f"act on, referencing the telemetry that triggered it."
    )
    answer = llm.ask(SYSTEM_INSTRUCTIONS, user_message, context=context)
    return QueryResult(answer, [event.event_id])


# ---------------------------------------------------------------------------
# 4. Compare bays
# ---------------------------------------------------------------------------

def compare_bays(store: EventStore, llm) -> QueryResult:
    risk_by_location = store.risk_counts_by_location()
    events = sorted(store.events, key=lambda x: (x.risk_level == "Critical", x.is_near_miss), reverse=True)[:5]
    context = build_context_block(events, label="SAMPLE 5 EVENTS ACROSS BAYS")
    user_message = (
        "Compare risk levels across bays this shift and identify which bay needs "
        f"attention first, and why. Risk counts by location: {dict(risk_by_location)}"
    )
    answer = llm.ask(SYSTEM_INSTRUCTIONS, user_message, context=context)
    return QueryResult(answer, [e.event_id for e in events])


# ---------------------------------------------------------------------------
# 5. Coaching notes from recurring patterns
# ---------------------------------------------------------------------------

def coaching_notes(store: EventStore, llm) -> QueryResult:
    repeated = store.repeated_behaviours()[:5]
    if not repeated:
        return QueryResult("No repeated behaviours logged this shift — nothing to coach on yet.", [])

    context = build_context_block(repeated, label="TOP 5 REPEATED BEHAVIOUR EVENTS")
    user_message = (
        "These events are flagged as repeated behaviours this shift. Turn them "
        "into short, specific coaching notes a supervisor could read out at a "
        "morning huddle to prevent these issues today."
    )
    answer = llm.ask(SYSTEM_INSTRUCTIONS, user_message, context=context)
    return QueryResult(answer, [e.event_id for e in repeated])


def ask(store: EventStore, llm, question: str) -> QueryResult:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "filter_events",
                "description": "Fetch a list of events matching specific criteria. Can also fetch by exact event_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "risk_level": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                        "keyword": {"type": "string", "description": "Search term for behaviour or reason (e.g., 'dragging', 'strap')"},
                        "location_id": {"type": "string"},
                        "near_miss_only": {"type": "boolean"}
                    }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_aggregate_counts",
                "description": "Get global aggregate shift statistics (total events, near misses).",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_location_breakdown",
                "description": "Get risk counts broken down by warehouse location.",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    def tool_handler(name: str, args: dict):
        if name == "filter_events":
            if args.get("event_id"):
                e = store.get(args.get("event_id"))
                matches = [e] if e else []
            else:
                matches = store.filter(
                    risk_level=args.get("risk_level"),
                    keyword=args.get("keyword"),
                    location_id=args.get("location_id"),
                    near_miss_only=args.get("near_miss_only", False)
                )
            
            total_matches = len(matches)
            matches = matches[:15]  # Cap to prevent Groq TPM limit errors

            # Aggressively strip data to avoid token limits
            stripped = []
            for e in matches:
                stripped.append({
                    "event_id": e.event_id,
                    "timestamp": e.timestamp_start,
                    "behaviour": e.behaviour_type,
                    "reason": e.reason,
                    "risk_level": e.risk_level,
                    "location_id": e.location_id,
                    "is_near_miss": e.is_near_miss
                })
            return {"total_matches_in_database": total_matches, "events": stripped}
            
        elif name == "get_aggregate_counts":
            return {
                "total_events": len(store.events),
                "total_near_misses": sum(1 for e in store.events if e.is_near_miss),
            }
            
        elif name == "get_location_breakdown":
            return dict(store.risk_counts_by_location())
            
        return {"error": "Unknown tool"}

    if hasattr(llm, "ask_with_tools"):
        answer = llm.ask_with_tools(SYSTEM_INSTRUCTIONS, question, tools, tool_handler)
    else:
        answer = llm.ask(SYSTEM_INSTRUCTIONS, question)
        
    return QueryResult(answer, [])
