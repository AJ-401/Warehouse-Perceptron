"""
schemas.py

Pydantic models mirroring the locked warehouse_events.json schema
(see PRD section 7.2). These give us validation + autocomplete
everywhere else in the assistant package.

If Person A/B ever change a field name in the real pipeline output,
this file is the single place to update — everything downstream
(event_loader, queries) reads through these models, not raw dicts.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class EntitiesInvolved(BaseModel):
    person_track_ids: list[int] = Field(default_factory=list)
    package_track_ids: list[int] = Field(default_factory=list)
    equipment_detected: Optional[str] = None


class Evidence(BaseModel):
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None


class WarehouseEvent(BaseModel):
    event_id: str
    video_id: str
    timestamp_start: str
    timestamp_end: str
    frame_range: list[int]
    location_id: str

    behaviour_type: str
    behaviour_code: str
    risk_level: str  # Low | Medium | High | Critical
    confidence_stage: str  # Observed | Potential Risk | Confirmed Damage

    is_near_miss: bool = False
    near_miss_probability: float = 0.0

    entities_involved: EntitiesInvolved
    # telemetry is intentionally loose (dict) since its keys differ
    # per behaviour_code (drag events have horizontal_distance_m,
    # drop events have drop_height_m, etc.) — see PRD section 7.2
    telemetry: dict = Field(default_factory=dict)

    reason: str
    recommended_action: str
    evidence: Evidence


class ShiftSummary(BaseModel):
    total_events: int
    near_misses_prevented: int
    high_critical_risks: int
    estimated_damage_avoided_inr: Optional[float] = None
    most_frequent_risk: Optional[str] = None


class WarehouseEventLog(BaseModel):
    events: list[WarehouseEvent]
    shift_summary: ShiftSummary
