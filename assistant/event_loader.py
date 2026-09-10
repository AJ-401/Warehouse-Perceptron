"""
event_loader.py

Loads data/warehouse_events.json, validates it against schemas.py,
and exposes a small helper class that queries.py builds on top of.

This is the ONLY module that touches the raw JSON file. Everything
else (queries.py, llm_client.py) should go through EventStore so
there's a single point of truth for "what data actually exists" —
which matters a lot for the no-hallucination guarantee: the LLM
should never see more than what EventStore hands it.
"""

from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

from .schemas import WarehouseEventLog, WarehouseEvent

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "outputs_person_b" / "warehouse_events.json"


class EventLoadError(Exception):
    """Raised when warehouse_events.json is missing or fails schema validation."""


class EventStore:
    """
    In-memory, validated view of the event log.

    Usage:
        store = EventStore.load()
        store.events                 -> list[WarehouseEvent]
        store.get(event_id)          -> WarehouseEvent | None
        store.filter(risk_level="High")
        store.by_location("Bay 02 - Unloading Dock")
        store.most_common_behaviour()
    """

    def __init__(self, log: WarehouseEventLog):
        self._log = log

    # ---- loading -----------------------------------------------------

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DATA_PATH) -> "EventStore":
        path = Path(path)
        if not path.exists():
            raise EventLoadError(f"No event log found at {path}")

        try:
            raw = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise EventLoadError(f"warehouse_events.json is not valid JSON: {e}") from e

        try:
            log = WarehouseEventLog.model_validate(raw)
        except Exception as e:  # pydantic ValidationError
            raise EventLoadError(
                f"warehouse_events.json does not match the locked schema: {e}"
            ) from e

        instance = cls(log)
        instance._path = path
        return instance

    def add_event(self, event: WarehouseEvent, persist_paths: list[str | Path] | None = None) -> None:
        """Appends a new event, updates shift_summary metrics, and persists to JSON on disk."""
        self._log.events.insert(0, event)

        # Update shift summary counts
        summary = self._log.shift_summary
        summary.total_events = len(self._log.events)
        summary.near_misses_prevented = sum(1 for e in self._log.events if e.is_near_miss)
        summary.high_critical_risks = sum(1 for e in self._log.events if e.risk_level.lower() in ("high", "critical"))

        # Most frequent risk
        counts = Counter(e.behaviour_type for e in self._log.events)
        if counts:
            summary.most_frequent_risk = counts.most_common(1)[0][0]

        # Persist to disk
        targets = list(persist_paths) if persist_paths else ([self._path] if hasattr(self, "_path") and self._path else [])
        for p in targets:
            try:
                p_path = Path(p)
                p_path.parent.mkdir(parents=True, exist_ok=True)
                data = self._log.model_dump() if hasattr(self._log, "model_dump") else self._log.dict()
                p_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception as ex:
                print(f"[EventStore] Warning: Failed to persist event to {p}: {ex}")

    # ---- basic access --------------------------------------------------

    @property
    def events(self) -> list[WarehouseEvent]:
        return self._log.events

    @property
    def shift_summary(self):
        return self._log.shift_summary

    def get(self, event_id: str) -> WarehouseEvent | None:
        return next((e for e in self.events if e.event_id == event_id), None)

    # ---- filters used by queries.py -------------------------------------

    def filter(
        self,
        risk_level: str | None = None,
        keyword: str | None = None,
        location_id: str | None = None,
        near_miss_only: bool = False,
    ) -> list[WarehouseEvent]:
        results = self.events
        if risk_level:
            results = [e for e in results if e.risk_level.lower() == risk_level.lower()]
        if keyword:
            k = keyword.lower()
            results = [
                e for e in results
                if k in e.behaviour_code.lower() 
                or k in e.behaviour_type.lower() 
                or k in e.reason.lower()
            ]
        if location_id:
            results = [e for e in results if e.location_id == location_id]
        if near_miss_only:
            results = [e for e in results if e.is_near_miss]
        return results

    def by_location(self, location_id: str) -> list[WarehouseEvent]:
        return self.filter(location_id=location_id)

    def locations(self) -> list[str]:
        return sorted({e.location_id for e in self.events})

    def most_common_behaviour(self) -> tuple[str, int] | None:
        counts = Counter(e.behaviour_type for e in self.events)
        if not counts:
            return None
        return counts.most_common(1)[0]

    def repeated_behaviours(self) -> list[WarehouseEvent]:
        return [e for e in self.events if e.telemetry.get("is_repeated_behaviour")]

    def risk_counts_by_location(self) -> dict[str, Counter]:
        out: dict[str, Counter] = {}
        for e in self.events:
            out.setdefault(e.location_id, Counter())[e.risk_level] += 1
        return out


if __name__ == "__main__":
    # quick manual sanity check: python -m assistant.event_loader
    store = EventStore.load()
    print(f"Loaded {len(store.events)} events")
    print("Locations:", store.locations())
    print("Most common behaviour:", store.most_common_behaviour())
    print("Repeated behaviours:", [e.event_id for e in store.repeated_behaviours()])
