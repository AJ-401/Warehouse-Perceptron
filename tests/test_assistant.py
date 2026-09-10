"""
test_assistant.py

Sanity checks — not exhaustive, but they catch the failure modes that
actually matter for this project:
  - does the event log load and validate against the schema
  - does filtering actually narrow the context (grounding depends on this)
  - does explain_event refuse gracefully on an unknown event_id
  - does the timestamp/overlap matching in metrics work correctly

Run with: pytest tests/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assistant.event_loader import EventStore
from assistant.llm_client import MockLLMClient
from assistant import queries as q
from metrics.evaluate import _parse_timestamp, _overlaps


def _store():
    return EventStore.load()


def test_event_log_loads_and_validates():
    store = _store()
    assert len(store.events) > 0
    for e in store.events:
        assert e.event_id.startswith("EVT-")
        assert e.risk_level in {"Low", "Medium", "High", "Critical"}
        assert e.confidence_stage in {"Observed", "Potential Risk", "Confirmed Damage"}


def test_all_ten_scenarios_present():
    store = _store()
    codes = {e.behaviour_code for e in store.events}
    expected = {
        "UNSAFE_FLOOR_DRAG", "DROP_HIGH_IMPACT", "DROP_LOW_SLIP",
        "NEAR_MISS_UNSAFE_CARRY", "ROUGH_THROW_SLIDE",
        "OPERATOR_STEPPING_CARTON", "EQUIPMENT_STRAP_LIFT", "STACK_UNSTABLE_WOBBLE",
        "STACK_INVERTED_PYRAMID", "BENCHMARK_SAFE_HANDLING",
    }
    assert expected.issubset(codes)


def test_high_risk_filter_is_narrower_than_full_log():
    store = _store()
    llm = MockLLMClient()
    result = q.get_high_risk_events(store, llm, min_risk="Critical")
    assert 0 < len(result.event_ids) < len(store.events)
    for eid in result.event_ids:
        assert store.get(eid).risk_level == "Critical"


def test_explain_unknown_event_does_not_hallucinate():
    store = _store()
    llm = MockLLMClient()
    result = q.explain_event(store, "EVT-DOES-NOT-EXIST", llm)
    assert result.event_ids == []
    assert "don't have an event" in result.answer.lower()


def test_ask_with_tools_mock_execution():
    store = _store()
    llm = MockLLMClient()
    result = q.ask(store, llm, "Tell me about today in general.")
    assert "[MOCK]" in result.answer


def test_timestamp_parsing():
    assert _parse_timestamp("00:01:24.500") == 84.5
    assert _parse_timestamp("00:00:00.000") == 0.0


def test_overlap_detection():
    assert _overlaps(10.0, 20.0, 15.0, 25.0) is True
    assert _overlaps(10.0, 20.0, 20.0, 30.0) is False  # touching, not overlapping
    assert _overlaps(10.0, 20.0, 30.0, 40.0) is False


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
