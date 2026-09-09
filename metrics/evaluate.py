"""
evaluate.py

Compares the pipeline's predicted events (warehouse_events.json) against
manually-labeled ground truth (ground_truth_labels.json — created AFTER
staged clips are filmed and someone marks which segments were meant to
be risky vs safe). See data/ground_truth_labels.example.json for the
expected format and an explanation of why this can't exist yet.

Matching logic: a ground-truth label and a predicted event are considered
the same real-world moment if they share a video_id AND their time ranges
overlap. This is deliberately simple (no IoU threshold, no fuzzy matching)
because the dataset is small and hand-labeled — precision here matters
more than sophistication.

"Risky" is defined as any event with risk_level in {Medium, High, Critical}.
An event logged as SAFE_HANDLING_CONTROL (risk_level=Low) is treated as a
correct non-risky detection, not ignored.
"""

from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass, field

from assistant.event_loader import EventStore
from assistant.schemas import WarehouseEvent

RISKY_LEVELS = {"medium", "high", "critical"}


@dataclass
class GroundTruthLabel:
    video_id: str
    segment_start_sec: float
    segment_end_sec: float
    true_behaviour_code: str
    is_risky: bool


@dataclass
class ConfusionMatrix:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    matched_pairs: list = field(default_factory=list)  # (label, matched_event_or_None)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0


def _parse_timestamp(ts: str) -> float:
    """'00:01:24.500' -> 84.5 seconds"""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def load_ground_truth(path: str | Path) -> list[GroundTruthLabel]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No ground truth file at {path}. This is expected until staged clips "
            "are filmed and labeled — see data/ground_truth_labels.example.json "
            "for the format to produce."
        )
    raw = json.loads(path.read_text())
    return [
        GroundTruthLabel(
            video_id=item["video_id"],
            segment_start_sec=item["segment_start_sec"],
            segment_end_sec=item["segment_end_sec"],
            true_behaviour_code=item["true_behaviour_code"],
            is_risky=item["is_risky"],
        )
        for item in raw["labels"]
    ]


def find_matching_events(label: GroundTruthLabel, events: list[WarehouseEvent]) -> list[WarehouseEvent]:
    matches = []
    for e in events:
        if e.video_id != label.video_id:
            continue
        e_start = _parse_timestamp(e.timestamp_start)
        e_end = _parse_timestamp(e.timestamp_end)
        if _overlaps(label.segment_start_sec, label.segment_end_sec, e_start, e_end):
            matches.append(e)
    return matches


def evaluate(store: EventStore, ground_truth_path: str | Path) -> ConfusionMatrix:
    labels = load_ground_truth(ground_truth_path)
    cm = ConfusionMatrix()

    for label in labels:
        matches = find_matching_events(label, store.events)
        predicted_risky = any(m.risk_level.lower() in RISKY_LEVELS for m in matches)
        matched_event = matches[0] if matches else None

        if label.is_risky and predicted_risky:
            cm.tp += 1
        elif label.is_risky and not predicted_risky:
            cm.fn += 1
        elif not label.is_risky and predicted_risky:
            cm.fp += 1
        else:
            cm.tn += 1

        cm.matched_pairs.append((label, matched_event))

    return cm


if __name__ == "__main__":
    # python -m metrics.evaluate  (run once ground_truth_labels.json exists)
    from pathlib import Path as _P

    store = EventStore.load()
    gt_path = _P(__file__).resolve().parent.parent / "data" / "ground_truth_labels.json"
    try:
        cm = evaluate(store, gt_path)
    except FileNotFoundError as e:
        print(e)
        print("\nRunning against the example template instead, for a dry-run check:")
        gt_path = _P(__file__).resolve().parent.parent / "data" / "ground_truth_labels.example.json"
        cm = evaluate(store, gt_path)

    print(f"TP={cm.tp} FP={cm.fp} FN={cm.fn} TN={cm.tn}")
    print(f"Precision={cm.precision:.2f} Recall={cm.recall:.2f} F1={cm.f1:.2f}")
    print(f"Accuracy={cm.accuracy:.2f} False Positive Rate={cm.false_positive_rate:.2f}")
