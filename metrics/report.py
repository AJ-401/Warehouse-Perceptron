"""
report.py

Turns a ConfusionMatrix (from evaluate.py) into a readable table for
the slide deck (PRD slide 5: "Impact, Damage Prevention & User
Validation" needs real precision/recall numbers, honestly scoped).
"""

from __future__ import annotations
from .evaluate import ConfusionMatrix


def as_markdown_table(cm: ConfusionMatrix) -> str:
    return (
        "| Metric | Value |\n"
        "|---|---|\n"
        f"| True Positives | {cm.tp} |\n"
        f"| False Positives | {cm.fp} |\n"
        f"| False Negatives | {cm.fn} |\n"
        f"| True Negatives | {cm.tn} |\n"
        f"| Precision | {cm.precision:.2%} |\n"
        f"| Recall | {cm.recall:.2%} |\n"
        f"| F1 Score | {cm.f1:.2%} |\n"
        f"| Accuracy | {cm.accuracy:.2%} |\n"
        f"| False Positive Rate | {cm.false_positive_rate:.2%} |\n"
        f"| Labeled clips evaluated | {cm.total} |\n"
    )


def as_plain_summary(cm: ConfusionMatrix) -> str:
    return (
        f"Evaluated against {cm.total} manually-labeled clips: "
        f"precision {cm.precision:.0%}, recall {cm.recall:.0%}, "
        f"F1 {cm.f1:.0%}, accuracy {cm.accuracy:.0%}. "
        f"{cm.fp} false positive(s), {cm.fn} false negative(s) — "
        "scoped strictly to this hackathon's staged dataset, not a "
        "production claim."
    )


def print_mismatches(cm: ConfusionMatrix) -> None:
    """Prints the specific labeled clips the pipeline got wrong — useful
    for debugging the rule engine (Person B) rather than just knowing
    the aggregate score."""
    for label, matched_event in cm.matched_pairs:
        predicted_risky = matched_event is not None and matched_event.risk_level.lower() in {
            "medium", "high", "critical",
        }
        if predicted_risky != label.is_risky:
            got = matched_event.event_id if matched_event else "no matching event"
            print(
                f"MISMATCH: {label.video_id} [{label.segment_start_sec}-"
                f"{label.segment_end_sec}s] expected is_risky={label.is_risky} "
                f"({label.true_behaviour_code}) -> got {got}"
            )


if __name__ == "__main__":
    # python -m metrics.report
    from pathlib import Path
    from assistant.event_loader import EventStore
    from .evaluate import evaluate

    store = EventStore.load()
    root = Path(__file__).resolve().parent.parent
    gt_path = root / "data" / "ground_truth_labels.json"
    if not gt_path.exists():
        gt_path = root / "data" / "ground_truth_labels.example.json"
        print(f"(using example template — real file not created yet: {gt_path.name})\n")

    cm = evaluate(store, gt_path)
    print(as_markdown_table(cm))
    print(as_plain_summary(cm))
    print()
    print_mismatches(cm)
