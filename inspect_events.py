"""
inspect_events.py
=================
Interactive inspector and audit tool for Person B warehouse events.
Displays events aligned with the official Godrej Problem Statement pain points.

Usage:
    python inspect_events.py
    python inspect_events.py "outputs_person_b/KD packets dragged, heavy box kept on other packets_warehouse_events.json"
"""

import json
import glob
import os
import sys
from collections import Counter

def display_events(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    events = data.get("events", [])
    summary = data.get("shift_summary", {})
    video_name = os.path.basename(json_path).replace("_warehouse_events.json", "")

    print("\n" + "=" * 105)
    print(f"VIDEO: {video_name}")
    print(f"FILE : {json_path}")
    print("=" * 105)
    print(f"Shift Summary: Total Events = {summary.get('total_events', len(events))} | "
          f"Near-Misses = {summary.get('near_misses_prevented', 0)} | "
          f"High/Critical = {summary.get('high_critical_risks', 0)} | "
          f"Top Risk = {summary.get('most_frequent_risk', 'None')}")
    print("-" * 105)

    if not events:
        print("  --> [NO EVENTS DETECTED] No safety violations or benchmarks fired.")
        return

    print(f"{'#':<3} | {'Time (Start - End)':<23} | {'Risk':<8} | {'Category (PS Pain Point)':<32} | {'Specific Incident':<30}")
    print("-" * 105)
    for idx, e in enumerate(events, 1):
        ts = f"{e['timestamp_start']} -> {e['timestamp_end']}"
        risk = e.get("risk_level", "Unknown")
        cat = e.get("category", "General Handling")
        btype = e.get("behaviour_type", e.get("behaviour_code", "Unknown"))
        prob = e.get("near_miss_probability", 0.0)
        
        tag = f"{btype} (USP: {int(prob*100)}%)" if prob > 0 else btype
        print(f"{idx:<3} | {ts:<23} | {risk:<8} | {cat[:30]:<32} | {tag[:30]:<30}")
        print(f"    Reason   : {e.get('reason')}")
        print(f"    Action   : {e.get('recommended_action')}")
        ev = e.get("evidence", {})
        snap_exists = os.path.exists(ev.get("snapshot_url", ""))
        clip_exists = os.path.exists(ev.get("clip_url", ""))
        print(f"    Evidence : Snapshot=[{'OK' if snap_exists else 'MISSING'}] {ev.get('snapshot_url')} | Clip=[{'OK' if clip_exists else 'MISSING'}] {ev.get('clip_url')}")
        print("-" * 105)

def display_audit_summary():
    merged_path = "outputs_person_b/warehouse_events.json"
    if not os.path.exists(merged_path):
        return

    with open(merged_path, "r") as f:
        data = json.load(f)

    events = data.get("events", [])
    print("\n" + "=" * 105)
    print("           GODREJ HACKATHON: 10-SCENARIO DEMONSTRATION & AUDIT COMPLIANCE MATRIX")
    print("=" * 105)

    cat_counts = Counter(e.get("category", "Unknown") for e in events)
    code_counts = Counter(e.get("behaviour_code", "Unknown") for e in events)

    print("\n1. OFFICIAL PROBLEM STATEMENT OPERATIONAL PAIN POINTS:")
    for cat, count in sorted(cat_counts.items()):
        print(f"   • {cat:<45} : {count:>4} events logged")

    print("\n2. SPECIFIC PREDEFINED SCENARIOS (LINE 242 COMPLIANCE):")
    print(f"   {'#':<3} | {'Code Identifier':<27} | {'Scenario Name':<38} | {'Detections':<10} | {'Status'}")
    print("   " + "-" * 100)

    from pipeline_person_b import RiskEngine
    seen = set()
    s_idx = 1
    for code, (cat, name, _) in RiskEngine.SCENARIO_MAP.items():
        if name in seen:
            continue
        seen.add(name)
        count = code_counts.get(code, 0)
        status = "[PASS] VERIFIED" if count > 0 else "[FAIL] NO EVENTS"
        print(f"   {s_idx:<3} | {code:<27} | {name:<38} | {count:<10} | {status}")
        s_idx += 1

    print("=" * 105 + "\n")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        display_events(path)
    else:
        display_audit_summary()
        files = sorted(glob.glob("outputs_person_b/*_warehouse_events.json"))
        if not files:
            print("[Info] No event files found in outputs_person_b/. Run pipeline_person_b.py first!")
            return

        print("Available individual video event files:")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {os.path.basename(f)}")

        print("\nDisplaying summary for all files...\n")
        for f in files:
            if "warehouse_events.json" not in os.path.basename(f):
                display_events(f)

if __name__ == "__main__":
    main()
