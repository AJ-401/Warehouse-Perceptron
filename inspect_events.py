"""
inspect_events.py
=================
Quick interactive inspector for Person B warehouse events.
Usage:
    python inspect_events.py
    python inspect_events.py "outputs_person_b/Rolling and dropping carton_warehouse_events.json"
"""

import json
import glob
import os
import sys

def display_events(json_path):
    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    events = data.get("events", [])
    summary = data.get("shift_summary", {})
    video_name = os.path.basename(json_path).replace("_warehouse_events.json", "")

    print("\n" + "=" * 90)
    print(f"VIDEO: {video_name}")
    print(f"FILE : {json_path}")
    print("=" * 90)
    print(f"Shift Summary: Total Events = {summary.get('total_events', len(events))} | "
          f"Near-Misses = {summary.get('near_misses_prevented', 0)} | "
          f"High/Critical = {summary.get('high_critical_risks', 0)} | "
          f"Top Risk = {summary.get('most_frequent_risk', 'None')}")
    print("-" * 90)

    if not events:
        print("  --> [NO EVENTS DETECTED] No safety violations or benchmarks fired.")
        return

    print(f"{'#':<3} | {'Time (Start - End)':<23} | {'Risk':<8} | {'Behaviour Type':<35}")
    print("-" * 90)
    for idx, e in enumerate(events, 1):
        ts = f"{e['timestamp_start']} -> {e['timestamp_end']}"
        risk = e.get("risk_level", "Unknown")
        btype = e.get("behaviour_type", e.get("behaviour_code", "Unknown"))
        print(f"{idx:<3} | {ts:<23} | {risk:<8} | {btype:<35}")
        print(f"    Reason : {e.get('reason')}")
        print(f"    Action : {e.get('recommended_action')}")
        print(f"    Telemetry: {e.get('telemetry')}")
        print("-" * 90)

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
        display_events(path)
    else:
        # Default: list all generated event files and show summary
        files = sorted(glob.glob("outputs_person_b/*_warehouse_events.json"))
        if not files:
            print("[Info] No event files found in outputs_person_b/. Run pipeline_person_b.py first!")
            return
        
        print("\nFound the following event files:")
        for i, f in enumerate(files, 1):
            print(f"  [{i}] {os.path.basename(f)}")
        
        print("\nDisplaying summary for all files...\n")
        for f in files:
            display_events(f)

if __name__ == "__main__":
    main()
