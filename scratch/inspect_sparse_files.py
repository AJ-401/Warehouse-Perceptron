import json

def inspect_file(path):
    print("=" * 60)
    print("INSPECTING:", path)
    with open(path) as f:
        d = json.load(f)
    print("Total frames:", len(d["frames"]))
    person_counts = []
    box_counts = []
    for fr in d["frames"]:
        p_c = sum(1 for t in fr["tracks"] if t["class"] == "person")
        b_c = sum(1 for t in fr["tracks"] if t["class"] != "person")
        person_counts.append(p_c)
        box_counts.append(b_c)
    print(f"Frames with persons: {sum(1 for c in person_counts if c > 0)} / {len(person_counts)}")
    print(f"Frames with boxes: {sum(1 for c in box_counts if c > 0)} / {len(box_counts)}")
    
    # Check if there are boxes in specific frames
    for fr in d["frames"]:
        boxes = [t for t in fr["tracks"] if t["class"] != "person"]
        if boxes:
            print(f"Frame {fr['frame_idx']} (t={fr['timestamp_sec']}s): {len(boxes)} boxes")
            for b in boxes:
                print(f"   Box {b['track_id']}: state={b.get('state')}, event={b.get('event')}, bbox_n={b.get('bbox_normalized')}")

inspect_file("outputs_person_a/Rolling and dragging on wet floor_tracking_results.json")
inspect_file("outputs_person_a/Throwing Mattresses_tracking_results.json")
