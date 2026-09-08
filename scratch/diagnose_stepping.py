import json

with open("outputs_person_a/Stepping on cartons, vertical product kept horizontally, heavy product kept on top_tracking_results.json") as f:
    d = json.load(f)

print("Resolution:", d.get("video_metadata", {}).get("resolution"))
for fr in d["frames"]:
    f_idx = fr["frame_idx"]
    if f_idx in [613, 615, 616, 617]:
        boxes = [t for t in fr["tracks"] if t["class"] != "person"]
        persons = [t for t in fr["tracks"] if t["class"] == "person"]
        print(f"Frame {f_idx}: Boxes={len(boxes)}, Persons={len(persons)}")
        for b in boxes:
            print("  Box:", b["track_id"], "bbox_n:", b.get("bbox_normalized"))
        for p in persons:
            kps = p.get("keypoints", {})
            la = kps.get("left_ankle")
            ra = kps.get("right_ankle")
            print(f"  Person {p['track_id']}: la={la}, ra={ra}")
