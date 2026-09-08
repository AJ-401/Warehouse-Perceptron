# Data Contract — Person A (Perception) to Person B & C
### Godrej Warehouse Video Intelligence Hackathon | graVITas 2026

---

## 1. Role Boundary & Overview

* **Person A (Perception & Tracking):** Ingests video, detects humans & boxes, extracts 17-point skeletal keypoints, tracks objects with persistent IDs across time, and handles non-linear motion (dynamic floor-clamped rolling & velocity-delta drop uncoupling).
* **Person B (Behavior & Risk Engine):** Ingests `*_tracking_results.json`, computes kinematics across rolling temporal windows, and generates `warehouse_events.json` (10 bad practice scenarios + Predictive Near-Miss USP).
* **Person C (AI Assistant & Metrics):** Ingests `warehouse_events.json`, computes precision/recall against ground truth, and powers the grounded LLM supervisor assistant.
* **Person D (Dashboard & Presentation):** Renders the frontend interface in Google Stitch using the video clips and event streams.

---

## 2. File Deliverables

For each processed video `<video_basename>.mp4`, Person A generates two production deliverables:

1. **Perception Video:** `<output_dir>/<video_basename>_perception.mp4`
   * Annotated preview with bounding boxes, skeletal wireframes, and dynamic HUD status.
2. **Data Contract JSON:** `<output_dir>/<video_basename>_tracking_results.json`
   * The machine-readable JSON ingested directly by Person B's kinematic engine.

---

## 3. JSON Schema Specification

```json
{
  "video_metadata": {
    "video_id": "Rolling and dropping carton.mp4",
    "filename": "Rolling and dropping carton.mp4",
    "fps": 30.0,
    "resolution": [1920, 1080],
    "total_frames_processed": 282,
    "processing_time_sec": 48.5,
    "average_fps": 5.81
  },
  "frames": [
    {
      "frame_idx": 120,
      "frame_id": 120,
      "timestamp_sec": 4.00,
      "tracks": [
        {
          "track_id": 5,
          "class": "person",
          "bbox": [226.0, 0.0, 1103.1, 1068.2],
          "bbox_normalized": [0.118, 0.0, 0.575, 0.989],
          "confidence": 0.927,
          "keypoints": {
            "nose": [1042.0, 72.7, 0.886],
            "left_shoulder": [839.6, 301.2, 0.875],
            "right_shoulder": [606.6, 219.6, 0.983],
            "left_elbow": [765.3, 660.8, 0.709],
            "right_elbow": [328.3, 609.4, 0.959],
            "left_wrist": [850.7, 898.2, 0.622],
            "right_wrist": [710.0, 891.0, 0.917],
            "left_hip": [632.3, 998.8, 0.586],
            "right_hip": [452.2, 988.7, 0.714],
            "left_knee": [764.2, 1080.0, 0.010],
            "right_knee": [516.9, 1080.0, 0.015],
            "left_ankle": [602.3, 1080.0, 0.001],
            "right_ankle": [418.2, 985.4, 0.001]
          }
        },
        {
          "track_id": 1001,
          "class": "cardboard box",
          "bbox": [610.5, 860.5, 1386.3, 1065.5],
          "bbox_normalized": [0.318, 0.797, 0.722, 0.987],
          "confidence": 0.717,
          "held_by": 5,
          "inferred": true,
          "state": "ROLLING",
          "event": "ROLLING"
        }
      ]
    }
  ]
}
```

---

## 4. Track Field Definitions

### 4.1 Common Fields
| Field | Type | Description |
| :--- | :--- | :--- |
| `track_id` | Integer | Persistent ID across frames. Worker IDs start at `1`; Box IDs are offset $\ge 1000$. |
| `class` | String | `"person"` or `"cardboard box"`. |
| `bbox` | `[x1, y1, x2, y2]` | Absolute pixel bounding box in `[left, top, right, bottom]` coordinates. |
| `bbox_normalized`| `[nx1, ny1, nx2, ny2]` | Scale-invariant bounding box normalized to $[0.0, 1.0]$ range (`val / width`, `val / height`). |
| `confidence` | Float | Detection / tracking confidence score between `0.0` and `1.0`. |

### 4.2 Person Track Fields
* **`keypoints`**: Dictionary of 17 standard COCO joints. Each joint maps to `[x, y, confidence]`:
  * Head: `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`
  * Upper Body: `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`
  * Lower Body: `left_hip`, `right_hip`, `left_knee`, `right_knee`, `left_ankle`, `right_ankle`
  * *Usage for Person B:* Use `left_wrist` and `right_wrist` for grasp/release analysis; `left_ankle` and `right_ankle` for floor reference; `left_hip` and `left_knee` for ergonomic lifting posture.

### 4.3 Box Track Fields (HOI Metadata)
| Field | Type | Values | Description |
| :--- | :--- | :--- | :--- |
| `state` | String | `"RESTING"`, `"HELD"`, `"ROLLING"`, `"DROPPED"` | High-level physical handling state. |
| `held_by` | Integer / `null` | Worker `track_id` (e.g. `5`) or `null` | Active carrier ID if within $\le 90\text{px}$ grasp distance. |
| `inferred` | Boolean | `true` / `false` | `true` if projected mathematically via floor clamping / top-edge wrist anchoring. |
| `event` | String / `null` | `"ROLLING"`, `"FREE_FALL"`, `"SETTLED"`, `"MANIPULATING"`, `null` | Instantaneous physical event classification. |

---

## 5. Visual Representation Legend (Video Preview)

When reviewing `<video_basename>_perception.mp4`:
* **Worker Skeleton:**
  * Bones: Vibrant Green (`#00FF80`)
  * Wrists (Grasp points): Bright Red circles (`#0000FF`)
  * Ankles (Floor anchors): Magenta circles (`#FF00FF`)
* **Box Bounding Boxes:**
  * Orange (`#008CFF`): Static / resting on floor or pallet (`RESTING`).
  * Gold (`#00D7FF`): Active human manipulation or rolling (`HELD`, `ROLLING`).
  * Amber (`#00A5FF`): In-flight gravitational free-fall (`FREE-FALL`).
* **HUD Bar:** Displays real-time frame index, timestamp, active worker count, active box count, and processing FPS.

---

## 6. Sample Consumption Code for Person B

```python
import json
import numpy as np

def load_tracking_contract(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    
    fps = data["video_metadata"]["fps"]
    
    for frame in data["frames"]:
        f_idx = frame["frame_idx"]
        t_sec = frame["timestamp_sec"]
        
        persons = {t["track_id"]: t for t in frame["tracks"] if t["class"] == "person"}
        boxes = [t for t in frame["tracks"] if t["class"] == "cardboard box"]
        
        for box in boxes:
            box_id = box["track_id"]
            state = box.get("state")
            carrier_id = box.get("held_by")
            bbox = box["bbox"]
            
            # 1. Detect Dragging (Floor contact + horizontal movement without lift)
            if state == "ROLLING" or (state == "RESTING" and carrier_id is not None):
                # Trigger DRAG_NO_EQUIPMENT logic
                pass
                
            # 2. Detect High-Impact Drop / Throw
            if state == "DROPPED" and box.get("event") == "FREE_FALL":
                # Compute vertical velocity and trigger UNCONTROLLED_DROP logic
                pass
                
            # 3. Predictive Near-Miss (USP)
            if state == "HELD" and carrier_id in persons:
                carrier = persons[carrier_id]
                # Check carrier speed and box elevation relative to hazardous edges
                # Trigger NEAR_MISS_UNSAFE_CARRY alert before drop occurs!
                pass

if __name__ == "__main__":
    load_tracking_contract("outputs_person_a/Rolling and dropping carton_tracking_results.json")
```
