# AI Video Intelligence for Warehouse Handling — Perception Pipeline (Person A)
### Godrej Enterprises Group Hackathon @ graVITas 2026 | Team of 4

---

## 1. Overview & Person A Responsibilities

This repository contains the **Person A Perception & Tracking Architecture** for the Godrej Hackathon. 

Person A delivers:
1. **Supervised Box Detection:** Custom-trained YOLO11 package detector (`weights/box_11s.pt`) calibrated for warehouse corrugated cartons.
2. **Worker & Pose Keypoints:** YOLOv8 Pose model (`yolov8n-pose.pt`) extracting 17 skeletal joints per worker for ergonomic and grasp analysis.
3. **Calibrated ByteTrack Tracking:** Unified persistent tracking with tuned buffers for occlusions and fast motion.
4. **Dynamic Floor-Clamped HOI Projection (The Roll Fix):** Sustains bounding boxes during non-linear rotational movements and tilts by anchoring top edges to wrists and clamping bottom edges to dynamic ankle floor planes.
5. **Velocity-Delta Uncoupling (The Drop Fix):** Evaluates downward separation velocity to immediately detach box ownership on releases/throws and project gravitational free-fall trajectories.
6. **Ghost Box & Scale Filtering:** Suppresses ID proliferation and filters out giant false positives (bedframes/furniture).
7. **Downstream Data Contract Export:** Generates standardized JSON (`*_tracking_results.json`) consumed by Person B (Risk Engine) and Person C (AI Assistant).

---

## 2. Directory Structure

```
d:/godrej/
├── official_videos/             # Official Godrej warehouse challenge MP4s
├── weights/
│   └── box_11s.pt              # Supervised YOLO11 box weights
├── yolov8n-pose.pt             # YOLOv8 pose weights
├── custom_bytetrack.yaml       # Calibrated ByteTrack tracker parameters
├── pipeline_person_a.py        # Core production perception pipeline
├── batch_run_all_videos.py     # Batch runner across all official videos
├── run_live_webcam.py          # Real-time live camera demo runner
├── RUN_LIVE_CAMERA.bat         # Double-click launcher for webcam
├── DATA_CONTRACT.md            # Technical JSON schema contract for Person B & C
├── PRD.md                      # Product Requirements Document & 10-Scenario Matrix
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git clean configuration
```

---

## 3. Quick Start & Usage

### 3.1 Installation
```bash
pip install -r requirements.txt
```

### 3.2 Process a Single Video
```bash
python pipeline_person_a.py --input "official_videos/Rolling and dropping carton.mp4" --output_dir "outputs_person_a"
```
* Generates:
  * Annotated Video: `outputs_person_a/Rolling and dropping carton_perception.mp4`
  * Contract JSON: `outputs_person_a/Rolling and dropping carton_tracking_results.json`

### 3.3 Batch Process All Official Videos
```bash
python batch_run_all_videos.py
```
* Processes all MP4 files in `official_videos/` and saves contracts in `outputs_person_a/`.

### 3.4 Run Real-Time Live Webcam
```bash
python run_live_webcam.py --camera_id 0
```
*(Or double click `RUN_LIVE_CAMERA.bat` on Windows).*

---

## 4. Deliverable Verification Status

| Deliverable | Status | Location / Details |
| :--- | :--- | :--- |
| **Supervised Box Perception** | ✅ Verified | `weights/box_11s.pt` with confidence filtering |
| **Worker Skeleton Tracking** | ✅ Verified | 17 COCO keypoints tracked per person |
| **Dynamic Floor-Clamping (Roll Fix)** | ✅ Verified | Tested on `Rolling and dropping carton.mp4` |
| **Velocity Uncoupling (Drop Fix)** | ✅ Verified | Tested on `WIN_20260908_14_39_18_Pro.mp4` (throw/drop) |
| **Ghost Box & Scale Suppression** | ✅ Verified | Strict 1–2 box tracking; eliminated rogue `9000+` IDs |
| **Data Contract for Person B** | ✅ Verified | Full specification documented in `DATA_CONTRACT.md` |
| **Resolution-Invariant BBoxes** | ✅ Verified | Includes both `bbox` (pixel) and `bbox_normalized` ($[0, 1]$) |

---

## 5. Handover to Teammates

* **To Person B (Behavior & Risk Engine):** Read [`DATA_CONTRACT.md`](DATA_CONTRACT.md). Ingest `outputs_person_a/<video>_tracking_results.json` to calculate kinematics, rolling windows, and output `warehouse_events.json`.
* **To Person C (LLM Assistant):** Review the event schema in [`PRD.md`](PRD.md) Section 7.2 to construct the zero-hallucination system prompt for supervisor queries.
* **To Person D (Stitch Dashboard):** Use `*_perception.mp4` for video playback and UI mockups.
