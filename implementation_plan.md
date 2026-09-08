# Execution Protocol — STRICT

You MUST follow this protocol for every implementation run.

## Rule 1 — One Step Per Run
You may implement EXACTLY ONE step from the Implementation Plan per run.
After completing that step, you MUST STOP.
You are NOT allowed to:
- start the next step
- partially implement the next step
- modify files in preparation for the next step
- refactor code that belongs to a later step
- "finish the remaining work" if it happens to be easy
- combine multiple plan steps into one implementation
Even if the next step is obvious or trivial, STOP.

## Rule 2 — Determine the Current Step
Before making any changes:
1. Read the Implementation Plan.
2. Identify the FIRST incomplete step.
3. Treat that as the ONLY step you are authorized to implement.
4. Do not work on any other step.

## Rule 3 — Explain Before Implementing
Before modifying anything, provide:
### Step N — <name>
**Thought process**
- What this step accomplishes.
- Why this step is needed now.
- The approach you will take.

## Rule 4 — Implement
Implement ONLY Step N.

## Rule 5 — Verify
After implementation:
**Outcome**
- List exactly what was changed for Step N.
**Verification**
- Explain how Step N was tested or verified.
- Report any problems encountered.

## Rule 6 — HARD STOP
Once Step N is complete: STOP. Wait for the next run/user instruction.

---

# Implementation Plan — Person A: Video Intelligence & Hybrid Perception Pipeline

**Team Role:** Person A (Perception, Skeletal Pose, Tracking & Video Intelligence)  
**Target Hackathon:** Godrej Enterprises Group Hackathon @ graVITas 2026 (VIT Vellore)  
**Deadline:** 10th September 2026

---

## 1. The Core Architectural Strategy (Hybrid Perception Model)

Rather than forcing a single generic detector to guess complex industrial actions from raw bounding boxes, Person A uses a **Hybrid Multi-Modal Pipeline**:

### Model 1: Operator Skeletal Pose Expert (`yolov8n-pose.pt`)
* **Role:** Detects operators and extracts 17 continuous skeletal keypoints at 30+ FPS.
* **Keypoints Monitored:**
  * `Ankles & Feet (Pts 15, 16):` Detects stepping or standing on cartons.
  * `Wrists & Hands (Pts 9, 10):` Detects manual lifting points, strap pulling, and handholds.
  * `Spine & Hips (Pts 11, 12):` Computes carrying posture and rapid forward lurches.

### Model 2: Product & Logistics Entity Expert (`yolov8s-world.pt`)
* **Role:** Detects physical warehouse entities using open-vocabulary semantic prompting.
* **Target Classes:** `["shipping package", "corrugated box", "furniture", "mattress", "pallet", "trolley"]`.
* **Solves:** Accurately identifies brown cartons, plastic-wrapped mattresses, white KD flatpacks, and cupboards without false-positive geometric triggers on walls/doors.

### Model 3: The Spatial-Temporal Fusion Engine
1. **Handling Anchor (PRD Line 24):** Only activates tracking when an operator's hands/body overlap a product. Static inventory on shelves is 100% suppressed.
2. **Stepping Detection:** Fires when Foot/Ankle keypoint coordinates penetrate the upper boundary of a box:
   $$\text{box.x1} \le \text{foot.x} \le \text{box.x2} \quad \text{and} \quad \text{box.y1} \le \text{foot.y} \le \text{box.y1} + 0.35 \times \text{height}$$
3. **Orientation Check:** Flags vertical items placed horizontally via bounding box aspect ratio:
   $$\text{Aspect Ratio} = \frac{\text{Width}}{\text{Height}} > 1.4 \implies \text{HORIZONTAL MISORIENTATION}$$
4. **Improper Stacking:** Detects inverted pyramids (larger box on smaller base):
   $$\text{Width}_{\text{top}} > 1.25 \times \text{Width}_{\text{bottom}} \implies \text{IMPROPER STACKING (HEAVY ON TOP)}$$
5. **Drop & Drag Kinematics:** Computes vertical velocity vector $(v_y)$ and floor-level horizontal velocity $(v_x)$.
6. **Predictive Near-Miss (The USP):** Continuously scores risk *mid-transit* before impact occurs.

---

## 2. Master Testing & Verification Checklist

- [x] Phase 1: Benchmark Approach A (YOLO-World) vs Approach B (Dual-Inference).
- [x] Phase 1: Test multi-video speed, latency, and clutter metrics across official clips.
- [ ] Phase 2: Verify Hybrid Pipeline on 2 full-length official Godrej videos:
  - [ ] Video 1: `Rolling and dropping carton.mp4` (Full drop & roll sequence).
  - [ ] Video 2: `Stepping on cartons...mp4` (Full stepping, horizontal orientation & stacking sequence).
- [ ] Phase 3: Persistent ByteTrack trajectory smoothing & normalized tracking export.
- [ ] Phase 4: Output Contract generation (`tracking_results.json`) for Person B.
- [ ] Phase 5: Live camera stream runner (`source=0` webcam / RTSP).
- [ ] Phase 6: Batch rendering of all 7 official Godrej demo clips for Person D's dashboard and deck.

---

## 3. Step-by-Step Implementation Roadmap

### Step 3.0: Hybrid Pipeline Build & 2-Video Full Verification [CURRENT STEP]
* Build `hybrid_pipeline.py` fusing `yolov8n-pose` + `yolov8s-world` + geometric rules.
* Render full-length verification videos for:
  1. `Rolling and dropping carton.mp4`
  2. `Stepping on cartons, vertical product kept horizontally, heavy product kept on top.mp4`
* Inspect and confirm visual clarity, foot keypoint tracking, and behavior triggers.

### Step 3.1: Persistent ByteTrack Integration & Trajectory Kinematics
* Attach persistent `track_id` across frames without ID flickering.
* Compute continuous frame-by-frame velocity vectors `(vx, vy)` and height estimates.

### Step 3.2: Person A Data Contract Exporter (`tracking_results.json`)
* Lock the exact schema for Person B: frame index, timestamp, `track_id`, class label, normalized coordinates, and velocities.
* Deliver to Person B to unblock behavior rule engine.

### Step 3.3: Live Real-Time Feed Runner (Webcam & RTSP Stream)
* Build `live_camera_feed.py` supporting `cv2.VideoCapture(0)` (webcam) and RTSP IP streams.
* Verify smooth real-time operation on CPU with live HUD overlays.

### Step 3.4: Batch Process All 7 Official Godrej Clips
* Process and render all 7 official videos with the final hybrid system for the demo video submission.
