# PRD — AI Video Intelligence for Warehouse Handling
### Godrej Enterprises Group Hackathon @ graVITas 2026 | Team of 4 | Deadline: 10th September 2026

---

## 1. What We're Building (in simple words)

A system that watches warehouse loading/unloading video and catches **bad handling — before damage happens**, not after.

Normal CCTV just records. Our system watches, understands *what someone is doing* (not just "there's a person and a box"), figures out if it looks risky, warns someone in time, and later tells supervisors what went wrong and how to fix it — in plain English, through a chat-like assistant.

**Our one core differentiator (USP): Predictive Near-Miss Warning.**
Most solutions only flag an event *after* it's fully happened (box already dropped). We continuously score risk *while the action is still in progress* (box being carried too fast, too high, near an unstable stack, no trolley used) and raise a warning **before** the drop occurs — so someone can actually intervene in time. This directly matches the brief's own stated goal: *"Damage Detection → Damage Prevention."*

---

## 2. Requirement Checklist — how we satisfy every part of the problem statement

| Requirement from PS doc | How we cover it (plain terms) |
|---|---|
| Ingest live/recorded video | We process uploaded/staged video files frame by frame using OpenCV |
| Detect people, products, pallets, equipment | Pretrained YOLO model spots these objects in every frame — no training needed, it already knows how to find people/boxes |
| Track objects across frames | ByteTrack (built into YOLO's tracking mode) gives each object a persistent ID so we know "this is the same box across 100 frames," not a new box each time |
| Identify loading/unloading activities | When a person's box overlaps a product's box and the product starts moving, we log that as "handling started" |
| Detect dropping, dragging, throwing, improper stacking | We watch each tracked object's movement pattern (speed, height, direction) over a short time window and match it against simple rules — e.g. sudden vertical speed = drop, floor-level sideways movement = drag |
| Detect sequence of actions, not single frames | Every behaviour decision looks at a short *rolling history* of the object's movement, not one snapshot — this is literally sequence reasoning |
| Assign risk level, identify high-risk events | Each flagged event gets scored Low/Medium/High/Critical using a simple weighted formula (behaviour type + estimated drop height + how often it repeats) |
| Generate timestamp + evidence | Every flagged event auto-saves its time, a snapshot image, and a short clip |
| AI assistant that explains, answers questions, recommends actions, summarizes shifts | We feed all detected events into an LLM (like Claude) with strict instructions to only answer using real logged events — never make things up |
| Visual alerts + dashboard (highlight events, risk category, timeline, summaries, trends) | Built in Stitch — video with highlighted boxes, a color-coded timeline, daily summary cards, trend charts |
| Identify recurring behaviours, recommend training, track improvement | We group events by type/location and let the LLM turn patterns into plain-language coaching notes ("Operator dragged 3x today — recommend trolley training") |
| Minimum 4-5 behaviours demonstrated (10 in final prototype) | We pick 5-6 behaviours from their list, stage clean video of each multiple times to hit 10+ scenarios |
| Controlled/simulated warehouse allowed | We build a mini staged setup with boxes, a table/trolley, and ourselves as "operators" — explicitly allowed in the brief |
| Responsible AI (privacy, consent, no punitive auto-decisions, false positives, explainability) | We only use our own staged footage (no real employee data = no privacy issue), system only *recommends*, never auto-punishes, and every flagged event shows *why* it was flagged |
| Success metrics beyond accuracy | We compute real precision/recall/false-positive rate from our own staged clips (we know the ground truth since we staged them), report operational counts honestly scoped to our demo, and clearly label business-impact numbers as *projections*, not measured facts |
| Working prototype | Full pipeline: video in → detection → risk classification → explanation out |
| 5-6 slide deck | Structured exactly to their required slide order (see below) |
| Team of 3-5 | We are 4 |

---

## 3. The USP, explained simply (again, so it's crystal clear)

Imagine a smoke detector. A **normal system** is a smoke detector that only beeps once the fire has already started — the box has already hit the ground.

**Our system** is more like someone watching you carry a hot pan too fast near the stairs and going "hey, slow down" *before* you trip — because the warning signs were already there (moving fast + carrying something high + near a hazard), even though nothing bad has happened yet.

**How we build this with zero extra tech:** We're already tracking each box's speed and position every frame for behaviour detection anyway. Instead of only checking "did it hit the floor" at the very end, we also check a few warning signs *while it's still being carried*:
- Is it moving unusually fast?
- Is it being held at a risky height?
- Is a trolley/pallet nearby, or is it just being hand-carried?
- Is it near an already-unstable stack?

If enough of these are true mid-action, we raise a live warning **before** the drop/impact happens — same data, just checked continuously instead of only at the end.

---

## 4. Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Object detection | YOLOv8/v11 (Ultralytics, pretrained) | No training needed; detects person/box/pallet out of the box |
| Object tracking | ByteTrack (built into Ultralytics) | Gives persistent IDs across frames, minimal setup |
| Video handling | OpenCV | Frame extraction, drawing overlay boxes |
| Behaviour + risk logic | Plain Python | Rule-based, fully explainable, no ML training required |
| Backend/API | FastAPI | Simple, fast to wire detection output to frontend and LLM |
| AI Assistant | Claude or GPT API | Answers only from our logged events (grounded, no hallucination) |
| Event storage | JSON files or SQLite | No cloud infra needed for a 4-day build |
| Dashboard | **Stitch** (Google's AI UI builder — prompt-to-frontend) | Fastest way to vibecode a polished dashboard without hand-coding every screen; export and wire up to backend via API calls |

---

## 5. Work Split — 4 People, No Overlap Chaos

**The one rule that prevents chaos:** everyone agrees on the **event data format** (the JSON structure every detected event follows) by the end of Day 1. Once that's locked, all four people can work in parallel without blocking each other, because each person just needs to produce or consume that one shared format.

**Person A — Detection & Tracking**
- Set up YOLO, run it on staged footage
- Get clean tracked-object data out (object ID, type, position, size, per frame)
- This is the foundation everyone else needs — prioritize finishing this first, even a rough version, by end of Day 1

**Person B — Behaviour Rules + Risk Scoring + the USP**
- Takes Person A's tracked data
- Writes the rule logic (drop/drag/stack/step detection) using short rolling windows of movement
- Builds the risk scorer (Low/Med/High/Critical) and the confidence ladder (Observed → Potential Risk → Confirmed Damage)
- Builds the near-miss / predictive scorer (the USP) — same data, continuous check instead of end-only check
- Outputs everything as events in the agreed JSON format

**Person C — AI Assistant + Event Log + Metrics**
- Defines the final event JSON schema (in coordination with A & B, locked Day 1)
- Builds the LLM layer: grounded Q&A, shift summaries, coaching notes from recurring patterns
- Also owns computing real precision/recall/false-positive rate later (since this person already lives in the event data)

**Person D — Dashboard (Stitch) + Demo + Deck**
- Builds the dashboard in Stitch: video+overlay view, color-coded timeline, risk gauge (for near-miss warnings), summary cards, trend/heatmap view, chat-style assistant interface
- Needs mock/sample event JSON from Person C early (Day 1 end) so this isn't blocked waiting on real detection
- Also owns filming the staged scenarios (can pull in A/B/C briefly for filming since it needs "operators" and helpers), building the final slide deck, and editing the demo video

---

## 6. Official Video Link & The 10 Demonstration Scenarios

### 6.1 Official Godrej Input Videos
* **Google Drive Link:** [Official Godrej Warehouse Footage](https://drive.google.com/drive/folders/1MG90LJowfSZ2qz5woDyarHdzskbCRLzP?usp=sharing)
* **Strategy:** Use at least 1-2 clips from the official drive to demonstrate pipeline generalization, complemented by controlled staged recordings for the full test matrix.

### 6.2 The 10 Required Scenarios Matrix (Directly from Godrej Bad Practices Table)
To strictly satisfy submission requirement Line 242 (*"demonstrate at least 10 predefined behaviours/scenarios"*), the prototype and event log will evaluate:
1. **Carton / KD Packet Floor Dragging:** Moving carton horizontally along the floor without lifting or using a trolley.
2. **High-Impact Carton Drop (> 0.8m):** Vertical velocity spike followed by stationary state on floor.
3. **Low-Height Drop / Carton Slip (< 0.5m):** Slip off a table or edge during transfer.
4. **Predictive Near-Miss (USP):** Operator carrying carton rapidly at unsafe height towards dock edge or unstable stack (alert fired *before* drop).
5. **Carton Throwing / Sliding:** Fast horizontal trajectory across floor or pallet surface.
6. **Improper Stacking (Inverted Pyramid):** Smaller footprint / lighter packet placed below a larger / heavier carton.
7. **Stepping / Standing on Carton:** Operator foot bounding box overlapping top surface of a carton.
8. **Strap Lifting / Pulling:** Lifting a heavy package by packaging straps instead of lifting points.
9. **Unstable Multi-Tier Stack Wobble:** Box tilt / overhang exceeding safe pallet boundary.
10. **Safe Handling Benchmark (Control):** Proper movement using pallet truck / trolley and controlled 2-person lift.

---

## 7. Data Schemas (The Locked Team Contracts)

### 7.1 Person A Output: Frame-Level Tracking JSON (`tracking_results.json`)
Produced frame-by-frame by YOLOv8 + ByteTrack:
```json
{
  "video_id": "bay02_unloading_clip01.mp4",
  "fps": 30,
  "total_frames": 450,
  "resolution": [1920, 1080],
  "frames": [
    {
      "frame_idx": 45,
      "timestamp_sec": 1.50,
      "tracks": [
        {
          "track_id": 1,
          "class": "person",
          "confidence": 0.92,
          "bbox": [540, 210, 720, 680],
          "bbox_normalized": [0.281, 0.194, 0.375, 0.630]
        },
        {
          "track_id": 2,
          "class": "package",
          "confidence": 0.88,
          "bbox": [600, 450, 750, 620],
          "bbox_normalized": [0.312, 0.416, 0.390, 0.574]
        }
      ]
    }
  ]
}
```

### 7.2 Person B Output: Behavior & Incident Event JSON (`warehouse_events.json`)
Consumed by Person C (LLM Assistant) and Person D (Dashboard):
```json
{
  "events": [
    {
      "event_id": "EVT-20260908-0104",
      "video_id": "bay02_unloading_clip01.mp4",
      "timestamp_start": "00:01:24.500",
      "timestamp_end": "00:01:28.000",
      "frame_range": [2535, 2640],
      "location_id": "Bay 02 - Unloading Dock",
      "behaviour_type": "Carton / KD Floor Dragging",
      "behaviour_code": "DRAG_NO_EQUIPMENT",
      "risk_level": "High",
      "confidence_stage": "Potential Risk",
      "is_near_miss": false,
      "near_miss_probability": 0.0,
      "entities_involved": {
        "person_track_ids": [1],
        "package_track_ids": [2],
        "equipment_detected": null
      },
      "telemetry": {
        "horizontal_distance_m": 3.8,
        "floor_contact_duration_sec": 3.5,
        "drop_height_m": 0.0,
        "is_repeated_behaviour": true
      },
      "reason": "Operator #1 dragged KD Carton #2 across warehouse floor for 3.8m instead of using an available pallet truck.",
      "recommended_action": "Inspect bottom surface of carton for abrasion damage. Recommend operator trolley refresh training.",
      "evidence": {
        "snapshot_url": "/evidence/snapshots/EVT-0104.jpg",
        "clip_url": "/evidence/clips/EVT-0104.mp4"
      }
    },
    {
      "event_id": "EVT-20260908-0105",
      "video_id": "bay02_unloading_clip01.mp4",
      "timestamp_start": "00:02:10.200",
      "timestamp_end": "00:02:12.800",
      "frame_range": [3906, 3984],
      "location_id": "Bay 02 - Unloading Dock",
      "behaviour_type": "Predictive Near-Miss: Rapid Unsafe Carry",
      "behaviour_code": "NEAR_MISS_UNSAFE_CARRY",
      "risk_level": "Critical",
      "confidence_stage": "Potential Risk",
      "is_near_miss": true,
      "near_miss_probability": 0.84,
      "entities_involved": {
        "person_track_ids": [3],
        "package_track_ids": [5],
        "equipment_detected": null
      },
      "telemetry": {
        "speed_px_per_sec": 240.5,
        "carry_height_m": 1.35,
        "proximity_to_hazard": "High (unstable stack edge)",
        "is_repeated_behaviour": false
      },
      "reason": "Carton #5 carried at unsafe elevation (1.35m) at excessive speed heading towards unstable stack edge. Warning triggered 1.4s before potential impact.",
      "recommended_action": "Immediate dock supervisor audio alert. Maintain controlled carry speed below 1.0 m/s.",
      "evidence": {
        "snapshot_url": "/evidence/snapshots/EVT-0105.jpg",
        "clip_url": "/evidence/clips/EVT-0105.mp4"
      }
    }
  ],
  "shift_summary": {
    "total_events": 12,
    "near_misses_prevented": 8,
    "high_critical_risks": 4,
    "estimated_damage_avoided_inr": 185000,
    "most_frequent_risk": "Carton / KD Floor Dragging"
  }
}
```

---

## 8. Implementation Steps

### 8.1 Detection & Tracking (Person A)
1. Install Ultralytics YOLO (`pip install ultralytics opencv-python`).
2. Run YOLO's built-in tracking mode (`model.track(source=video, tracker='bytetrack.yaml')`).
3. Class mapping: Person (`class 0`), Package / Carton / Equipment (utilize COCO classes like suitcase/backpack or specialized lightweight package weights / heuristic box clustering).
4. Export: `tracking_results.json` and annotated debug video with tracking IDs and bounding boxes.

### 8.2 Behaviour Rules & Risk Engine (Person B)
1. Ingest `tracking_results.json`.
2. Compute kinematics: 15-frame rolling buffer for velocity, trajectory, floor distance, and bounding box overlap.
3. Fire detection triggers for the 10 defined scenarios.
4. Calculate Near-Miss probability score continuously during active transit.
5. Export `warehouse_events.json`.

### 8.3 Grounded AI Assistant (Person C)
1. Ingest `warehouse_events.json`.
2. System prompt enforcing zero-hallucination, strictly citing `event_id`, timestamps, and Godrej terminology.
3. Provide supervisor query endpoints (Shift summaries, incident rationale, Bay risk comparison).
4. Compute accuracy/precision/recall matrix against staged ground truth.

### 8.4 Dashboard & Presentation (Person D)
1. Visual alert dashboard (Risk Gauge, Video Player with sync timeline, Incident Replay, Chat Assistant).
2. Film and edit 3-5 high-impact scenarios into demo video.
3. Compile 5-6 slide deck matching official specifications.
4. Conduct 5-minute User Validation interview with VIT campus store / courier hub supervisor for Slide 5.

---

## 9. Slide Deck Structure (Mandatory 5-6 Slides)

1. **Slide 1: Solution & Team** — App name, team name, members, one-line value proposition (*Predictive Damage Prevention*).
2. **Slide 2: Problem, Solution & User Journey** — Warehouse Activity → AI Perception → Sequence Understanding → Near-Miss Alert → Proactive Intervention → Zero Damage.
3. **Slide 3: Technical Architecture & Stack** — YOLOv8 + ByteTrack + Kinematic Rule Engine + Grounded LLM + FastAPI + Interactive Dashboard.
4. **Slide 4: Prototype Screenshots & Demo Video** — Live risk gauge, incident replay, sequence detection, side-by-side comparative demo.
5. **Slide 5: Impact, Metrics & User Validation** — Precision/Recall, damage financial projections, and documented feedback from campus logistics/store in-charge.

---

## 10. Timeline (7th Sept – 10th Sept)
- **Day 1 (7th Night / 8th Morning):** Person A environment setup, tracking pipeline execution, and `tracking_results.json` generation.
- **Day 2 (8th):** Person B behavior rule implementation; Person C LLM integration; Person D dashboard layout against mock schema.
- **Day 3 (9th):** End-to-end pipeline integration; film staged clips; campus user validation interview; slide deck draft.
- **Day 4 (10th):** Record demo video, finalize PPT deck, and submit before the deadline.
