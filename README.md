# ImpactZero // AI Video Intelligence for Warehouse Handling
### An Edge-First AI Field Intelligence Platform for Safer, Damage-Free Operations
**Godrej Enterprises Group Hackathon @ graVITas 2026 | Team Perceptron**

---

## 1. Executive Summary & Vision

Traditional warehouse CCTV systems are fundamentally **retrospective** — they record what happened and force supervisors to conduct post-mortem investigations after products have already sustained costly damage. 

**ImpactZero** transforms conventional warehouse cameras into a proactive **AI Field Intelligence Assistant**. By fusing high-frame-rate computer vision, worker pose estimation, and temporal kinematics, ImpactZero continuously monitors loading and unloading operations to detect unsafe material-handling behaviors **while they are still preventable**.

### The Core Differentiator: Predictive Near-Miss Early Warning
Most automated monitoring systems only flag an incident *after* impact (e.g., once a carton has struck the concrete). ImpactZero's kinematics engine evaluates worker-object interaction, carry heights, velocity deltas, and proximity to drop hazards in real time. It raises visual and audible bay warnings **before** drop impact or structural rupture occurs, giving floor operators the critical window needed to intervene and prevent damage.

---

## 2. End-to-End System Architecture

ImpactZero is designed on a modular **Edge-to-Insight** architecture consisting of four integrated tiers:

```
+───────────────────────────────────────────────────────────────────────────────────────────+
│                                  TIER 1: EDGE PERCEPTION                                  │
│  RTSP / CCTV / Webcam Streams ➔ Supervised YOLO11s (Cartons) ➔ YOLOv8-Pose (17 Joints)     │
│  ➔ Calibrated ByteTrack ➔ Dynamic Floor-Plane Clamping & Velocity-Delta Uncoupling        │
+─────────────────────────────────────────────┬─────────────────────────────────────────────+
                                              ▼
+───────────────────────────────────────────────────────────────────────────────────────────+
│                            TIER 2: TEMPORAL KINEMATICS ENGINE                             │
│  15-Frame Rolling Spatio-Temporal Window ➔ Hand-Object Interaction (HOI) Modeling          │
│  ➔ Continuous Near-Miss Risk Scoring ➔ Deterministic 10-Scenario Safety Rules             │
+─────────────────────────────────────────────┬─────────────────────────────────────────────+
                                              ▼
+───────────────────────────────────────────────────────────────────────────────────────────+
│                         TIER 3: AUDIT DATA LEDGER & PRODUCTION HUD                        │
│  Standardized JSON Schema (warehouse_events.json) ➔ Local FastAPI Server (30 FPS Edge)    │
│  ➔ 3-Second Evidentiary Replay Snippets ➔ Multi-Page Operational Dashboard (Stitch/HTML5) │
+─────────────────────────────────────────────▲─────────────────────────────────────────────+
                                              │ (Bidirectional Grounded Queries)
+─────────────────────────────────────────────▼─────────────────────────────────────────────+
│                       TIER 4: GROUNDED CONVERSATIONAL AI COPILOT                          │
│  Multi-Tier Resilient Failover: Gemini 3.6 Flash ➔ Groq LLaMA ➔ Offline Local Engine       │
│  Zero-Hallucination Event Ledger Grounding ➔ Automated Shift Summaries & Toolbox SOPs     │
+───────────────────────────────────────────────────────────────────────────────────────────+
```

### Core Architecture Components:
1. **Perception & Tracking:** Custom-calibrated YOLO11s package detector (`weights/box_11s.pt`) paired with YOLOv8-Pose (`yolov8n-pose.pt`) extracting 17 skeletal keypoints per operator. Calibrated ByteTrack maintains identity persistence through occlusions.
2. **Floor-Clamped HOI Projection & Drop Fix:** Evaluates downward velocity deltas to immediately detach box ownership during throws or drops while projecting dynamic floor planes anchored to worker ankle keypoints.
3. **Deterministic Kinematics Engine:** Evaluates velocity vectors ($v_x, v_y, a_y$) across rolling windows to detect non-linear rotational movement, drags, and slips.
4. **Grounded AI Supervisor Assistant:** A conversational assistant that consumes structured event telemetry (`warehouse_events.json`). Every response cites verified event IDs (`EVT-...`), timestamps, and locations with zero hallucinations.
5. **Edge Dashboard:** Multi-view operational interface featuring live camera feeds, interactive incident scrubbers, risk distribution heatmaps, and shift trend analytics.

---

## 3. The 10 Demonstrated Handling Scenarios

ImpactZero covers 10 predefined handling scenarios directly derived from Godrej’s material-handling guidelines:

| # | Handling Behavior | Trigger Logic & Kinematic Telemetry | Classification |
|---|---|---|---|
| **1** | **Carton / KD Floor Dragging** | Sustained horizontal movement on floor plane without lifting or trolley equipment. | High Risk |
| **2** | **High-Impact Drop (> 0.8m)** | Downward gravitational velocity spike followed by stationary state on floor. | Critical Risk |
| **3** | **Low-Height Drop / Slip (< 0.5m)** | Sudden vertical drop from table, conveyor, or dock leveler edge. | Medium Risk |
| **4** | **Predictive Near-Miss (USP)** | Rapid carry at unsafe elevation heading toward an edge or unstable stack (alert latched pre-impact). | Critical Risk |
| **5** | **Carton Throwing / Sliding** | High horizontal velocity trajectory across staging floor or pallet surface. | High Risk |
| **6** | **Improper Stacking** | Inverted pyramid stacking (heavier or larger footprint carton placed atop smaller package). | High Risk |
| **7** | **Stepping / Standing on Carton** | Operator foot keypoints overlapping top bounding surface of packaging. | Critical Risk |
| **8** | **Strap Pulling / Lifting** | Lifting heavy packages by packaging straps rather than designated lifting points. | Medium Risk |
| **9** | **Unstable Stack Overhang** | Severe box tilt and overhang exceeding safe pallet boundaries. | High Risk |
| **10** | **Controlled Handling Benchmark** | Compliant ergonomic lift, two-person team carry, or pallet truck transport. | Nominal (Safe) |

---

## 4. Responsible AI & Data Governance

ImpactZero is built as an ergonomic operational guardian, **not an employee surveillance tool**:

* **Privacy by Design (Worker Anonymization):** The system features on-device worker identity masking (`mask=true`). Human operators are abstracted into neutral 17-keypoint ergonomic skeletons, with facial features blurred locally before storage.
* **Zero Worker Scorekeeping:** Incident logs track loading bays, process bottlenecks, and equipment availability — never individual worker penalty leaderboards. The focus is on staging tools (e.g., adding hydraulic trolleys at Bay 02) rather than punitive write-ups.
* **The Confidence Ladder:** Events progress strictly through `Observed Behaviour` $\rightarrow$ `Potential Risk` $\rightarrow$ `Confirmed Damage`. The system never claims product damage occurred without verified physical evidence.
* **Full Explainability (XAI):** Every alert includes exact telemetry triggers (height, speed, duration) and a 3-second verifiable clip for fair human-in-the-loop review.
* **Local Data Minimization:** 100% of video inference runs on-prem at the edge. Raw continuous CCTV footage is never uploaded to external clouds.

---

## 5. Technology Stack

* **Computer Vision & Tracking:** Ultralytics YOLO11s, YOLOv8-Pose, ByteTrack, OpenCV (`cv2`)
* **Kinematics & Analytics:** NumPy, SciPy, Python 3.10+
* **Backend & Streaming:** FastAPI, Uvicorn, Asynchronous WebSockets / MJPEG
* **Conversational AI:** Google GenAI (Gemini 3.6 Flash), Groq API, Multi-Tier Resilient Failover Client
* **Frontend UI:** Vanilla HTML5, CSS3, JavaScript (Google Stitch-inspired industrial HUD)
* **Storage & Contract:** Standardized JSON Event Ledger (`data/warehouse_events.json`)

---

## 6. Repository Structure

```
d:/godrej/
├── Godrej/                         # Front-end dashboard views
│   ├── overview.html               # Executive HUD & shift KPIs
│   ├── intelligence.html          # Live camera feed & real-time inference overlay
│   ├── incidents.html             # Incident investigation & 3s evidence replay
│   ├── behaviour.html             # Behavior frequency & bay breakdown
│   ├── prevention.html            # Corrective action & SOP coaching center
│   └── assistance.html            # Grounded Conversational AI Assistant
├── assistant/                      # Conversational Copilot & RAG Engine
│   ├── event_loader.py            # Validated EventStore data loader
│   ├── llm_client.py              # Multi-tier failover client (Gemini -> Groq -> Local)
│   ├── queries.py                 # Grounded supervisor query handlers & tool calling
│   ├── schemas.py                 # Pydantic schemas for event contracts
│   └── system_prompt.py           # Zero-hallucination system instructions
├── official_videos/                # Official Godrej challenge video files
├── weights/                        # Neural network model weights
│   └── box_11s.pt                 # Supervised YOLO11 package weights
├── yolov8n-pose.pt                 # YOLOv8 pose keypoint weights
├── data/                           # Standardized warehouse event logs & benchmarks
│   ├── warehouse_events.json      # Verified physical event ledger
│   └── ground_truth.json          # Evaluation ground truth annotations
├── api.py                          # Unified FastAPI server & live video streamer
├── run_dashboard.py                # 1-Click launcher for dashboard & backend
├── run_live_webcam.py              # Standalone live webcam demo runner
├── RUN_DASHBOARD.bat               # Windows batch launcher for dashboard
├── RUN_LIVE_CAMERA.bat             # Windows batch launcher for live camera
├── pipeline_person_a.py           # Video perception & ByteTrack tracking runner
├── pipeline_person_b.py           # Kinematics engine & incident rule processor
├── batch_run_all_videos.py        # Automated batch processor across all input videos
├── requirements.txt                # Python environment dependencies
└── README.md                       # Master solution documentation
```

---

## 7. Quick Start & Execution Guide

### 7.1 Installation

Ensure Python 3.10 or higher is installed with CUDA support (optional for GPU acceleration):

```bash
git clone https://github.com/AJ-401/Warehouse-Perceptron.git
cd Warehouse-Perceptron
pip install -r requirements.txt
```

Set up your optional LLM API keys in `.env` (the system automatically falls back to the local engine if keys are omitted):

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.6-flash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

### 7.2 Launch the Full Dashboard & Backend (1-Click)

```bash
python run_dashboard.py
```
*(Or double-click `RUN_DASHBOARD.bat` on Windows).*  
This launches the FastAPI server and opens `http://localhost:8000/overview.html` in your default browser.

### 7.3 Run Real-Time Webcam Inference

```bash
python run_live_webcam.py --camera_id 0
```
*(Or double-click `RUN_LIVE_CAMERA.bat` on Windows).*

### 7.4 Batch Process Video Files

Process all video files in `official_videos/` and update the event ledger:

```bash
python batch_run_all_videos.py
```

---

## 8. Summary of Capabilities

* **Proactive Damage Prevention:** Replaces passive CCTV recording with pre-impact alerts.
* **100% Edge-Capable:** Real-time local inference without mandatory cloud connectivity.
* **Zero-Hallucination AI:** Conversational copilot grounded strictly in verified physical telemetry.
* **Responsible by Design:** Protects worker privacy through on-device anonymization and process-focused coaching.
