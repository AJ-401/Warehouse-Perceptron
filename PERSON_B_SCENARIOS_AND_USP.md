# Person B Scenarios & The Predictive Near-Miss USP
### Godrej Video Intelligence for Warehouse Handling | graVITas 2026

---

## 1. Problem Statement Pain Points (The 9 Core Categories)

Directly from the Godrej Challenge Brief (*Line 11–20 & Line 178*):
> *"The system should observe loading and unloading activities, identify potentially damaging or unsafe behaviours, provide real-time alerts, and enable corrective intervention before damage occurs: **Damage Detection → Damage Prevention**."*

The 9 official operational pain points are:
1. **Dropping or Impact**
2. **Unsafe Movement of Material**
3. **Rough Handling during Loading / Unloading**
4. **Incorrect Stacking & Unstable Loading**
5. **Improper Use of Pallets or Handling Equipment**
6. **Potential Damage-Causing Operator Behaviour**
7. **Improper Handling & Misorientation**
8. **Overloading or Unstable Loading**
9. **Safe Handling Practice (Operating Benchmark / Control)**

---

## 2. The Core Differentiator (USP): Predictive Near-Miss Prevention

### 2.1 The Concept
* **Traditional CCTV / Competitor AI:** Post-mortem incident detection. A box hits the concrete floor and breaks $\rightarrow$ system logs "Product Damaged".
* **Our AI (The USP):** **Active Transit Risk Prediction.** The system continuously tracks the kinematics of the carry while the action is still in progress. If risky warning signs converge, an alert fires **1.2 to 2.0 seconds BEFORE** the impact occurs, enabling proactive operator intervention.

### 2.2 Mathematical Risk Formulation
During active carry (`state == 'HELD'` or interacting with a worker), Person B computes:
$$\text{Risk}_{\text{near-miss}} = w_1 \cdot \mathcal{S}(v_{\text{horiz}}) + w_2 \cdot \mathcal{S}(h_{\text{floor}}) + w_3 \cdot \mathcal{I}_{\text{hazard}}$$
Where:
* $v_{\text{horiz}}$: Horizontal velocity vector of package across rolling 15-frame window ($> 0.07$ normalized).
* $h_{\text{floor}}$: Vertical clearance between package bottom edge and floor plane ($> 1.0\text{m}$ normalized/metric elevation).
* $\mathcal{I}_{\text{hazard}}$: Proximity to unstable stacks, dock edges, or rapid worker deceleration.

When $\text{Risk}_{\text{near-miss}} \ge 0.70$, the system emits:
`[CRITICAL ALERT] UNSAFE MOVEMENT OF MATERIAL — PREDICTIVE NEAR-MISS (Probability: 84%)`

---

## 3. The 10+ Scenario Taxonomy Matrix

To satisfy Line 242 (*"demonstrate at least 10 predefined behaviours/scenarios"*), every event is categorized under an official Problem Statement Pain Point with a specific sub-behaviour:

| # | Official Problem Statement Category | Specific Behaviour (Sub-Type) | Behaviour Code | Severity | Description & Kinematic Trigger |
| :-: | :--- | :--- | :--- | :-: | :--- |
| **1** | **Dropping or Impact** | High-Impact Carton Drop (>0.8m) | `DROP_HIGH_IMPACT` | **Critical** | Sudden downward vertical velocity spike ($v_y \ge 1.8\text{ m/s}$) followed by stationary state on floor. |
| **2** | **Dropping or Impact** | Low-Height Drop / Slip (<0.5m) | `DROP_LOW_SLIP` | **High** | Slip off a pallet/table edge ($0.25\text{m} - 0.8\text{m}$ drop) during transfer. |
| **3** | **Unsafe Movement of Material** | **Predictive Near-Miss (USP)** | `NEAR_MISS_UNSAFE_CARRY` | **Critical** | Operator carrying carton rapidly at unsafe elevation ($> 1.0\text{m}$) before drop occurs. |
| **4** | **Unsafe Movement of Material** | Floor Dragging (No Equipment) | `UNSAFE_FLOOR_DRAG` | **High** | Sustained horizontal movement on floor ($> 0.03$ norm disp) without mechanical equipment. |
| **5** | **Rough Handling** | Carton Throwing / Sliding | `ROUGH_THROW_SLIDE` | **High / Crit** | Rapid horizontal trajectory across floor/pallet ($v_x > 1.2\text{ m/s}$) or operator throwing motion. |
| **6** | **Rough Handling** | Rolling Cartons | `ROUGH_CARTON_ROLLING` | **Medium** | Tumbling/rolling carton across floor without equipment or controlled lifting. |
| **7** | **Incorrect Stacking & Loading** | Inverted Pyramid Stacking | `STACK_INVERTED_PYRAMID` | **High** | Larger footprint or heavier packet stacked directly on top of a smaller base box. |
| **8** | **Incorrect Stacking & Loading** | Unstable Multi-Tier Stack Wobble | `STACK_UNSTABLE_WOBBLE` | **Critical** | Box tilt or center-of-mass jitter on resting stack exceeding safe stability threshold. |
| **9** | **Improper Use of Equipment** | Strap Lifting / Pulling | `EQUIPMENT_STRAP_LIFT` | **High** | Lifting heavy carton by packaging straps rather than designated bottom support points. |
| **10** | **Operator Risk Behaviour** | Stepping / Standing on Products | `OPERATOR_STEPPING_CARTON` | **Critical** | Operator foot/ankle keypoint bounding box overlapping top surface of package. |
| **11** | **Improper Handling** | Vertical Product Kept Horizontally | `IMPROPER_MISORIENTATION` | **Medium** | High aspect ratio item ($W / H > 1.4$) stored or moved horizontally against packaging guidelines. |
| **12** | **Safe Operating Standard** | Safe Handling Benchmark | `BENCHMARK_SAFE_HANDLING` | **Low (Info)** | Controlled carry speed ($< 0.04$ norm) and safe elevation — positive reinforcement control. |

---

## 4. Responsible AI: Dynamic Risk Scoring & Confidence Ladder

In strict compliance with **Problem Statement Lines 100–118**, the engine distinguishes between:
$$\text{Observed Behaviour} \longrightarrow \text{Potential Risk} \longrightarrow \text{Confirmed Damage}$$

### Confidence Stages:
1. **Observed Behaviour:** Physical kinematics detected (e.g., box sliding across floor at $0.8\text{ m/s}$).
2. **Potential Risk:** Action violates safety protocol, risk of damage elevated (e.g., Near-Miss carry at $1.4\text{m}$ height).
3. **Confirmed Damage:** High-impact shock ($> 0.8\text{m}$ drop or high-speed drop onto hard concrete surface).

---

## 5. Standard Event JSON Contract (`warehouse_events.json`)

```json
{
  "event_id": "EVT-20260909-0104",
  "video_id": "KD packets dragged, heavy box kept on other packets.mp4",
  "timestamp_start": "00:00:05.433",
  "timestamp_end": "00:00:07.100",
  "frame_range": [163, 213],
  "location_id": "Bay 01 - Inbound Unloading",
  "category": "Incorrect Stacking & Unstable Loading",
  "behaviour_type": "Inverted Pyramid Stacking",
  "behaviour_code": "STACK_INVERTED_PYRAMID",
  "risk_level": "High",
  "confidence_stage": "Potential Risk",
  "is_near_miss": false,
  "near_miss_probability": 0.0,
  "entities_involved": {
    "person_track_ids": [1025],
    "package_track_ids": [1463, 1001],
    "equipment_detected": null
  },
  "telemetry": {
    "top_box_area_norm": 0.0842,
    "bottom_box_area_norm": 0.0386,
    "area_ratio": 2.18,
    "is_repeated_behaviour": true,
    "occurrence_count_in_run": 2
  },
  "reason": "Carton #1463 (larger footprint, ratio 2.18x) stacked directly on top of smaller carton #1001 — inverted pyramid stack.",
  "recommended_action": "Re-stack with heavier/larger packages at the bottom base to prevent crushing and tipping.",
  "evidence": {
    "snapshot_url": "evidence/snapshots/KD_packets_dragged_163.jpg",
    "clip_url": "evidence/clips/KD_packets_dragged_163_213.mp4"
  }
}
```
