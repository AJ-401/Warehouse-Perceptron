import json
import os
from datetime import datetime

def parse_time(time_str: str) -> float:
    """Convert 'HH:MM:SS.mmm' string to seconds float."""
    try:
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1000000.0
    except ValueError:
        return 0.0

def load_json(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing file: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def evaluate(predictions_path: str, ground_truth_path: str, time_tolerance_sec: float = 2.0):
    print(f"Loading predictions from: {predictions_path}")
    print(f"Loading ground truth from: {ground_truth_path}")
    
    pred_data = load_json(predictions_path)
    gt_data = load_json(ground_truth_path)
    
    # In case predictions is wrapped in {"events": [...]}
    if isinstance(pred_data, dict) and "events" in pred_data:
        predictions = pred_data["events"]
    else:
        predictions = pred_data
        
    print(f"Found {len(predictions)} predicted events and {len(gt_data)} ground truth events.\n")
    
    tp = 0
    fn = 0
    
    # We will keep track of which predictions matched a GT to calculate FPs later
    matched_pred_ids = set()
    
    # 1. Match Ground Truth to Predictions (Find TP and FN)
    for gt in gt_data:
        gt_video = gt.get("video_id")
        gt_code = gt.get("behaviour_code")
        gt_time = parse_time(gt.get("timestamp_start", "00:00:00.000"))
        
        match_found = False
        for pred in predictions:
            pred_id = pred.get("event_id")
            if pred_id in matched_pred_ids:
                continue # Already matched this prediction to another GT
                
            if pred.get("video_id") == gt_video and pred.get("behaviour_code") == gt_code:
                pred_time = parse_time(pred.get("timestamp_start", "00:00:00.000"))
                
                if abs(pred_time - gt_time) <= time_tolerance_sec:
                    match_found = True
                    matched_pred_ids.add(pred_id)
                    break # Matched!
                    
        if match_found:
            tp += 1
        else:
            fn += 1
            
    # 2. Calculate False Positives (Predictions that didn't match any GT)
    # Since our ground_truth.json is tiny (mock data), there will be a LOT of FPs
    # from the 250 predictions.
    fp = len(predictions) - len(matched_pred_ids)
    
    print("-" * 30)
    print("EVALUATION METRICS")
    print("-" * 30)
    print(f"True Positives (TP) : {tp}")
    print(f"False Negatives (FN): {fn} (Missed detections)")
    print(f"False Positives (FP): {fp} (False alarms)")
    print("-" * 30)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"Precision : {precision:.2%}")
    print(f"Recall    : {recall:.2%}")
    print(f"F1 Score  : {f1_score:.2%}")
    print("-" * 30)
    print("Note: Because you are currently using a mock ground_truth.json containing only")
    print("a few events, the False Positives (and thus Precision) are artificially low.")
    print("Add the complete ground truth data to data/ground_truth.json to get real metrics.")

if __name__ == "__main__":
    PREDICTIONS_FILE = "outputs_person_b/warehouse_events.json"
    GROUND_TRUTH_FILE = "data/ground_truth.json"
    
    evaluate(PREDICTIONS_FILE, GROUND_TRUTH_FILE)
