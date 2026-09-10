"""
test_accuracy.py
================
Comprehensive Accuracy & Robustness Benchmark for the Warehouse AI Assistant.

Tests all categories of warehouse supervisor queries:
  1. Global Shift & Aggregate Statistics
  2. Specific Event ID Explanations
  3. Behaviour-Specific Queries (The 10 Scenarios)
  4. Predictive Near-Miss (USP) Queries
  5. Location & Bay Comparison
  6. Supervisor Recommendations & Coaching
  7. Edge Cases & Anti-Hallucination Tests (Non-existent IDs, Irrelevant queries, Zero occurrences)

Usage:
    python test_accuracy.py
"""

import os
import sys
import json
import time
from assistant.event_loader import EventStore
from assistant.llm_client import get_llm_client
from assistant import queries as q

TEST_SUITE = [
    # 1. Global Shift & Aggregate Statistics
    {
        "category": "Global Aggregates",
        "question": "How many total warehouse events happened today, and how many were near misses?",
        "expected_facts": ["250", "15"],
        "description": "Checks global count aggregation."
    },
    # 2. Specific Event ID Explanations & Telemetry
    {
        "category": "Specific Event ID",
        "question": "Tell me about event EVT-20260908-0001. Why was it flagged and what was recommended?",
        "expected_facts": ["EVT-20260908-0001", "Throw", "slide", "prohibit"],
        "description": "Checks exact event lookup and recommendation grounding."
    },
    # 3. Behaviour-Specific Queries (The 10 Scenarios)
    {
        "category": "Behaviour Detection (Dragging)",
        "question": "Did any operators drag cartons across the floor without mechanical equipment today?",
        "expected_facts": ["drag", "EVT-"],
        "description": "Scenario 1: DRAG_NO_EQUIPMENT."
    },
    {
        "category": "Behaviour Detection (Throwing)",
        "question": "Were there cartons thrown or slid across the floor?",
        "expected_facts": ["throw", "EVT-"],
        "description": "Scenario 5: CARTON_THROW_SLIDE."
    },
    # 4. Predictive Near-Miss (USP)
    {
        "category": "Predictive Near-Miss (USP)",
        "question": "Explain our predictive near-miss detections: how many were prevented before impact?",
        "expected_facts": ["15", "near miss"],
        "description": "Checks near-miss USP reasoning."
    },
    # 5. EDGE CASES & Anti-Hallucination Guardrails
    {
        "category": "Edge Case - Non-existent ID",
        "question": "Give me the details and telemetry for event EVT-99999999.",
        "expected_facts": ["not found", "don't have", "no event", "does not exist", "unable to find"],
        "description": "Must not hallucinate non-existent event IDs."
    },
    {
        "category": "Edge Case - Out-of-Domain Query",
        "question": "What is the capital of France and who won the FIFA World Cup?",
        "expected_facts": ["warehouse", "cannot answer", "only", "safety", "assist"],
        "description": "Must decline or redirect non-warehouse questions."
    },
    {
        "category": "Edge Case - False Premise Trap",
        "question": "Why did operator #999 drop a 50kg refrigerator from a 15-meter crane today?",
        "expected_facts": ["no record", "no event", "not found", "did not occur", "no evidence", "no such"],
        "description": "Must catch false premise and reject hallucination."
    }
]

def run_benchmark():
    print("=" * 80)
    print("        WAREHOUSE ASSISTANT ACCURACY & ROBUSTNESS BENCHMARK")
    print("=" * 80)
    
    events_path = "data/warehouse_events.json"
    if not os.path.exists(events_path):
        events_path = "outputs_person_b/warehouse_events.json"
        
    store = EventStore.load(events_path)
    llm = get_llm_client()
    
    provider_name = type(llm).__name__
    model_name = getattr(llm, "model", "mock")
    print(f"Loaded {len(store.events)} events.")
    print(f"Backend LLM: {provider_name} ({model_name})\n")
    
    results = []
    total_tests = len(TEST_SUITE)
    passed_tests = 0
    
    for idx, test in enumerate(TEST_SUITE, 1):
        cat = test["category"]
        question = test["question"]
        expected = test["expected_facts"]
        desc = test["description"]
        
        print(f"[{idx}/{total_tests}] [{cat}]")
        print(f"  Q: {question}")
        
        start_time = time.time()
        try:
            res = q.ask(store, llm, question)
            answer = res.answer
            elapsed = time.time() - start_time
            
            # Check factual match (at least one or more expected keywords/facts present)
            answer_lower = answer.lower()
            matched_facts = [f for f in expected if f.lower() in answer_lower]
            passed = len(matched_facts) > 0
            
            if passed:
                passed_tests += 1
                status = "[PASS]"
            else:
                status = "[FAIL]"
                
            print(f"  Status: {status} ({elapsed:.2f}s) | Matched: {matched_facts}")
            snippet = answer.replace('\n', ' ')[:140]
            print(f"  Answer: {snippet}...\n")
            time.sleep(12)  # Respect free tier 5 requests per minute limit
            
            results.append({
                "id": idx,
                "category": cat,
                "question": question,
                "status": status,
                "elapsed_sec": round(elapsed, 2),
                "matched_facts": matched_facts,
                "expected_facts": expected,
                "answer_snippet": snippet
            })
            
        except Exception as e:
            print(f"  Status: [ERROR] -> {e}\n")
            results.append({
                "id": idx,
                "category": cat,
                "question": question,
                "status": "[ERROR]",
                "error": str(e)
            })
            
    print("=" * 80)
    print("                      BENCHMARK SUMMARY")
    print("=" * 80)
    pass_rate = (passed_tests / total_tests) * 100
    print(f"Total Tests Run   : {total_tests}")
    print(f"Passed Tests      : {passed_tests} / {total_tests}")
    print(f"Accuracy Rate     : {pass_rate:.1f}%")
    print("=" * 80)
    
    # Save benchmark report
    report_path = "data/accuracy_benchmark_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "provider": provider_name,
            "pass_rate": f"{pass_rate:.1f}%",
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "results": results
        }, f, indent=2)
    print(f"Detailed benchmark report saved to: {report_path}\n")

if __name__ == "__main__":
    run_benchmark()
