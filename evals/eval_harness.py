"""
Evaluation Harness: TF-IDF + Calibrated Linear Baseline vs. Zero-Shot LLM.
Evaluates Intent/Priority F1, latency, and cost on the public test split (data/splits/test.json).
Enforces real API execution: skips the LLM column if no API key is configured.
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.metrics import f1_score, precision_score, recall_score

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.pipeline.classifiers import TfidfCalibratedBaseline, ZeroShotLLMClassifier
from app.security.guardrails import inspect_for_prompt_injection
from app.config import settings

PUBLIC_TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/splits/test.json"))
ADVERSARIAL_SUITE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "adversarial_and_edge_cases.json"))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval_results.json"))

COST_PER_PROMPT_TOKEN = 0.15 / 1_000_000
COST_PER_COMPLETION_TOKEN = 0.60 / 1_000_000

def run_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("🚀 AI Support Ticket Triage Agent: Benchmark Evaluation")
    print("=" * 70)

    # 1. Load Public Test Benchmark
    if not os.path.exists(PUBLIC_TEST_PATH):
        print(f"Test split not found at {PUBLIC_TEST_PATH}. Running evals/prepare_data.py...")
        from evals.prepare_data import prepare_splits
        prepare_splits()

    with open(PUBLIC_TEST_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print(f"Loaded {len(test_samples)} public test benchmark samples from data/splits/test.json.")
    
    # 2. Initialize Classifiers
    baseline_clf = TfidfCalibratedBaseline()
    llm_clf = ZeroShotLLMClassifier()

    has_llm_key = llm_clf.has_api_key
    if not has_llm_key:
        print("\n⚠️  LLM column skipped: no API key")
        print("   Set OPENAI_API_KEY or GEMINI_API_KEY to benchmark live LLM inference.\n")

    ground_truth_intents = []
    ground_truth_priorities = []
    
    # Baseline collections
    base_pred_intents = []
    base_pred_priorities = []
    base_latencies = []

    # LLM collections (only if live key present)
    llm_pred_intents = []
    llm_pred_priorities = []
    llm_latencies = []
    llm_costs = []
    llm_prompt_tokens = 0
    llm_comp_tokens = 0

    print(f"Evaluating {len(test_samples)} samples on TF-IDF + calibrated linear baseline...")
    for item in test_samples:
        subject = item["subject"]
        body = item["body"]
        gt_intent = item.get("ground_truth_intent") or item.get("intent")
        gt_prio = item.get("ground_truth_priority") or item.get("priority")

        ground_truth_intents.append(gt_intent)
        ground_truth_priorities.append(gt_prio)

        # Baseline Model (TF-IDF + Calibrated Logistic Regression)
        t0 = time.time()
        b_res = baseline_clf.classify(subject, body)
        base_latencies.append((time.time() - t0) * 1000)
        base_pred_intents.append(b_res.intent)
        base_pred_priorities.append(b_res.priority)

        # Live LLM Model (strictly live API calls, never simulated)
        if has_llm_key:
            try:
                t1 = time.time()
                l_res = llm_clf.classify(subject, body, require_live=True)
                llm_latencies.append((time.time() - t1) * 1000)
                llm_pred_intents.append(l_res.intent)
                llm_pred_priorities.append(l_res.priority)

                p_tok = l_res.token_usage.get("prompt_tokens", 0)
                c_tok = l_res.token_usage.get("completion_tokens", 0)
                llm_prompt_tokens += p_tok
                llm_comp_tokens += c_tok
                cost = (p_tok * COST_PER_PROMPT_TOKEN) + (c_tok * COST_PER_COMPLETION_TOKEN)
                llm_costs.append(cost)
            except Exception as e:
                print(f"Live LLM call error: {e}")
                has_llm_key = False

    # 3. Compute Metrics for Baseline
    base_intent_macro_f1 = float(f1_score(ground_truth_intents, base_pred_intents, average="macro", zero_division=0))
    base_intent_micro_f1 = float(f1_score(ground_truth_intents, base_pred_intents, average="micro", zero_division=0))
    base_prio_macro_f1 = float(f1_score(ground_truth_priorities, base_pred_priorities, average="macro", zero_division=0))
    base_prio_micro_f1 = float(f1_score(ground_truth_priorities, base_pred_priorities, average="micro", zero_division=0))

    base_p50 = float(np.percentile(base_latencies, 50))
    base_p95 = float(np.percentile(base_latencies, 95))
    base_p99 = float(np.percentile(base_latencies, 99))

    # 4. Compute Metrics for LLM (if executed)
    llm_results = None
    if has_llm_key and llm_pred_intents:
        llm_intent_macro_f1 = float(f1_score(ground_truth_intents, llm_pred_intents, average="macro", zero_division=0))
        llm_intent_micro_f1 = float(f1_score(ground_truth_intents, llm_pred_intents, average="micro", zero_division=0))
        llm_prio_macro_f1 = float(f1_score(ground_truth_priorities, llm_pred_priorities, average="macro", zero_division=0))
        llm_prio_micro_f1 = float(f1_score(ground_truth_priorities, llm_pred_priorities, average="micro", zero_division=0))

        llm_p50 = float(np.percentile(llm_latencies, 50))
        llm_p95 = float(np.percentile(llm_latencies, 95))
        llm_p99 = float(np.percentile(llm_latencies, 99))
        llm_cost_per_1k = float(np.mean(llm_costs) * 1000) if llm_costs else 0.0

        llm_results = {
            "name": f"Zero-Shot LLM ({settings.LLM_PROVIDER} / {settings.LLM_MODEL})",
            "intent_macro_f1": round(llm_intent_macro_f1, 4),
            "intent_micro_f1": round(llm_intent_micro_f1, 4),
            "priority_macro_f1": round(llm_prio_macro_f1, 4),
            "priority_micro_f1": round(llm_prio_micro_f1, 4),
            "latency_p50_ms": round(llm_p50, 2),
            "latency_p95_ms": round(llm_p95, 2),
            "latency_p99_ms": round(llm_p99, 2),
            "cost_per_1000_tickets_usd": round(llm_cost_per_1k, 4),
            "token_usage": {
                "total_prompt_tokens": llm_prompt_tokens,
                "total_completion_tokens": llm_comp_tokens
            }
        }

    # 5. Evaluate Separate Adversarial & Edge Cases Suite
    edge_results = {"total_cases": 0, "injection_cases": 0, "injection_blocked": 0, "block_rate": 0.0}
    if os.path.exists(ADVERSARIAL_SUITE_PATH):
        with open(ADVERSARIAL_SUITE_PATH, "r", encoding="utf-8") as f:
            edge_samples = json.load(f)
        
        edge_results["total_cases"] = len(edge_samples)
        for s in edge_samples:
            if s.get("is_injection"):
                edge_results["injection_cases"] += 1
                assessment = inspect_for_prompt_injection(f"{s['subject']} {s['body']}")
                if not assessment.is_safe:
                    edge_results["injection_blocked"] += 1

        if edge_results["injection_cases"] > 0:
            edge_results["block_rate"] = round(
                (edge_results["injection_blocked"] / edge_results["injection_cases"]) * 100, 1
            )

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {
            "name": "PolyAI/banking77",
            "source": "Hugging Face (CC BY 4.0 License)",
            "test_split_size": len(test_samples),
            "classes": 5
        },
        "models": {
            "tfidf_calibrated_baseline": {
                "name": "TF-IDF + Calibrated Linear Baseline",
                "intent_macro_f1": round(base_intent_macro_f1, 4),
                "intent_micro_f1": round(base_intent_micro_f1, 4),
                "priority_macro_f1": round(base_prio_macro_f1, 4),
                "priority_micro_f1": round(base_prio_micro_f1, 4),
                "latency_p50_ms": round(base_p50, 2),
                "latency_p95_ms": round(base_p95, 2),
                "latency_p99_ms": round(base_p99, 2),
                "cost_per_1000_tickets_usd": 0.0
            },
            "zero_shot_llm": llm_results
        },
        "adversarial_and_edge_suite": edge_results
    }

    # Save to disk
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print(f"📊 BENCHMARK EVALUATION RESULTS (PUBLIC TEST SPLIT: {len(test_samples)} TICKETS)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'TF-IDF Calibrated Baseline':<28} | {'Zero-Shot LLM':<18}")
    print("-" * 70)
    
    llm_f1_str = f"{llm_results['intent_macro_f1']*100:.2f}%" if llm_results else "Skipped (no API key)"
    llm_acc_str = f"{llm_results['intent_micro_f1']*100:.2f}%" if llm_results else "Skipped (no API key)"
    llm_prio_str = f"{llm_results['priority_macro_f1']*100:.2f}%" if llm_results else "Skipped (no API key)"
    llm_lat_str = f"{llm_results['latency_p50_ms']:.2f}ms" if llm_results else "Skipped (no API key)"
    llm_cost_str = f"${llm_results['cost_per_1000_tickets_usd']:.4f}" if llm_results else "Skipped (no API key)"

    print(f"{'Intent Macro F1':<30} | {base_intent_macro_f1*100:>26.2f}% | {llm_f1_str:>18}")
    print(f"{'Intent Micro F1 (Accuracy)':<30} | {base_intent_micro_f1*100:>26.2f}% | {llm_acc_str:>18}")
    print(f"{'Priority Macro F1':<30} | {base_prio_macro_f1*100:>26.2f}% | {llm_prio_str:>18}")
    print(f"{'Priority Micro F1':<30} | {base_prio_micro_f1*100:>26.2f}% | {'Skipped' if not llm_results else ''}")
    print(f"{'Latency p50 (Median)':<30} | {base_p50:>25.2f}ms | {llm_lat_str:>18}")
    print(f"{'Latency p95':<30} | {base_p95:>25.2f}ms | {'Skipped' if not llm_results else ''}")
    print(f"{'Cost per 1,000 Tickets':<30} | {'$0.00':>28} | {llm_cost_str:>18}")
    print("-" * 70)
    print(f"Adversarial Suite (Separate): {edge_results['injection_blocked']}/{edge_results['injection_cases']} injection attacks caught by regex filter ({edge_results['block_rate']}%).")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_PATH}\n")

    return results

if __name__ == "__main__":
    run_evaluation()
