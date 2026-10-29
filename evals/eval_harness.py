"""
Evaluation Harness: Fine-Tuned Classifier vs. Zero-Shot LLM Benchmark.
Compares Macro/Micro F1, Precision, Recall, Latency (p50/p95/p99), and Cost per 1,000 tickets.
Saves results to evals/eval_results.json for display in the dashboard.
"""

from __future__ import annotations
import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


from app.pipeline.classifiers import FineTunedClassifier, ZeroShotLLMClassifier
from app.security.guardrails import inspect_for_prompt_injection

TEST_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "test_dataset.json"))
RESULTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "eval_results.json"))

# Pricing estimates per 1M tokens ($0.15/1M input, $0.60/1M output for flash/mini models)
COST_PER_PROMPT_TOKEN = 0.15 / 1_000_000
COST_PER_COMPLETION_TOKEN = 0.60 / 1_000_000

def run_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("🚀 Running AI Support Ticket Triage Agent Evaluation Benchmark")
    print("=" * 70)

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test dataset not found at {TEST_DATA_PATH}")

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_samples = json.load(f)

    print(f"Loaded {len(test_samples)} benchmark test samples.")
    print("Initializing Fine-Tuned Classifier & Zero-Shot LLM Classifier...")

    ft_clf = FineTunedClassifier()
    zs_clf = ZeroShotLLMClassifier()

    ground_truth_intents = []
    ground_truth_priorities = []
    
    ft_pred_intents = []
    ft_pred_priorities = []
    ft_latencies = []
    ft_costs = []

    zs_pred_intents = []
    zs_pred_priorities = []
    zs_latencies = []
    zs_costs = []
    zs_total_prompt_tokens = 0
    zs_total_comp_tokens = 0

    # Guardrail metrics
    guardrail_correct = 0
    total_adversarial = 0

    print("Evaluating test tickets...")
    for item in test_samples:
        subject = item["subject"]
        body = item["body"]
        gt_intent = item["ground_truth_intent"]
        gt_prio = item["ground_truth_priority"]
        is_injection = item.get("is_injection", False)

        ground_truth_intents.append(gt_intent)
        ground_truth_priorities.append(gt_prio)

        # 1. Guardrail test
        guardrail_res = inspect_for_prompt_injection(f"{subject} {body}")
        if is_injection:
            total_adversarial += 1
            if not guardrail_res.is_safe:
                guardrail_correct += 1

        # 2. Fine-Tuned Classifier
        t0 = time.time()
        ft_res = ft_clf.classify(subject, body)
        ft_lat = (time.time() - t0) * 1000
        ft_latencies.append(ft_lat)
        ft_pred_intents.append(ft_res.intent)
        ft_pred_priorities.append(ft_res.priority)
        ft_costs.append(0.0)

        # 3. Zero-Shot LLM Classifier
        t1 = time.time()
        zs_res = zs_clf.classify(subject, body)
        zs_lat = (time.time() - t1) * 1000
        zs_latencies.append(zs_lat)
        zs_pred_intents.append(zs_res.intent)
        zs_pred_priorities.append(zs_res.priority)
        
        p_tok = zs_res.token_usage.get("prompt_tokens", 0)
        c_tok = zs_res.token_usage.get("completion_tokens", 0)
        zs_total_prompt_tokens += p_tok
        zs_total_comp_tokens += c_tok
        cost = (p_tok * COST_PER_PROMPT_TOKEN) + (c_tok * COST_PER_COMPLETION_TOKEN)
        zs_costs.append(cost)

    # -------------------------------------------------------------
    # Compute Metrics: Fine-Tuned
    # -------------------------------------------------------------
    ft_intent_macro_f1 = f1_score(ground_truth_intents, ft_pred_intents, average="macro", zero_division=0)
    ft_intent_micro_f1 = f1_score(ground_truth_intents, ft_pred_intents, average="micro", zero_division=0)
    ft_prio_macro_f1 = f1_score(ground_truth_priorities, ft_pred_priorities, average="macro", zero_division=0)
    ft_prio_micro_f1 = f1_score(ground_truth_priorities, ft_pred_priorities, average="micro", zero_division=0)

    ft_p50 = float(np.percentile(ft_latencies, 50))
    ft_p95 = float(np.percentile(ft_latencies, 95))
    ft_p99 = float(np.percentile(ft_latencies, 99))
    ft_cost_per_1k = 0.0

    # -------------------------------------------------------------
    # Compute Metrics: Zero-Shot LLM
    # -------------------------------------------------------------
    zs_intent_macro_f1 = f1_score(ground_truth_intents, zs_pred_intents, average="macro", zero_division=0)
    zs_intent_micro_f1 = f1_score(ground_truth_intents, zs_pred_intents, average="micro", zero_division=0)
    zs_prio_macro_f1 = f1_score(ground_truth_priorities, zs_pred_priorities, average="macro", zero_division=0)
    zs_prio_micro_f1 = f1_score(ground_truth_priorities, zs_pred_priorities, average="micro", zero_division=0)

    zs_p50 = float(np.percentile(zs_latencies, 50))
    zs_p95 = float(np.percentile(zs_latencies, 95))
    zs_p99 = float(np.percentile(zs_latencies, 99))
    zs_avg_cost_per_ticket = float(np.mean(zs_costs))
    zs_cost_per_1k = zs_avg_cost_per_ticket * 1000

    guardrail_accuracy = (guardrail_correct / total_adversarial * 100) if total_adversarial > 0 else 100.0

    results = {
        "benchmark_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_samples_count": len(test_samples),
        "guardrail_adversarial_detection_rate": round(guardrail_accuracy, 2),
        "models": {
            "fine_tuned": {
                "name": "Specialized TF-IDF + Calibrated Classifier",
                "intent_macro_f1": round(ft_intent_macro_f1, 4),
                "intent_micro_f1": round(ft_intent_micro_f1, 4),
                "priority_macro_f1": round(ft_prio_macro_f1, 4),
                "priority_micro_f1": round(ft_prio_micro_f1, 4),
                "latency_p50_ms": round(ft_p50, 2),
                "latency_p95_ms": round(ft_p95, 2),
                "latency_p99_ms": round(ft_p99, 2),
                "cost_per_1000_tickets_usd": 0.0,
                "token_usage": {"total_prompt_tokens": 0, "total_completion_tokens": 0}
            },
            "zero_shot_llm": {
                "name": "Zero-Shot LLM Reasoning (Gemini/OpenAI)",
                "intent_macro_f1": round(zs_intent_macro_f1, 4),
                "intent_micro_f1": round(zs_intent_micro_f1, 4),
                "priority_macro_f1": round(zs_prio_macro_f1, 4),
                "priority_micro_f1": round(zs_prio_micro_f1, 4),
                "latency_p50_ms": round(zs_p50, 2),
                "latency_p95_ms": round(zs_p95, 2),
                "latency_p99_ms": round(zs_p99, 2),
                "cost_per_1000_tickets_usd": round(zs_cost_per_1k, 4),
                "token_usage": {
                    "total_prompt_tokens": zs_total_prompt_tokens,
                    "total_completion_tokens": zs_total_comp_tokens
                }
            }
        }
    }

    # Save to disk
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Print Summary Table
    print("\n" + "=" * 70)
    print("📊 BENCHMARK EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Fine-Tuned Classifier':<18} | {'Zero-Shot LLM':<18}")
    print("-" * 70)
    print(f"{'Intent Macro F1':<30} | {ft_intent_macro_f1*100:>17.2f}% | {zs_intent_macro_f1*100:>17.2f}%")
    print(f"{'Intent Micro F1 (Accuracy)':<30} | {ft_intent_micro_f1*100:>17.2f}% | {zs_intent_micro_f1*100:>17.2f}%")
    print(f"{'Priority Macro F1':<30} | {ft_prio_macro_f1*100:>17.2f}% | {zs_prio_macro_f1*100:>17.2f}%")
    print(f"{'Priority Micro F1':<30} | {ft_prio_micro_f1*100:>17.2f}% | {zs_prio_micro_f1*100:>17.2f}%")
    print(f"{'Latency p50 (Median)':<30} | {ft_p50:>16.2f}ms | {zs_p50:>16.2f}ms")
    print(f"{'Latency p95':<30} | {ft_p95:>16.2f}ms | {zs_p95:>16.2f}ms")
    print(f"{'Latency p99':<30} | {ft_p99:>16.2f}ms | {zs_p99:>16.2f}ms")
    print(f"{'Cost per 1,000 Tickets':<30} | {'$0.00':>18} | ${zs_cost_per_1k:>17.4f}")
    print(f"{'Adversarial Injection Defense':<30} | {guardrail_accuracy:>17.1f}% | {'Protected':>18}")
    print("=" * 70)
    print(f"Results saved to: {RESULTS_PATH}\n")

    return results

if __name__ == "__main__":
    run_evaluation()
