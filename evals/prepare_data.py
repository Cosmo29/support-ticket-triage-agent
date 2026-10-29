"""
Public Dataset Ingestion & Fixed-Seed Stratified Splitting Script.
Ingests PolyAI/banking77 (CC BY 4.0 License), maps fine-grained intents into 5 standardized
enterprise support categories and 4 priority levels, and saves train/val/test splits with seed=42.
"""

import os
import sys
import json
from typing import Dict, Any, List
import pandas as pd
from sklearn.model_selection import train_test_split

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

RANDOM_SEED = 42

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/splits"))
os.makedirs(DATA_DIR, exist_ok=True)

def map_intent(label_name: str) -> str:
    l = label_name.lower()
    if any(k in l for k in ["refund", "dispute", "not_recognised", "wrong_amount", "cancel_transfer"]):
        return "Refunds & Disputes"
    elif any(k in l for k in ["fee", "payment", "transfer", "top_up", "balance", "charge", "exchange_charge"]):
        return "Billing & Payments"
    elif any(k in l for k in ["pin", "passcode", "identity", "verify", "compromised", "stolen", "security", "lost"]):
        return "Account & Authentication"
    elif any(k in l for k in ["not_working", "declined", "failed", "blocked", "atm_support", "apple_pay"]):
        return "Technical & System Issues"
    else:
        return "General Service Inquiry"

def map_priority(label_name: str) -> str:
    l = label_name.lower()
    if any(k in l for k in ["compromised", "stolen", "lost"]):
        return "P0"  # Blocker / Security Critical
    elif any(k in l for k in ["passcode", "declined", "blocked", "not_recognised"]):
        return "P1"  # High Priority Access/Financial
    elif any(k in l for k in ["refund", "failed", "not_working", "wrong_amount"]):
        return "P2"  # Medium Priority Investigation
    else:
        return "P3"  # Low Priority Inquiry / Information

def prepare_splits() -> Dict[str, int]:
    print("=" * 70)
    print("📦 Ingesting Public Customer Support Dataset (PolyAI/banking77)")
    print("=" * 70)
    
    from datasets import load_dataset
    ds = load_dataset("PolyAI/banking77")
    names = ds["train"].features["label"].names

    # Combine train and test for clean fixed-seed 70/15/15 stratified split
    all_texts = list(ds["train"]["text"]) + list(ds["test"]["text"])
    raw_labels = list(ds["train"]["label"]) + list(ds["test"]["label"])

    mapped_intents = [map_intent(names[idx]) for idx in raw_labels]
    mapped_priorities = [map_priority(names[idx]) for idx in raw_labels]
    raw_names = [names[idx] for idx in raw_labels]

    df = pd.DataFrame({
        "body": all_texts,
        "subject": [f"Support request: {n.replace('_', ' ').title()}" for n in raw_names],
        "raw_label": raw_names,
        "intent": mapped_intents,
        "priority": mapped_priorities
    })

    print(f"Total dataset size: {len(df)} samples across {len(names)} original classes.")
    print("Mapped Intent distribution:")
    print(df["intent"].value_counts())
    print("\nMapped Priority distribution:")
    print(df["priority"].value_counts())

    # Sample balanced, fast evaluation set of 3,000 samples for deterministic fast evals:
    # 2,100 train (70%), 450 val (15%), 450 test (15%)
    df_sampled = (
        df.groupby("intent", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 600), random_state=RANDOM_SEED))
        .sample(frac=1.0, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    train_df, temp_df = train_test_split(
        df_sampled, test_size=0.30, random_state=RANDOM_SEED, stratify=df_sampled["intent"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=RANDOM_SEED, stratify=temp_df["intent"]
    )

    def to_records(split_df, prefix="PUB"):
        out = []
        for idx, row in split_df.iterrows():
            out.append({
                "id": f"{prefix}-{idx+1:04d}",
                "subject": row["subject"],
                "body": row["body"],
                "customer_email": f"user_{idx+1}@customer-sample.org",
                "ground_truth_intent": row["intent"],
                "ground_truth_priority": row["priority"],
                "raw_label": row["raw_label"],
                "intent": row["intent"],
                "priority": row["priority"]
            })
        return out

    train_data = to_records(train_df, "TRAIN")
    val_data = to_records(val_df, "VAL")
    test_data = to_records(test_df, "TEST")

    train_path = os.path.join(DATA_DIR, "train.json")
    val_path = os.path.join(DATA_DIR, "val.json")
    test_path = os.path.join(DATA_DIR, "test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)

    print(f"\nSaved {len(train_data)} train samples to: {train_path}")
    print(f"Saved {len(val_data)} val samples to: {val_path}")
    print(f"Saved {len(test_data)} test samples to: {test_path}")

    # Generate DATASET.md
    doc_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "DATASET.md"))
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(f"""# Dataset Provenance and Splitting Methodology

## 1. Source and Provenance
- **Dataset:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Publication:** Casanueva et al., *"Efficient Intent Detection with Dual Sentence Encoders"*, EMNLP Findings 2020.
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Total Population Size:** 13,083 real customer queries across 77 fine-grained intents
- **Evaluation Sample Size:** {len(df_sampled)} balanced customer tickets (600 per category across 5 macro support intents)

## 2. Intent Taxonomy Mapping
The 77 fine-grained intents are mapped deterministically to 5 enterprise support categories:

| Target Pipeline Intent | Criteria / Keywords from 77 Intents | Example Original Intents |
| :--- | :--- | :--- |
| **Billing & Payments** | Fees, transfers, top-up, card payments, balances | `transfer_fee_charged`, `automatic_top_up`, `card_payment_fee_charged` |
| **Refunds & Disputes** | Chargeback, refunds, unrecognized charges, cancellation | `request_refund`, `refund_not_showing_up`, `card_payment_not_recognised` |
| **Account & Authentication** | PIN, passwords, biometric identity, compromised credentials | `passcode_forgotten`, `verify_my_identity`, `compromised_card` |
| **Technical & System Issues**| Errors, service downtime, card failures, ATM crashes | `card_not_working`, `apple_pay_or_google_pay`, `atm_support` |
| **General Service Inquiry** | Delivery estimates, exchanges, account questions | `card_arrival`, `card_delivery_estimate`, `country_support` |

## 3. Priority Mapping Rubric
- **P0 (Critical / Immediate Security Risk):** Compromised cards, stolen devices, security breaches. Target SLA: < 15–30 min.
- **P1 (High / Access & Transfer Failure):** Forgotten master passcode, unrecognized payment, declined payment at point of sale. Target SLA: < 1–2 hours.
- **P2 (Medium / Processing & Hardware Issues):** Refund dispute processing, card failure, app errors with workarounds. Target SLA: < 4 hours.
- **P3 (Low / Informational):** Delivery status, documentation, currency rate inquiry. Target SLA: < 24 hours.

## 4. Reproducible Fixed-Seed Splits
- **Seed:** `{RANDOM_SEED}` (`random_state=42`, stratified on intent)
- **Train Split (70%):** {len(train_data)} tickets (`data/splits/train.json`)
- **Validation Split (15%):** {len(val_data)} tickets (`data/splits/val.json`)
- **Test Benchmark Split (15%):** {len(test_data)} tickets (`data/splits/test.json`)

## 5. Adversarial & Edge Cases Suite
- Hand-crafted test cases (24 samples) are isolated in `evals/adversarial_and_edge_cases.json`.
- Evaluated as a separate unit test suite to test regex defenses and malformed payloads, keeping the generalization benchmark statistically sound.
""")

    print(f"Generated dataset documentation: {doc_path}")
    return {"total": len(df_sampled), "train": len(train_data), "val": len(val_data), "test": len(test_data)}

if __name__ == "__main__":
    prepare_splits()
