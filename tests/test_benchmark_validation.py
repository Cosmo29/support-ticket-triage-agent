"""
Unit Tests for Benchmark Validation, Split Disjointness, Calibration Validity, and Live LLM Enforcement.
"""

import os
import json
import re
import numpy as np
from app.pipeline.classifiers import TfidfCalibratedBaseline, ZeroShotLLMClassifier

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/splits"))

def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def test_split_disjointness():
    """
    Assert that train, validation, and test splits share ZERO exact normalized texts.
    Guarantees no data leakage across evaluation boundaries.
    """
    train_path = os.path.join(DATA_DIR, "train.json")
    val_path = os.path.join(DATA_DIR, "val.json")
    test_path = os.path.join(DATA_DIR, "test.json")

    assert os.path.exists(train_path), "train.json missing"
    assert os.path.exists(val_path), "val.json missing"
    assert os.path.exists(test_path), "test.json missing"

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    train_set = {norm(item["body"]) for item in train_data}
    val_set = {norm(item["body"]) for item in val_data}
    test_set = {norm(item["body"]) for item in test_data}

    # Intersections must be strictly empty
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    assert len(train_val_overlap) == 0, f"Found {len(train_val_overlap)} shared texts between train and val"
    assert len(train_test_overlap) == 0, f"Found {len(train_test_overlap)} shared texts between train and test"
    assert len(val_test_overlap) == 0, f"Found {len(val_test_overlap)} shared texts between val and test"

def test_deterministic_split_with_seed_42():
    """
    Assert that running with random_state=42 produces consistent record counts
    and exact deterministic indices across splits.
    """
    train_path = os.path.join(DATA_DIR, "train.json")
    val_path = os.path.join(DATA_DIR, "val.json")
    test_path = os.path.join(DATA_DIR, "test.json")

    with open(train_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(val_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # Official Banking77 test set is 3,080 rows
    assert len(test_data) == 3080, f"Expected 3080 test samples, got {len(test_data)}"
    # Clean disjoint train is 8,492 rows and Val is 1,500 rows (from 9,992 unique train rows)
    assert len(train_data) == 8492, f"Expected 8492 train samples, got {len(train_data)}"
    assert len(val_data) == 1500, f"Expected 1500 val samples, got {len(val_data)}"

    # Check deterministic record ID sequence
    assert train_data[0]["id"] == "TICK-TRN-00001"
    assert val_data[0]["id"] == "TICK-VAL-00001"
    assert test_data[0]["id"] == "TICK-TST-00001"

def test_calibration_validity():
    """
    Assert that predicted probabilities from the calibrated baseline model are in [0, 1]
    and each row sums to 1.0 (within numerical precision 1e-5).
    """
    clf = TfidfCalibratedBaseline()
    assert clf.is_trained is True

    test_queries = [
        "Where is my replacement debit card? It has not arrived in the mail.",
        "I need a refund for an extra charge on my monthly statement.",
        "My PIN is blocked at the ATM and I cannot log into my mobile app.",
        "System outage: all users getting 500 error when submitting transfers.",
        "What are your exchange rates for converting USD to EUR?"
    ]

    for q in test_queries:
        combined = f"Support {q}"
        prob_dict = clf.predict_proba(combined)
        assert len(prob_dict) == 5, f"Expected 5 classes, got {len(prob_dict)}"

        prob_sum = 0.0
        for cls_name, prob in prob_dict.items():
            assert 0.0 <= prob <= 1.0, f"Probability {prob} for class {cls_name} not in [0, 1]"
            prob_sum += prob

        assert abs(prob_sum - 1.0) < 1e-5, f"Probabilities do not sum to 1.0 (sum={prob_sum})"

        # Also verify classify confidence equals top probability
        res = clf.classify(subject="Support", body=q)
        assert 0.0 <= res.confidence <= 1.0
        max_prob = max(prob_dict.values())
        assert abs(res.confidence - max_prob) < 1e-3, f"Confidence {res.confidence} != top prob {max_prob}"

def test_llm_require_live_behavior():
    """
    Assert that ZeroShotLLMClassifier.classify(require_live=True) strictly refuses to run
    when no API key is provided and raises RuntimeError rather than using a simulator.
    """
    clf = ZeroShotLLMClassifier()
    if not clf.has_api_key:
        try:
            clf.classify(
                subject="URGENT",
                body="Production server down",
                require_live=True
            )
            assert False, "Should have raised RuntimeError when no API key is set"
        except RuntimeError as e:
            assert "No LLM API key configured" in str(e)
    else:
        # If API key is present, require_live=True runs live without error
        res = clf.classify(subject="Inquiry", body="How do I change password?", require_live=True)
        assert res.classifier_type == "zero-shot-llm"
