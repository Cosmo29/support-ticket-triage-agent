"""
Unit Tests for FineTuned and ZeroShot Classifiers.
"""

from app.pipeline.classifiers import FineTunedClassifier, ZeroShotLLMClassifier

def test_fine_tuned_classifier():
    clf = FineTunedClassifier()
    assert clf.is_trained is True

    result = clf.classify(
        subject="Payment failed on subscription renewal",
        body="Our corporate Visa was charged but returned ERR_PAYMENT_FAILED_01. Please help process refund or retry."
    )
    assert result.intent == "Billing & Refunds"
    assert result.priority in ("P0", "P1", "P2", "P3")
    assert 0.0 <= result.confidence <= 1.0
    assert result.classifier_type == "fine-tuned"
    assert result.token_usage["prompt_tokens"] == 0

def test_zero_shot_classifier():
    clf = ZeroShotLLMClassifier()
    result = clf.classify(
        subject="CRITICAL: Complete database cluster offline - 500 error",
        body="All production servers are unreachable. Total outage impacting enterprise users."
    )
    assert result.intent == "Cloud Outage & Downtime"
    assert result.priority == "P0"
    assert result.urgency_score >= 0.8
    assert result.confidence >= 0.8
    assert result.token_usage["prompt_tokens"] > 0
