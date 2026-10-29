"""
Unit Tests for TF-IDF Calibrated Baseline and ZeroShot LLM Classifiers.
"""

from app.pipeline.classifiers import TfidfCalibratedBaseline, FineTunedClassifier, ZeroShotLLMClassifier

def test_tfidf_calibrated_baseline():
    clf = TfidfCalibratedBaseline()
    assert clf.is_trained is True

    result = clf.classify(
        subject="Payment failed on subscription renewal",
        body="Our corporate Visa was charged but returned fee error. Please help process refund or retry transfer."
    )
    assert result.intent in [
        "Billing & Payments", "Refunds & Disputes", "Account & Authentication",
        "Technical & System Issues", "General Service Inquiry", "Billing & Refunds"
    ]
    assert result.priority in ("P0", "P1", "P2", "P3")
    assert 0.0 <= result.confidence <= 1.0
    assert result.classifier_type == "tfidf-calibrated-baseline"
    assert result.token_usage["prompt_tokens"] == 0

    # Test alias compatibility
    alias_clf = FineTunedClassifier()
    assert alias_clf.is_trained is True

def test_zero_shot_classifier_key_enforcement():
    clf = ZeroShotLLMClassifier()
    if not clf.has_api_key:
        # Enforce that require_live=True raises an error when no API key is configured
        try:
            clf.classify(
                subject="CRITICAL: Outage",
                body="All production servers unreachable",
                require_live=True
            )
            assert False, "Should have raised RuntimeError when no API key is present"
        except RuntimeError as e:
            assert "No LLM API key configured" in str(e)
            
        # Demo fallback runs safely for interactive dashboard exploration
        res = clf.classify(subject="Help", body="Need invoice", require_live=False)
        assert res.classifier_type == "zero-shot-llm-demo-fallback"
    else:
        res = clf.classify(subject="Help", body="Need invoice", require_live=True)
        assert res.classifier_type == "zero-shot-llm"
