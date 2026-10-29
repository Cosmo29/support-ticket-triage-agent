"""
Standard Library Test Runner for Support Ticket Triage Agent.
Executes all unit tests in tests/ without requiring third-party test runners.
"""

import sys
import os
import traceback

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_all_tests():
    print("=" * 70)
    print("🧪 Running Support Ticket Triage Agent Test Suite")
    print("=" * 70)

    from tests.test_guardrails import (
        test_clean_support_ticket,
        test_system_prompt_override_attempt,
        test_role_hijacking_and_jailbreak,
        test_text_sanitizer
    )
    from tests.test_classifiers import (
        test_fine_tuned_classifier,
        test_zero_shot_classifier
    )
    from tests.test_triage_pipeline import (
        test_enterprise_outage_triage,
        test_malicious_injection_quarantine
    )
    from tests.test_webhooks import (
        test_zendesk_webhook_parser,
        test_slack_payload_parser,
        test_email_webhook_parser
    )

    test_functions = [
        ("Guardrail: Clean Ticket Pass-through", test_clean_support_ticket),
        ("Guardrail: System Prompt Override Interception", test_system_prompt_override_attempt),
        ("Guardrail: Role Hijacking Quarantine", test_role_hijacking_and_jailbreak),
        ("Guardrail: HTML/Delimiter Sanitizer", test_text_sanitizer),
        ("Classifier: Fine-Tuned Model Prediction", test_fine_tuned_classifier),
        ("Classifier: Zero-Shot LLM Reasoning", test_zero_shot_classifier),
        ("Pipeline: Enterprise P0 Triage & Tool Calling", test_enterprise_outage_triage),
        ("Pipeline: Adversarial Attack Containment", test_malicious_injection_quarantine),
        ("Connectors: Zendesk Webhook Parser", test_zendesk_webhook_parser),
        ("Connectors: Slack Inbound Parser", test_slack_payload_parser),
        ("Connectors: Email Inbound Webhook Parser", test_email_webhook_parser),
    ]

    passed = 0
    failed = 0

    for name, func in test_functions:
        try:
            func()
            print(f"  ✅ PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ FAIL: {name} -> {e}")
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"Test Run Finished: {passed} Passed, {failed} Failed")
    print("=" * 70)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all_tests()
