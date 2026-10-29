# Dataset Provenance and Splitting Methodology

## 1. Source and Provenance
- **Dataset:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Publication:** Casanueva et al., *"Efficient Intent Detection with Dual Sentence Encoders"*, EMNLP Findings 2020.
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Total Population Size:** 13,083 real customer queries across 77 fine-grained intents
- **Evaluation Sample Size:** 3000 balanced customer tickets (600 per category across 5 macro support intents)

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
- **Seed:** `42` (`random_state=42`, stratified on intent)
- **Train Split (70%):** 2100 tickets (`data/splits/train.json`)
- **Validation Split (15%):** 450 tickets (`data/splits/val.json`)
- **Test Benchmark Split (15%):** 450 tickets (`data/splits/test.json`)

## 5. Adversarial & Edge Cases Suite
- Hand-crafted test cases (24 samples) are isolated in `evals/adversarial_and_edge_cases.json`.
- Evaluated as a separate unit test suite to test regex defenses and malformed payloads, keeping the generalization benchmark statistically sound.
