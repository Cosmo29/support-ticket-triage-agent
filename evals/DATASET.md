# Dataset Provenance, Leakage Validation, and Splitting Methodology

## 1. Source and License
- **Dataset:** [PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)
- **Publication:** Casanueva et al., *"Efficient Intent Detection with Dual Sentence Encoders"*, EMNLP Findings 2020.
- **License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Total Population:** 13,083 customer service queries across 77 fine-grained intents
  - Upstream Official Train: 10,003 queries
  - Upstream Official Test: 3,080 queries

---

## 2. Leakage Analysis & Split Purging

### Upstream Overlap Analysis
1. **Exact Text Matches:**
   - 7 test queries in the raw upstream dataset were identical to training queries (e.g. *"where can i use my card?"*, *"how do i unblock my pin?"*).
2. **Near-Duplicate Queries (TF-IDF Cosine Similarity > 0.95):**
   - **Standard TF-IDF:** 237 test queries (**7.69%**) have a cosine similarity > 0.95 with a training example.
   - **Stop-Words Removed TF-IDF:** 1,016 test queries (**32.99%**) exceed 0.95 similarity. This is an intrinsic characteristic of customer support domains where users ask short paraphrases of the same question (e.g., *"My card hasn't arrived yet"* vs. *"Why hasn't my card arrived yet?"*).
3. **Internal Train Duplicates:**
   - 4 duplicate utterances existed internally within the upstream training split.

### Purging for Guaranteed Split Disjointness
- All 7 test-matching queries and 4 internal duplicates were purged from the training set.
- The resulting clean training pool of **9,992 unique queries** shares **zero exact texts** with the test set:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

---

## 3. Split Methodology & Zero Test-Peeking Guarantee

### Curated Splits
- **Test Split (`data/splits/test.json`):** Exactly 3,080 rows (the official Banking77 benchmark test set, untouched).
- **Train Split (`data/splits/train.json`):** 8,492 rows (stratified by intent from the clean training pool).
- **Validation Split (`data/splits/val.json`):** 1,500 rows (stratified by intent from the clean training pool, `seed=42`).

### Why the Initial 3,000-Row Sample Was Replaced
In an earlier iteration, 3,000 rows were sampled after pooling upstream train and test sets together. This was identified and corrected in Milestone 2b:
1. Pooling contaminated the official evaluation boundary.
2. The initial generation embedded `{raw_label}` into the `subject` field, causing target label leakage.
3. In Milestone 2b, all splits were regenerated directly from the official partitions with neutral subjects (`"Customer Support Inquiry"`), eliminating all label leakage.

### Zero Test Peeking
- **Feature Extraction & Weights:** TF-IDF vocabulary (`ngram_range=(1,2)`, `max_features=5000`) and Logistic Regression parameters are fit strictly on `train.json`.
- **Probability Calibration:** 5-fold cross-validation (`CalibratedClassifierCV`) is performed internally on `train.json`.
- **Escalation Threshold Tuning:** The confidence threshold $\tau$ was selected by sweeping validation accuracy and cost exclusively on `val.json`.
- **Zero test data** was observed during fitting, feature selection, or threshold tuning.

---

## 4. Intent Taxonomy Mapping (77 Intents $\to$ 5 Enterprise Categories)

| Target Pipeline Category | Keyword & Functional Criteria | Example Banking77 Intents |
| :--- | :--- | :--- |
| **Billing & Payments** | Fees, transfers, top-up, card payments, balance inquiries | `transfer_fee_charged`, `automatic_top_up`, `card_payment_fee_charged` |
| **Refunds & Disputes** | Chargebacks, refunds, unrecognized charges, transfer cancellations | `request_refund`, `refund_not_showing_up`, `card_payment_not_recognised` |
| **Account & Authentication** | PIN, passcodes, identity verification, compromised credentials | `passcode_forgotten`, `verify_my_identity`, `compromised_card` |
| **Technical & System Issues**| Outages, declines, app failures, ATM errors, contactless issues | `card_not_working`, `apple_pay_or_google_pay`, `atm_support` |
| **General Service Inquiry** | Delivery estimates, exchange rates, card ordering, documentation | `card_arrival`, `card_delivery_estimate`, `country_support` |

---

## 5. Priority Labels: Explicit Heuristic Rules (Not Ground Truth)

> [!IMPORTANT]
> **Priority is NOT an ML-learned label or ground truth.**
> In enterprise support operations, ticket priority is a business policy function of SLA entitlements, account ARR, outage severity, and security risk—not an inherent linguistic label of a single customer sentence.

### Why "Priority F1" is Omitted
Computing statistical F1 on rule-derived priority labels is circular and scientifically invalid. We intentionally omit Priority F1 from the benchmark tables.

### Rule-Based Priority Policy Rubric
The app derives priority deterministically in `app/pipeline/classifiers.py` via `determine_priority_heuristic()`:
- **P0 (Critical Blocker / Security Compromise):**
  - Triggered by: Outage keywords (`"outage"`, `"downtime"`, `"system down"`, `"crash"`, `"unreachable"`) or security compromise (`"stolen"`, `"compromised"`, `"fraud"`, `"unauthorized access"`).
  - SLA: < 15–30 minutes (Immediate on-call page).
- **P1 (High Priority / Financial Access Blocker):**
  - Triggered by: Account lockout or financial blocker (`"blocked"`, `"declined"`, `"passcode forgotten"`, `"pin blocked"`, `"locked out"`, `"card swallowed"`) or Enterprise VIP accounts with billing/refund requests.
  - SLA: < 1–2 hours.
- **P2 (Medium Priority / Investigation Required):**
  - Triggered by: Refunds & Disputes category, Technical Issues category, or urgency language (`"urgent"`, `"asap"`, `"wrong amount"`, `"failed"`).
  - SLA: < 4 hours.
- **P3 (Low Priority / Standard Inquiry):**
  - General service questions, exchange rate inquiries, delivery estimates.
  - SLA: < 24 hours.

---

## 6. Adversarial & Edge Cases Suite
The 24 hand-crafted test cases in `evals/adversarial_and_edge_cases.json` are evaluated separately as a unit testing suite for regex prompt-injection interception. They are never mixed into the statistical generalization benchmark.
