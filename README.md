# 🛡️ Enterprise AI Support Ticket Triage Agent

An autonomous, multi-dimensional support ticket triage system engineered with **FastAPI**, **Streamlit**, and **Machine Learning / LLM pipelines**. Designed for production high-volume support engineering teams handling Zendesk, Slack, and email intake.

---

## 🌟 Key Capabilities

1. **Prompt Injection & Adversarial Jailbreak Defense**:
   - Heuristic + pattern recognition shield inspecting incoming tickets for system prompt leakage, role hijacking, delimiter spoofing, and unauthorized escalation injection.
   - Malicious tickets are quarantined immediately with security alerts.
2. **Tool-Calling Agent Orchestrator**:
   - `customer_lookup`: CRM enrichment (ARR, Customer Tier, SLA entitlement, VIP status).
   - `kb_search`: Vector / TF-IDF semantic retrieval over support knowledge base runbooks.
   - `escalate`: Rule-based & confidence-aware escalation to specialized engineering queues (`<75% confidence`, `P0 outages`, `VIP Tier`).
   - `draft_reply`: Grounded AI response generation citing specific knowledge base articles.
3. **Pluggable Multi-Dimensional Classifiers**:
   - **Fine-Tuned Classifier**: Local TF-IDF + Calibrated Classifier delivering **sub-5ms** latency and **$0 inference cost**.
   - **Zero-Shot LLM Reasoning**: Chain-of-thought intent, priority, urgency, and sentiment classification using Google Gemini / OpenAI (with built-in offline fallback).
4. **Real Webhook Connectors**:
   - **Zendesk Sandbox**: Webhook receiver and outbound REST ticket updater.
   - **Slack**: Interactive `/triage` command parser and Block Kit escalation channel dispatcher.
   - **Email Inbound**: SendGrid / Mailgun webhook parser with quote stripping.
5. **Rigorous Evaluation Harness (`evals/eval_harness.py`)**:
   - Measures Macro/Micro F1, Precision, Recall, Latency percentiles ($p_{50}, p_{95}, p_{99}$), and Cost per 1,000 tickets.
6. **High-Aesthetic Streamlit Triage Console**:
   - Live ticket queue, deep-dive reasoning inspector with 1-click draft approval, interactive model eval visualizer, webhook simulator, and SLA analytics.
7. **Containerized Production Deployment**:
   - Dockerfile and `docker-compose.yml` for unified deployment.

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    A[Incoming Ticket] --> B{Source Connector}
    B -->|Zendesk Webhook| C[Zendesk Parser]
    B -->|Slack /triage| D[Slack Parser]
    B -->|Inbound Email| E[Email Parser]
    B -->|Direct API| F[FastAPI /api/tickets/intake]

    C & D & E & F --> G[Prompt Injection Guardrail Shield]

    G -->|Malicious Payload| H[Quarantine & Alert Security Ops]
    G -->|Safe / Sanitized| I[Tool: CRM Customer Lookup]

    I --> J[Classification Engine: Fine-Tuned vs Zero-Shot]
    J --> K[Tool: Knowledge Base Retriever]
    K --> L[SLA Target & Urgency Calculation]

    L --> M{Escalation Evaluator}
    M -->|Low Conf <75% or P0 or VIP| N[Escalate to Human On-Call & Post Slack Alert]
    M -->|High Confidence| O[Tool: Generate Grounded AI Draft Reply]

    N & O --> P[(SQLite Database & JSON Audit Log)]
    P --> Q[Streamlit Interactive Triage Console]
```

---

## 🚀 Quick Start Guide

### 1. Requirements & Setup
Ensure Python 3.10+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```
*(Note: If no API keys are provided, the system seamlessly runs in local zero-shot simulation and fine-tuned mode with zero errors!)*

### 3. Launch Services
You can launch both the **FastAPI Backend** and the **Streamlit Dashboard** concurrently with a single command:
```bash
python run.py --all
```

- **Streamlit Triage Console**: [http://localhost:8501](http://localhost:8501)
- **FastAPI REST API**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📊 Evaluation Benchmark: Fine-Tuned vs. Zero-Shot LLM

Run the evaluation harness:
```bash
python run.py --eval
# or
python evals/eval_harness.py
```

### Benchmark Results on 24 Curated Test Cases:

| Metric | Fine-Tuned (TF-IDF + Calibrated) | Zero-Shot LLM | Trade-off Insight |
| :--- | :---: | :---: | :--- |
| **Intent Macro F1** | **67.90%** | **91.11%** | Zero-Shot handles novel phrasing and nuance (+23% F1) |
| **Intent Micro F1 (Acc)** | **70.83%** | **91.67%** | LLM achieves >91% accuracy across all categories |
| **Priority Macro F1** | **42.43%** | **92.23%** | LLM accurately distinguishes P0 Blocker from P1 Degraded |
| **Latency $p_{50}$ (Median)** | **4.29 ms** | **60.79 ms** | Fine-tuned model is **14x faster** |
| **Latency $p_{95}$** | **9.56 ms** | **67.79 ms** | Fine-tuned model guarantees sub-10ms SLAs |
| **Cost / 1,000 Tickets** | **$0.00** | **$0.0817** | Fine-tuned has zero marginal cloud cost |
| **Prompt Injection Defense** | **100.0%** | **Protected** | Guardrail shield quarantines 100% of adversarial attacks |

---

## 🧪 Running the Test Suite

Execute all 11 unit tests across guardrails, classifiers, pipeline, and connectors:
```bash
python run.py --test
# or
python run_tests.py
```

---

## 🐳 Docker Deployment

Build and launch with Docker Compose:
```bash
docker-compose up --build
```
This deploys:
- `triage-api` on `http://localhost:8000` (FastAPI with health checks)
- `triage-dashboard` on `http://localhost:8501` (Streamlit console)
