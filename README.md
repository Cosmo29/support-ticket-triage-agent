# Support Ticket Triage Agent

A FastAPI service that triages support tickets (intent, priority, sentiment), retrieves relevant knowledge-base articles, drafts grounded replies for human review, and escalates low-confidence or high-risk tickets. Includes a prompt-injection filter, a Streamlit console, and an evaluation harness comparing a classical ML baseline against an LLM classifier.

> **Status:** work in progress. Connectors (Zendesk, Slack, email) are tested with simulated payloads only. The LLM column in the benchmark below was produced by a keyword-based simulator, not a real LLM, and is being replaced. The TF-IDF classifier is trained on a 20-row synthetic set.

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
   - **TF-IDF + logistic regression baseline**: Local TF-IDF + Logistic Regression model delivering **sub-5ms** latency and **$0 inference cost**.
   - **Zero-Shot LLM Reasoning**: Chain-of-thought intent, priority, urgency, and sentiment classification using Google Gemini / OpenAI (with built-in offline fallback).
4. **Webhook connectors (simulated payloads)**:
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

    I --> J[Classification Engine: TF-IDF Baseline vs Zero-Shot]
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

## 📊 Evaluation Benchmark: TF-IDF Baseline vs. Zero-Shot LLM

Run the evaluation harness:
```bash
python run.py --eval
# or
python evals/eval_harness.py
```

Results being regenerated with real data.

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
