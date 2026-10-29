"""
Multi-Dimensional Classifiers:
1. TfidfCalibratedBaseline: TF-IDF vectorizer + Calibrated Logistic Regression (cv=5), sub-5ms latency, $0 token cost.
2. ZeroShotLLMClassifier: Structured zero-shot LLM reasoning with Google Gemini or OpenAI API.
"""

from __future__ import annotations
import os
import json
import time
from typing import Dict, Any, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

from app.models.ticket import ClassificationResult
from app.core.logging import logger
from app.config import settings

# Prefer public dataset train split, fallback to local dataset
SPLIT_TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/splits/train.json"))
LEGACY_TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../evals/train_dataset.json"))
TRAIN_PATH = SPLIT_TRAIN_PATH if os.path.exists(SPLIT_TRAIN_PATH) else LEGACY_TRAIN_PATH

SUPPORTED_INTENTS = [
    "Billing & Payments",
    "Refunds & Disputes",
    "Account & Authentication",
    "Technical & System Issues",
    "General Service Inquiry"
]

PRIORITY_LEVELS = ["P0", "P1", "P2", "P3"]

# ---------------------------------------------------------------------------
# 1. TF-IDF + Calibrated Linear Baseline
# ---------------------------------------------------------------------------
def determine_priority_heuristic(
    subject: str,
    body: str,
    intent: str,
    customer_tier: str = "Free",
    vip_status: bool = False
) -> str:
    """
    Rule-based heuristic for ticket priority.
    Priority is NOT an ML ground-truth label; it is derived from operational policy signals:
    - Outage/security breach keywords -> P0
    - Financial block / account lockout -> P1
    - Refunds/disputes or customer urgency keywords -> P2
    - Enterprise / VIP status elevates priority
    - General service inquiries -> P3
    """
    text = f"{subject} {body}".lower()

    # 1. P0 Blocker: Outage & Critical Security Compromise
    if any(k in text for k in [
        "outage", "downtime", "system down", "all users affected", "crash",
        "unreachable", "stolen", "compromised", "lost card", "lost phone",
        "identity theft", "fraud", "unauthorized access"
    ]):
        return "P0"

    # 2. P1 High Priority: Financial Blocker / Account Lockout
    if any(k in text for k in [
        "blocked", "declined", "passcode forgotten", "pin blocked",
        "cannot login", "locked out", "not recognised", "unrecognised",
        "card swallowed", "stolen card"
    ]) or (vip_status and intent in ("Refunds & Disputes", "Billing & Payments")):
        return "P1"

    # 3. P2 Medium Priority: Refunds, Disputes & Operational Issues
    if intent in ("Refunds & Disputes", "Technical & System Issues") or any(k in text for k in [
        "refund", "dispute", "failed", "not working", "wrong amount", "urgent", "asap", "error"
    ]):
        return "P2"

    # 4. P3 Low Priority: General Information
    return "P3"

class TfidfCalibratedBaseline:
    """
    TF-IDF + Calibrated Linear Baseline.
    Trains on labeled support tickets using TF-IDF n-grams and Logistic Regression
    calibrated via 5-fold cross-validation (CalibratedClassifierCV) for true posterior probabilities.
    Priority is determined via deterministic rule heuristics, not pseudo-ML labels.
    """
    def __init__(self, training_data_path: str = TRAIN_PATH):
        self.training_data_path = training_data_path
        self.intent_pipeline: Optional[Pipeline] = None
        self.is_trained = False
        self._train()

    def _train(self):
        train_file = self.training_data_path
        if not os.path.exists(train_file):
            if os.path.exists(LEGACY_TRAIN_PATH):
                train_file = LEGACY_TRAIN_PATH
            else:
                logger.warning(f"Training dataset not found at {train_file}. Skipping baseline training.")
                return

        # Check for serialized cache
        cache_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/models"))
        cache_path = os.path.join(cache_dir, "tfidf_baseline.joblib")
        if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= os.path.getmtime(train_file):
            try:
                import joblib
                self.intent_pipeline = joblib.load(cache_path)
                self.is_trained = True
                logger.info(f"Loaded cached TF-IDF calibrated baseline from {cache_path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached model from {cache_path}: {e}")

        try:
            with open(train_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            texts = [f"{item.get('subject', '')} {item.get('body', '')}".strip() for item in data]
            intent_labels = [item.get("intent") or item.get("ground_truth_intent", "General Service Inquiry") for item in data]

            # Intent Classifier Pipeline: TF-IDF + 5-fold Calibrated Logistic Regression
            base_intent_lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
            calibrated_intent = CalibratedClassifierCV(estimator=base_intent_lr, cv=min(5, len(data)), method="sigmoid")
            
            self.intent_pipeline = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
                ("clf", calibrated_intent)
            ])
            self.intent_pipeline.fit(texts, intent_labels)

            self.is_trained = True
            logger.info(f"TF-IDF + calibrated linear baseline successfully trained on {len(data)} examples from {os.path.basename(train_file)}.")

            # Persist to disk cache
            try:
                import joblib
                os.makedirs(cache_dir, exist_ok=True)
                joblib.dump(self.intent_pipeline, cache_path)
            except Exception as ce:
                logger.warning(f"Failed to cache baseline model: {ce}")
        except Exception as e:
            logger.error(f"Error training TF-IDF calibrated baseline: {e}", exc_info=True)

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Returns calibrated posterior probability distribution across intent classes."""
        if not self.is_trained or not self.intent_pipeline:
            return {}
        probs = self.intent_pipeline.predict_proba([text])[0]
        classes = self.intent_pipeline.classes_
        return {str(cls): float(prob) for cls, prob in zip(classes, probs)}

    def classify(self, subject: str, body: str, customer_tier: str = "Free", vip_status: bool = False) -> ClassificationResult:
        start_time = time.time()
        text = f"{subject} {body}".strip()

        if not self.is_trained or not text:
            return ClassificationResult(
                intent="General Service Inquiry",
                priority="P3",
                urgency_score=0.2,
                sentiment="NEUTRAL",
                confidence=0.50,
                reasoning="Default classification (model untrained or blank text)",
                classifier_type="tfidf-calibrated-baseline",
                token_usage={"prompt_tokens": 0, "completion_tokens": 0}
            )

        # Predict intent + calibrated posterior probability
        intent_probs = self.intent_pipeline.predict_proba([text])[0]
        intent_classes = self.intent_pipeline.classes_
        top_intent_idx = np.argmax(intent_probs)
        predicted_intent = intent_classes[top_intent_idx]
        calibrated_intent_confidence = float(intent_probs[top_intent_idx])

        # Priority is derived via explicit heuristic rule (signals: outage, urgency, tier, intent)
        predicted_priority = determine_priority_heuristic(subject, body, predicted_intent, customer_tier, vip_status)

        # Heuristic Sentiment & Urgency based on keywords
        sentiment, urgency = self._extract_sentiment_and_urgency(text, predicted_priority)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        reasoning = (
            f"TF-IDF + calibrated linear model predicted '{predicted_intent}' with "
            f"{calibrated_intent_confidence*100:.1f}% calibrated confidence. "
            f"Priority '{predicted_priority}' assigned via rule heuristic (not ground truth)."
        )

        return ClassificationResult(
            intent=predicted_intent,
            priority=predicted_priority,
            urgency_score=urgency,
            sentiment=sentiment,
            confidence=round(calibrated_intent_confidence, 3),
            reasoning=reasoning,
            classifier_type="tfidf-calibrated-baseline",
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )

    def _extract_sentiment_and_urgency(self, text: str, priority: str) -> Tuple[str, float]:
        text_lower = text.lower()
        if any(w in text_lower for w in ["unacceptable", "furious", "angry", "broken completely", "lawsuit", "refund immediately"]):
            sentiment = "ANGRY"
        elif any(w in text_lower for w in ["frustrated", "blocked", "failing", "error", "urgent", "down", "stuck"]):
            sentiment = "FRUSTRATED"
        elif any(w in text_lower for w in ["thank you", "great", "love", "awesome", "appreciate"]):
            sentiment = "POSITIVE"
        else:
            sentiment = "NEUTRAL"

        urgency_map = {"P0": 0.95, "P1": 0.80, "P2": 0.55, "P3": 0.25}
        base_urgency = urgency_map.get(priority, 0.5)
        if "urgent" in text_lower or "emergency" in text_lower:
            base_urgency = min(1.0, base_urgency + 0.15)
            
        return sentiment, round(base_urgency, 2)

# Backward-compatibility alias
FineTunedClassifier = TfidfCalibratedBaseline


# ---------------------------------------------------------------------------
# 2. Zero-Shot LLM Classifier
# ---------------------------------------------------------------------------
class ZeroShotLLMClassifier:
    """
    Zero-Shot LLM Classifier using structured reasoning prompts.
    Requires GEMINI_API_KEY or OPENAI_API_KEY for live execution.
    """
    def __init__(self):
        self.gemini_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        self.openai_key = settings.OPENAI_API_KEY or os.environ.get("OPENAI_API_KEY")
        self.model_name = settings.LLM_MODEL

    @property
    def has_api_key(self) -> bool:
        return bool(self.gemini_key or self.openai_key)

    def classify(self, subject: str, body: str, require_live: bool = False, **kwargs) -> ClassificationResult:
        start_time = time.time()
        combined_text = f"Subject: {subject}\nBody: {body}".strip()

        # Check for live API execution
        if self.gemini_key and settings.LLM_PROVIDER == "gemini":
            try:
                return self._call_gemini(combined_text, start_time)
            except Exception as e:
                logger.error(f"Live Gemini API call failed: {e}")
                if require_live:
                    raise
        elif self.openai_key and settings.LLM_PROVIDER == "openai":
            try:
                return self._call_openai(combined_text, start_time)
            except Exception as e:
                logger.error(f"Live OpenAI API call failed: {e}")
                if require_live:
                    raise

        if require_live:
            raise RuntimeError("No LLM API key configured (set OPENAI_API_KEY or GEMINI_API_KEY).")

        # Interactive fallback for local UI testing when no API key is set
        return self._interactive_demo_fallback(subject, body, start_time)

    def _call_gemini(self, text: str, start_time: float) -> ClassificationResult:
        from google import genai
        client = genai.Client(api_key=self.gemini_key)
        model_name = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")
        if model_name in ("gemini-1.5-flash", "gemini-2.5-flash"):
            model_name = "gemini-3.5-flash-lite"
        
        prompt = self._build_prompt(text)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        content = response.text.strip() if response.text else "{}"
        
        result_dict = self._parse_json_response(content)
        prompt_tokens = int(len(prompt) / 4)
        completion_tokens = int(len(content) / 4)
        
        return ClassificationResult(
            intent=result_dict.get("intent", "General Service Inquiry"),
            priority=result_dict.get("priority", "P3"),
            urgency_score=float(result_dict.get("urgency_score", 0.5)),
            sentiment=result_dict.get("sentiment", "NEUTRAL"),
            confidence=float(result_dict.get("confidence", 0.90)),
            reasoning=result_dict.get("reasoning", f"Live Gemini Zero-Shot Classification ({model_name})"),
            classifier_type="zero-shot-llm",
            token_usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        )

    def _call_openai(self, text: str, start_time: float) -> ClassificationResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        prompt = self._build_prompt(text)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an enterprise support triage classifier. Respond only in valid JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content or "{}"
        result_dict = json.loads(content)
        usage = response.usage
        
        return ClassificationResult(
            intent=result_dict.get("intent", "General Service Inquiry"),
            priority=result_dict.get("priority", "P3"),
            urgency_score=float(result_dict.get("urgency_score", 0.5)),
            sentiment=result_dict.get("sentiment", "NEUTRAL"),
            confidence=float(result_dict.get("confidence", 0.92)),
            reasoning=result_dict.get("reasoning", "Live OpenAI Zero-Shot Classification"),
            classifier_type="zero-shot-llm",
            token_usage={"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        )

    def _interactive_demo_fallback(self, subject: str, body: str, start_time: float) -> ClassificationResult:
        """Demo placeholder used strictly when exploring UI without API keys."""
        text_lower = f"{subject} {body}".lower()
        if any(k in text_lower for k in ["invoice", "charge", "billing", "card", "payment"]):
            intent = "Billing & Payments"
        elif any(k in text_lower for k in ["refund", "money back", "dispute", "return"]):
            intent = "Refunds & Disputes"
        elif any(k in text_lower for k in ["cancel", "subscription", "login", "password", "pin", "verify"]):
            intent = "Account & Authentication"
        elif any(k in text_lower for k in ["error", "issue", "bug", "crash", "down", "outage", "500", "transfer"]):
            intent = "Technical & System Issues"
        else:
            intent = "General Service Inquiry"

        priority = "P0" if any(w in text_lower for w in ["outage", "critical", "downtime", "unreachable"]) else "P2"
        return ClassificationResult(
            intent=intent,
            priority=priority,
            urgency_score=0.5,
            sentiment="NEUTRAL",
            confidence=0.75,
            reasoning="Demo placeholder fallback (live LLM API key not configured)",
            classifier_type="zero-shot-llm-demo-fallback",
            token_usage={"prompt_tokens": 0, "completion_tokens": 0}
        )

    def _build_prompt(self, text: str) -> str:
        return f"""You are an enterprise AI support ticket triage classifier.
Analyze the following support ticket and classify it into multi-dimensional categories.

Candidate Intents: {json.dumps(SUPPORTED_INTENTS)}
Candidate Priorities: ["P0", "P1", "P2", "P3"]
Candidate Sentiments: ["POSITIVE", "NEUTRAL", "FRUSTRATED", "ANGRY"]

Return pure JSON matching this schema:
{{
  "intent": "<string>",
  "priority": "<P0|P1|P2|P3>",
  "urgency_score": <float 0.0 to 1.0>,
  "sentiment": "<string>",
  "confidence": <float 0.0 to 1.0>,
  "reasoning": "<concise explanation>"
}}

Ticket Content:
{text}
"""

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        try:
            clean = text.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            return json.loads(clean.strip())
        except Exception:
            return {}
