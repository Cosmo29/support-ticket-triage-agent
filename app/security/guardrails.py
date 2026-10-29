"""
Prompt Injection Defense & Input Sanitization Guardrail.
Protects the AI Support Ticket Triage Agent from adversarial jailbreaks,
system prompt leakage, delimiter hijacking, unauthorized tool calling, and exfiltration attempts.
"""

from __future__ import annotations
import re
import html
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field

class GuardrailAssessment(BaseModel):
    is_safe: bool = True
    threat_level: str = "CLEAN"  # "CLEAN", "SUSPICIOUS", "MALICIOUS"
    risk_score: float = Field(default=0.0, ge=0.0, le=1.0)
    detected_patterns: List[str] = Field(default_factory=list)
    sanitized_text: str = ""
    action: str = "ALLOW"  # "ALLOW", "FLAG_SUSPICIOUS", "QUARANTINE"

# High-confidence adversarial injection signatures
PROMPT_INJECTION_PATTERNS: List[Tuple[str, str, float]] = [
    # 1. Direct instruction overrides
    (r"(?i)\bignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions|prompts|rules|commands|directives)", "INSTRUCTION_OVERRIDE", 0.95),
    (r"(?i)\bdisregard\s+(all\s+)?(prior|previous|existing)\s+(constraints|rules|guidelines)", "INSTRUCTION_OVERRIDE", 0.95),
    (r"(?i)\byou\s+are\s+now\s+(an?\s+)?(unrestricted|unfiltered|jailbroken|DAN|developer\s+mode|root|admin)", "ROLE_HIJACKING", 0.95),
    (r"(?i)\bpretend\s+you\s+have\s+no\s+(rules|restrictions|limits|guidelines)", "ROLE_HIJACKING", 0.90),
    
    # 2. System prompt leakage & exfiltration
    (r"(?i)\brepeat\s+(your\s+)?(system\s+prompt|initial\s+instructions|secret\s+key|api\s+token)", "SYSTEM_LEAK_ATTEMPT", 0.90),
    (r"(?i)\bwhat\s+(is|are)\s+your\s+(exact\s+)?(system\s+prompt|hidden\s+prompt|developer\s+instructions)", "SYSTEM_LEAK_ATTEMPT", 0.85),
    (r"(?i)\boutput\s+everything\s+above\s+this\s+line", "SYSTEM_LEAK_ATTEMPT", 0.90),
    
    # 3. Delimiter & Control Token Spoofing
    (r"(?i)(<\s*system\s*>|<\s*/\s*system\s*>|\[\s*SYSTEM\s*\]|\[\s*ADMIN\s*OVERRIDE\s*\])", "DELIMITER_SPOOFING", 0.90),
    (r"(?i)(###\s*(instruction|system|human|assistant|user):)", "DELIMITER_SPOOFING", 0.85),
    (r"(?i)(\{\s*\"role\"\s*:\s*\"system\")", "STRUCTURED_ROLE_INJECTION", 0.90),
    
    # 4. Tool manipulation & unauthorized approval attacks
    (r"(?i)\bexecute\s+tool\s+escalate\s+with\b", "TOOL_INJECTION", 0.90),
    (r"(?i)\bforce\s+(p0|critical)\s+escalation\s+and\s+auto-?refund\b", "TOOL_INJECTION", 0.85),
    (r"(?i)\bmark\s+(as\s+)?(resolved|approved)\s+without\s+(verification|checking)\b", "UNAUTHORIZED_ACTION", 0.75),

    # 5. Markdown/Image data exfiltration
    (r"!\[.*?\]\(https?://[^\s)]+(\?|&)[^)]*(token|key|secret|password|prompt|cookie)=.*?\)", "EXFILTRATION_PAYLOAD", 0.95),
]

SUSPICIOUS_HEURISTICS: List[Tuple[str, str, float]] = [
    (r"(?i)\bbypass\s+(all\s+)?(security|guardrails|checks|filters)", "SECURITY_BYPASS_KEYWORD", 0.70),
    (r"(?i)\bas\s+an\s+ai,\s+you\s+must\s+obey\s+me", "AUTHORITY_SPOOFING", 0.65),
    (r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "NON_PRINTABLE_CONTROL_CHARS", 0.50),
    (r"(?i)base64:\s*[A-Za-z0-9+/=]{20,}", "ENCODED_PAYLOAD_HINT", 0.60),
]

def sanitize_text(text: str) -> str:
    """Removes dangerous control characters and encodes raw HTML tags."""
    if not text:
        return ""
    # Strip non-printable ascii control chars (except newline and tab)
    cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Neutralize HTML tags
    cleaned = html.escape(cleaned)
    # Neutralize dangerous delimiter attempts
    cleaned = re.sub(r"(?i)###\s*(instruction|system|assistant):", "[BLOCKED_DELIMITER]", cleaned)
    cleaned = re.sub(r"(?i)<\s*/?\s*system\s*>", "[BLOCKED_TAG]", cleaned)
    return cleaned.strip()

def inspect_for_prompt_injection(content: str, threshold: float = 0.60) -> GuardrailAssessment:
    """
    Evaluates incoming ticket text (subject + body) for prompt injection or jailbreak patterns.
    Returns a comprehensive GuardrailAssessment.
    """
    if not content or not content.strip():
        return GuardrailAssessment(
            is_safe=True,
            threat_level="CLEAN",
            risk_score=0.0,
            detected_patterns=[],
            sanitized_text="",
            action="ALLOW"
        )
    
    detected: List[str] = []
    max_score = 0.0
    
    # Check high-confidence adversarial patterns
    for pattern, name, score in PROMPT_INJECTION_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            detected.append(f"{name} (pattern match)")
            if score > max_score:
                max_score = score
                
    # Check suspicious heuristic signals
    for pattern, name, score in SUSPICIOUS_HEURISTICS:
        matches = re.findall(pattern, content)
        if matches:
            detected.append(f"{name} (heuristic signal)")
            if score > max_score:
                max_score = score

    # Determine threat level and action
    if max_score >= 0.80 or len(detected) >= 2:
        threat_level = "MALICIOUS"
        action = "QUARANTINE"
        is_safe = False
    elif max_score >= threshold or len(detected) == 1:
        threat_level = "SUSPICIOUS"
        action = "FLAG_SUSPICIOUS"
        is_safe = False
    else:
        threat_level = "CLEAN"
        action = "ALLOW"
        is_safe = True

    sanitized = sanitize_text(content)

    return GuardrailAssessment(
        is_safe=is_safe,
        threat_level=threat_level,
        risk_score=round(max_score, 3),
        detected_patterns=list(set(detected)),
        sanitized_text=sanitized,
        action=action
    )
