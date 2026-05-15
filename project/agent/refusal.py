"""
Refusal engine for out-of-scope requests.

Handles:
  1. Legal / compliance questions
  2. Completely off-topic topics
  3. Prompt injection attempts
  4. External tool recommendations
  5. Questions the agent cannot answer grounded in catalog data

Works in conjunction with guards.py (fast pattern matching).
This module provides higher-level refusal logic and response building.
"""

import re
from typing import Optional

from agent.prompts import (
    REFUSAL_LEGAL,
    REFUSAL_OFF_TOPIC,
    REFUSAL_INJECTION,
    REFUSAL_EXTERNAL_TOOL,
)
from utils.helpers import get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Refusal category detection
# ---------------------------------------------------------------------------

_LEGAL_SIGNALS = re.compile(
    r"(are we (legally|required by law)|legal (requirement|obligation|advice)|"
    r"compliance (requirement|obligation)|does (this|the) (test|assessment) satisfy|"
    r"hipaa.*require|gdpr.*require|eeoc.*require|title vii|discrimination law|"
    r"employment law|regulatory compliance)",
    re.IGNORECASE,
)

_INJECTION_SIGNALS = re.compile(
    r"(ignore (previous|prior) instructions?|you are now|forget your|"
    r"system prompt|jailbreak|bypass (your )?rules?|override (your )?instructions?|"
    r"pretend (you are|to be)|roleplay as|reveal (your )?prompt)",
    re.IGNORECASE,
)

_EXTERNAL_PRODUCT_SIGNALS = re.compile(
    r"(codility|hackerrank|testgorilla|indeed assessment|linkedin assessment|"
    r"korn ferry|hogan assess|wonderlic|arctic shores|pymetrics|hirevue|"
    r"vs\.?\s+(?:korn ferry|hogan|criteria))",
    re.IGNORECASE,
)

_GENERAL_OFFTOPIC_SIGNALS = re.compile(
    r"(salary|how much (does|do)|compensation|interview question|"
    r"write (a |my )?(cv|resume|job description)|termination|fire (someone|employee)|"
    r"background check|drug test|work permit|visa|pricing|cost of shl)",
    re.IGNORECASE,
)


def classify_refusal(message: str) -> Optional[str]:
    """
    Classify whether a message should be refused and why.

    Returns:
        "injection" | "legal" | "external" | "off_topic" | None
    """
    if _INJECTION_SIGNALS.search(message):
        return "injection"
    if _EXTERNAL_PRODUCT_SIGNALS.search(message):
        return "external"
    if _LEGAL_SIGNALS.search(message):
        return "legal"
    if _GENERAL_OFFTOPIC_SIGNALS.search(message):
        return "off_topic"
    return None


def build_refusal_response(
    reason: str,
    original_message: str = "",
    follow_up_hint: Optional[str] = None,
) -> str:
    """
    Build a polite, professional refusal response.

    Args:
        reason: One of "injection" | "legal" | "external" | "off_topic"
        original_message: The original user message (used for context)
        follow_up_hint: Optional hint for what the user could ask instead

    Returns:
        Refusal response string
    """
    base_responses = {
        "injection": REFUSAL_INJECTION,
        "legal": REFUSAL_LEGAL,
        "external": REFUSAL_EXTERNAL_TOOL,
        "off_topic": REFUSAL_OFF_TOPIC,
    }

    response = base_responses.get(reason, REFUSAL_OFF_TOPIC)

    if follow_up_hint:
        response = f"{response} {follow_up_hint}"

    _log.info("Refusal issued. Reason: %s | Message preview: %.60s", reason, original_message)
    return response


def is_vague_request(message: str) -> bool:
    """
    Detect if the user message is too vague to act on without clarification.

    A vague request is one that:
    - Mentions assessments/testing without any role/context
    - Is fewer than 5 meaningful words
    - Contains only generic terms with no hiring signal
    """
    # Very short messages are vague
    words = [w for w in message.split() if len(w) > 2]
    if len(words) < 4:
        return True

    # Generic assessment requests with no role context
    generic_patterns = [
        r"^(?:i need|we need|i want|we want|give me|show me|what|can you)\s+"
        r"(?:a|an|some|the)?\s*(?:assessment|test|solution|tool)\s*\.?$",
        r"^(?:recommend|suggest)\s+(?:a|an|some)?\s*(?:assessment|test|evaluation)\s*\.?$",
        r"^(?:what|which)\s+(?:shl\s+)?(?:assessment|test)s?\s+"
        r"(?:do you have|should i use|are available)\s*\??$",
    ]
    for pat in generic_patterns:
        if re.match(pat, message.strip(), re.IGNORECASE):
            return True

    # Has SOME hiring context (role, level, skill) → not vague
    role_signals = re.compile(
        r"(hiring|hire|recruit|select|assess|screen|evaluat|develop|"
        r"engineer|developer|analyst|manager|director|executive|graduate|"
        r"contact.?cent(er|re)|sales|customer service|java|python|sql|"
        r"leadership|team|department|role|position|job)",
        re.IGNORECASE,
    )
    if role_signals.search(message):
        return False

    return True
