"""
Prompt injection and off-topic request guard.

Protects the agent from:
  1. Prompt injection attempts (jailbreak, role override, system prompt leaks)
  2. Off-topic requests (legal, compliance, HR general advice, competitor products)
  3. External tool / non-SHL product recommendations

Returns a RefusalDecision dataclass with reason and suggested response.
Uses fast regex + keyword pattern matching — no LLM call needed here.
"""

import re
from dataclasses import dataclass
from typing import Optional

from agent.prompts import (
    REFUSAL_INJECTION,
    REFUSAL_LEGAL,
    REFUSAL_OFF_TOPIC,
    REFUSAL_EXTERNAL_TOOL,
)
from utils.helpers import get_logger, normalize_text

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Refusal result
# ---------------------------------------------------------------------------

@dataclass
class RefusalDecision:
    should_refuse: bool
    reason: str                  # internal reason code
    response: str                # text to return to user


PASS = RefusalDecision(should_refuse=False, reason="pass", response="")

# ---------------------------------------------------------------------------
# Pattern banks
# ---------------------------------------------------------------------------

# Prompt injection patterns — case-insensitive
_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above|earlier) (instructions?|prompt|context|rules?)",
    r"disregard (all )?(previous|prior|above|earlier)",
    r"you are now",
    r"act as (a |an )?(different|new|another|unrestricted)",
    r"forget (all )?(your )?(previous|prior|above|earlier|instructions?|rules?|constraints?)",
    r"do not follow",
    r"bypass (your )?(rules?|constraints?|instructions?|filters?)",
    r"override (your )?(rules?|constraints?|instructions?|system|prompt)",
    r"jailbreak",
    r"dan mode",
    r"pretend (you are|to be)",
    r"roleplay as",
    r"you have no restrictions",
    r"your (new |real )?(instructions?|rules?|directives?) are",
    r"system prompt",
    r"reveal (your )?(prompt|instructions?|system message|configuration)",
    r"what (are|were) your (instructions?|prompt|system message)",
    r"print (your )?(instructions?|prompt|system message)",
    r"ignore (safety|ethical) (guidelines?|rules?|constraints?)",
    r"### (new )?instruction",
    r"\[system\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Legal / compliance keywords
_LEGAL_PATTERNS = [
    r"\bhipaa\b.*\brequir",
    r"\blegal(ly)?\b.*\brequir",
    r"\bcompl(y|iance|iant)\b.*\brequir",
    r"\brequir\b.*\blaw\b",
    r"\blaw\b.*\brequir",
    r"are we (legally|legally required|required by law)",
    r"does (this|the) (test|assessment) (satisfy|meet|fulfill|fulfil) (a |the )?(legal|regulatory|compliance|hipaa|gdpr|eeoc)",
    r"\bgdpr\b.*\brequir",
    r"\beeoc\b.*\brequir",
    r"\bada\b.*\brequir",
    r"\blegal (obligation|advice|requirement|liability)",
    r"\bcompliance (obligation|advice|requirement)",
    r"\btitle vii\b",
    r"\bemployment law\b",
    r"\bdiscrimination law\b",
    r"\bregulatory (compliance|requirement|obligation)\b",
]

_LEGAL_RE = [re.compile(p, re.IGNORECASE) for p in _LEGAL_PATTERNS]

# General off-topic patterns (non-assessment topics)
_OFFTOPIC_PATTERNS = [
    r"\b(salary|compensation|pay|wage|remuneration)\b",
    r"\b(interview question|interview technique)\b",
    r"\bonboarding\b",
    r"\bperformance review\b",
    r"\bfire (someone|an employee|a staff)\b",
    r"\btermination\b",
    r"\b(lay off|layoff|redundan)\b",
    r"\b(visa|work permit|immigration)\b",
    r"\bbackground check\b",
    r"\bdrug test\b",
    r"\bwrite (a |my )?(resume|cv|cover letter)\b",
    r"\bjob description for\b",  # writing JD is off-topic; reading JD is fine
    r"\bcompetitor\b",
    r"\b(korn ferry|hogan|criteria corp|wonderlic|revelian|arctic shores|pymetrics|hirevue|codility|hackerrank|leetcode|testgorilla|indeed|linkedin)\b",
    r"\bhow (much|many) (does|do) (it|they|shl) (cost|charge|price)\b",
    r"\bpricing\b.*\bshl\b",
    r"\bwhat is (the )?price\b",
]

_OFFTOPIC_RE = [re.compile(p, re.IGNORECASE) for p in _OFFTOPIC_PATTERNS]

# External tool / non-SHL product recommendation requests
_EXTERNAL_TOOL_PATTERNS = [
    r"\b(codility|hackerrank|leetcode|testgorilla|indeed assessments?|linkedin assessments?)\b",
    r"\brecommend (a |an )?(different|alternative|other|third.?party) (tool|platform|vendor|assessment)\b",
    r"\bwhat (other|alternative) (assessment|tool|platform) (vendor|provider)s?\b",
    r"\bcompare shl (with|to|vs\.?) (korn ferry|hogan|criteria|wonderlic)\b",
]

_EXTERNAL_TOOL_RE = [re.compile(p, re.IGNORECASE) for p in _EXTERNAL_TOOL_PATTERNS]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_guards(user_message: str) -> RefusalDecision:
    """
    Run all guards against the user's latest message.
    Returns a RefusalDecision with should_refuse=True if any guard fires,
    or PASS (should_refuse=False) if clean.

    Call order: injection → external tool → legal → off-topic.
    """
    msg = user_message.strip()

    # 1. Prompt injection
    for pattern in _INJECTION_RE:
        if pattern.search(msg):
            _log.warning("Prompt injection detected: %.80s", msg)
            return RefusalDecision(
                should_refuse=True,
                reason="injection",
                response=REFUSAL_INJECTION,
            )

    # 2. External tool recommendation
    for pattern in _EXTERNAL_TOOL_RE:
        if pattern.search(msg):
            _log.info("External tool request detected: %.80s", msg)
            return RefusalDecision(
                should_refuse=True,
                reason="external_tool",
                response=REFUSAL_EXTERNAL_TOOL,
            )

    # 3. Legal / compliance advice
    for pattern in _LEGAL_RE:
        if pattern.search(msg):
            _log.info("Legal/compliance question detected: %.80s", msg)
            return RefusalDecision(
                should_refuse=True,
                reason="legal",
                response=REFUSAL_LEGAL,
            )

    # 4. General off-topic
    for pattern in _OFFTOPIC_RE:
        if pattern.search(msg):
            _log.info("Off-topic request detected: %.80s", msg)
            return RefusalDecision(
                should_refuse=True,
                reason="off_topic",
                response=REFUSAL_OFF_TOPIC,
            )

    return PASS
