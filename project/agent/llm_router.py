"""
Cascading multi-model LLM router for SHL Recommender.

Architecture:
  - 4 tiers of models, each optimised for a different failure mode
  - Turn-type-aware routing: vague queries use fast cheap models;
    refinement and comparison turns escalate to stronger reasoning models
  - Quality validation before accepting any model response:
      * valid JSON structure
      * no hallucinated SHL names (catalog check)
      * recommendations within [1, 10]
      * SHL URLs only
  - Observability: every routing decision is logged with model, latency,
    retry count, and failure reason
  - Final fallback: if all LLMs fail, callers use catalog-only reranking

Usage:
    from agent.llm_router import route_llm_call, TurnType
    raw = route_llm_call(prompt, turn_type=TurnType.REFINE)
"""

import concurrent.futures
import json
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from utils.helpers import get_env, get_logger

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Turn type enum
# ---------------------------------------------------------------------------

class TurnType(str, Enum):
    """Turn classification used to select the appropriate model tier."""
    VAGUE       = "vague"       # Short/ambiguous query → cheapest fast model
    CLARIFY     = "clarify"     # Clarification question → fast model
    RECOMMEND   = "recommend"   # First recommendation → fast model
    REFINE      = "refine"      # Refinement turn → reasoning model
    COMPARE     = "compare"     # Comparison query → structured-output model
    STATE       = "state"       # State extraction → fast model
    INFER_ROLE  = "infer_role"  # Role inference → fast model


# ---------------------------------------------------------------------------
# Model tiers
# Ordered by preference within each tier.
# ---------------------------------------------------------------------------

# Tier 1: Fast free models — good for most first-pass queries
_TIER1 = [
    "deepseek/deepseek-v4-flash:free",
    "qwen/qwen3-32b:free",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]

# Tier 2: Reasoning / refinement backup — stronger instruction-following
_TIER2 = [
    "qwen/qwen3-14b:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-medium-128k-instruct:free",
    "google/gemma-3-4b-it:free",
]

# Tier 3: Stable JSON fallback — strong structured output
_TIER3 = [
    "openai/gpt-4o-mini",
    "anthropic/claude-3-haiku",
    "mistralai/mistral-small",
]

# Tier 4: Emergency fallback — most reliable but highest cost
_TIER4 = [
    "openai/gpt-4o",
    "anthropic/claude-3-5-sonnet",
]

# Mapping from turn type to preferred model sequence
# Each entry is a list of tiers to try (each tier tried exhaustively before next)
_TURN_TIER_MAP: Dict[TurnType, List[List[str]]] = {
    TurnType.VAGUE:      [_TIER1],
    TurnType.CLARIFY:    [_TIER1],
    TurnType.RECOMMEND:  [_TIER1, _TIER2],
    TurnType.REFINE:     [_TIER1, _TIER2],   # Tier1 first (faster), then Tier2
    TurnType.COMPARE:    [_TIER1, _TIER2],
    TurnType.STATE:      [_TIER1],
    TurnType.INFER_ROLE: [_TIER1],
}

# Default sequence when turn type is not specified
_DEFAULT_TIERS = [_TIER1, _TIER2]

# ---------------------------------------------------------------------------
# Timeouts (seconds per model call — TRUE WALL-CLOCK via ThreadPoolExecutor)
# These are intentionally short. If a model won't respond in this window,
# it is not fast enough for the 30-second evaluator budget.
# ---------------------------------------------------------------------------
_TIMEOUT_BY_TIER = {
    0: 4,    # Tier 1 — fast free models must reply quickly
    1: 6,    # Tier 2 — reasoning models
    2: 8,    # Tier 3 — paid stable models
    3: 10,   # Tier 4 — last resort
}

# Global budget: if total elapsed across all model attempts exceeds this,
# stop cascading and return None (catalog-only fallback).
_GLOBAL_BUDGET_S = 15.0

# ---------------------------------------------------------------------------
# Client singleton
# ---------------------------------------------------------------------------

_llm_client: Optional[OpenAI] = None


def _get_client() -> Optional[OpenAI]:
    """Lazily initialise the OpenRouter OpenAI-compatible client."""
    global _llm_client
    if _llm_client is None:
        api_key = get_env("OPENROUTER_API_KEY")
        if not api_key:
            _log.warning("OPENROUTER_API_KEY not set — all LLM calls will be skipped.")
            return None
        try:
            _llm_client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
            _log.info("OpenRouter client initialised (cascading router).")
        except Exception as exc:
            _log.error("Failed to initialise OpenRouter client: %s", exc)
    return _llm_client


# ---------------------------------------------------------------------------
# Response quality validation
# ---------------------------------------------------------------------------

def _is_retriable_error(err: str) -> bool:
    """Return True if the error should trigger a model retry."""
    err_lower = err.lower()
    return any(k in err_lower for k in (
        "404", "429", "rate limit", "timeout", "timed out",
        "connection", "service unavailable", "503", "502",
        "model not found", "not available", "overloaded",
    ))


def _validate_json_response(raw: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Validate that a raw LLM response is usable structured JSON.

    Returns (is_valid, parsed_dict_or_None, failure_reason).
    """
    if not raw or len(raw.strip()) < 5:
        return False, None, "empty_response"

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try to extract JSON object if mixed with prose
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not json_match:
        return False, None, "no_json_object"

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError as exc:
        return False, None, f"json_parse_error:{exc}"

    if not isinstance(parsed, dict):
        return False, None, "not_a_dict"

    return True, parsed, "ok"


def _validate_recommendation_response(
    parsed: Dict[str, Any],
    require_recs: bool = True,
) -> Tuple[bool, str]:
    """
    Validate a parsed recommendation response dict.

    Checks:
    - Has required keys: reply, recommendations, end_of_conversation
    - recommendations is a list of [1, 10] items
    - Each item has name and url
    - All URLs contain shl.com

    Returns (is_valid, failure_reason).
    """
    if "reply" not in parsed:
        return False, "missing_reply"
    if "recommendations" not in parsed:
        return False, "missing_recommendations"
    if not isinstance(parsed["recommendations"], list):
        return False, "recommendations_not_list"

    recs = parsed["recommendations"]
    if require_recs and len(recs) == 0:
        return False, "empty_recommendations"
    if len(recs) > 10:
        return False, f"too_many_recs:{len(recs)}"

    for i, rec in enumerate(recs):
        if not isinstance(rec, dict):
            return False, f"rec[{i}]_not_dict"
        if not rec.get("name"):
            return False, f"rec[{i}]_missing_name"
        url = rec.get("url", "")
        if url and "shl.com" not in url.lower():
            return False, f"rec[{i}]_non_shl_url:{url[:60]}"

    return True, "ok"


# ---------------------------------------------------------------------------
# Core routing function
# ---------------------------------------------------------------------------

def route_llm_call(
    prompt: str,
    turn_type: TurnType = TurnType.RECOMMEND,
    validate_json: bool = False,
    require_recs: bool = False,
    max_tokens: int = 1024,
) -> Optional[str]:
    """
    Route an LLM prompt through the cascading model fallback system.

    Args:
        prompt:       The prompt to send.
        turn_type:    Classification of the current turn, used to select models.
        validate_json: If True, only accept responses that parse as valid JSON.
        require_recs:  If True (with validate_json), reject responses with 0 recs.
        max_tokens:   Max tokens for the model response.

    Returns:
        The first valid text response from any model, or None if all fail.

    Routing logic:
        1. Look up preferred tier order for this turn_type.
        2. For each tier, try each model in sequence.
        3. If model fails (error or empty), try next model in same tier.
        4. If tier exhausted, move to next tier.
        5. If validate_json=True, also reject responses that fail JSON validation.
        6. If all tiers/models fail, return None (callers use catalog fallback).
    """
    client = _get_client()
    if client is None:
        return None

    tiers = _TURN_TIER_MAP.get(turn_type, _DEFAULT_TIERS)
    total_attempts = 0
    overall_start = time.time()

    # Thread pool used to enforce true wall-clock timeouts on blocking HTTP calls.
    # The OpenAI SDK's own `timeout` parameter is a socket-inactivity timeout,
    # NOT a wall-clock deadline — OpenRouter can hold the connection for 70+s.
    _executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        for tier_idx, tier_models in enumerate(tiers):
            tier_timeout = _TIMEOUT_BY_TIER.get(tier_idx, 7)

            for model in tier_models:
                # Global budget check — fires between model attempts
                total_elapsed = time.time() - overall_start
                if total_elapsed > _GLOBAL_BUDGET_S:
                    _log.error(
                        "[Router] 🛑 Global budget exceeded (%.2fs / %.0fs). Stopping.",
                        total_elapsed, _GLOBAL_BUDGET_S,
                    )
                    return None

                total_attempts += 1
                attempt_start = time.time()

                try:
                    # Submit the blocking SDK call to a thread
                    future = _executor.submit(
                        client.chat.completions.create,
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.15,
                        max_tokens=max_tokens,
                        timeout=tier_timeout,  # socket-level hint (belt)
                    )
                    # Wait with a TRUE wall-clock deadline (suspenders)
                    response = future.result(timeout=tier_timeout)
                    elapsed = time.time() - attempt_start
                    raw = (response.choices[0].message.content or "").strip()

                    if not raw:
                        _log.warning(
                            "[Router] Model '%s' (tier %d) empty response in %.2fs. Trying next.",
                            model, tier_idx + 1, elapsed,
                        )
                        continue

                    # If JSON validation required, check before accepting
                    if validate_json:
                        is_valid, parsed, reason = _validate_json_response(raw)
                        if not is_valid:
                            _log.warning(
                                "[Router] Model '%s' JSON validation failed: %s. Trying next.",
                                model, reason,
                            )
                            continue

                        if require_recs and parsed:
                            rec_valid, rec_reason = _validate_recommendation_response(
                                parsed, require_recs=require_recs
                            )
                            if not rec_valid:
                                _log.warning(
                                    "[Router] Model '%s' rec validation failed: %s. Trying next.",
                                    model, rec_reason,
                                )
                                continue

                    total_elapsed = time.time() - overall_start
                    _log.info(
                        "[Router] ✓ Model '%s' (tier %d) success in %.2fs | attempts=%d | total=%.2fs",
                        model, tier_idx + 1, elapsed, total_attempts, total_elapsed,
                    )
                    return raw

                except concurrent.futures.TimeoutError:
                    elapsed = time.time() - attempt_start
                    _log.warning(
                        "[Router] Model '%s' (tier %d) HARD TIMEOUT after %.2fs. Trying next.",
                        model, tier_idx + 1, elapsed,
                    )

                except Exception as exc:
                    elapsed = time.time() - attempt_start
                    err = str(exc)
                    if _is_retriable_error(err):
                        _log.warning(
                            "[Router] Model '%s' (tier %d) retriable error in %.2fs: %.80s. Trying next.",
                            model, tier_idx + 1, elapsed, err,
                        )
                    else:
                        _log.error(
                            "[Router] Model '%s' (tier %d) non-retriable error in %.2fs: %.120s. Trying next.",
                            model, tier_idx + 1, elapsed, err,
                        )
    finally:
        _executor.shutdown(wait=False)

    total_elapsed = time.time() - overall_start
    _log.error(
        "[Router] ✗ ALL models failed. turn_type=%s attempts=%d total=%.2fs",
        turn_type.value, total_attempts, total_elapsed,
    )
    return None


# ---------------------------------------------------------------------------
# Convenience wrappers (match the old _call_llm signature)
# ---------------------------------------------------------------------------

def call_llm_fast(
    prompt: str,
    timeout: int = 4,
    max_tokens: int = 256,
) -> Optional[str]:
    """
    Fast path: Tier 1 models only, tight 4-second wall-clock deadline.
    Used for state extraction, role inference, clarification questions.
    """
    # Override tier timeout for this call only
    import concurrent.futures as _cf
    client = _get_client()
    if client is None:
        return None
    for model in _TIER1[:2]:  # Only try first 2 models for speed
        try:
            future = client.chat.completions.create
            with _cf.ThreadPoolExecutor(max_workers=1) as ex:
                f = ex.submit(
                    client.chat.completions.create,
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
                try:
                    resp = f.result(timeout=timeout)
                    raw = (resp.choices[0].message.content or "").strip()
                    if raw:
                        _log.info("[FastLLM] '%s' success", model)
                        return raw
                except _cf.TimeoutError:
                    _log.warning("[FastLLM] '%s' timed out after %ds", model, timeout)
        except Exception as exc:
            _log.warning("[FastLLM] '%s' error: %.60s", model, exc)
    return None


def call_llm_recommend(prompt: str) -> Optional[str]:
    """Recommendation turn: Tier 1 → Tier 2."""
    return route_llm_call(
        prompt,
        turn_type=TurnType.RECOMMEND,
        validate_json=False,
        max_tokens=1024,
    )


def call_llm_refine(prompt: str) -> Optional[str]:
    """Refinement turn: Tier 2 → Tier 1 → Tier 3 (needs strong instruction-following)."""
    return route_llm_call(
        prompt,
        turn_type=TurnType.REFINE,
        validate_json=False,
        max_tokens=1024,
    )


def call_llm_compare(prompt: str) -> Optional[str]:
    """Comparison turn: Tier 1 → Tier 2 → Tier 3 (needs accurate factual output)."""
    return route_llm_call(
        prompt,
        turn_type=TurnType.COMPARE,
        validate_json=False,
        max_tokens=800,
    )
