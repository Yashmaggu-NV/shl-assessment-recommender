"""
Grounded comparison engine for SHL assessments.

Handles user requests like:
  "What is the difference between OPQ and GSA?"
  "Is the Contact Center Simulation different from the Customer Service Phone Simulation?"

Strategy:
  1. Parse both assessment names from the user message
  2. Look them up in the catalog (fuzzy match)
  3. Build a structured diff using catalog metadata only
  4. Pass to LLM with a strict comparison prompt (no hallucination)
  5. Return grounded textual answer

Never adds information not present in the catalog.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from utils.helpers import (
    KEY_TO_CODE,
    get_logger,
    normalize_text,
    load_catalog,
)
from agent.retriever import get_item_by_name, keyword_search

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Name extraction patterns
# ---------------------------------------------------------------------------

# Patterns for "difference between X and Y" or "X vs Y" or "compare X and Y"
_COMPARE_PATTERNS = [
    r"(?:difference|diff|compare|comparison)\s+between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
    r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
    r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$)",
    r"(?:how (?:is|are)|what (?:is|are)) (.+?) different (?:from|to|than) (.+?)(?:\?|$)",
    r"(.+?) (?:or|versus) (.+?)\s*\?",
]

_COMPARE_RE = [re.compile(p, re.IGNORECASE) for p in _COMPARE_PATTERNS]


def is_comparison_request(message: str) -> bool:
    """Return True if the message appears to be a comparison request."""
    indicators = [
        r"\bdifference\b", r"\bcompare\b", r"\bvs\.?\b", r"\bversus\b",
        r"\bcontrast\b", r"\bsame as\b", r"\bdifferent from\b",
    ]
    for pattern in indicators:
        if re.search(pattern, message, re.IGNORECASE):
            return True
    return False


def extract_comparison_names(message: str) -> Optional[Tuple[str, str]]:
    """
    Extract the two assessment names being compared.

    Returns a tuple of (name_a, name_b) or None if extraction fails.
    """
    for pattern in _COMPARE_RE:
        match = pattern.search(message)
        if match:
            a = match.group(1).strip().strip("\"'")
            b = match.group(2).strip().strip("\"'")
            if a and b and len(a) > 2 and len(b) > 2:
                return a, b
    return None


# ---------------------------------------------------------------------------
# Catalog lookup helpers
# ---------------------------------------------------------------------------

def _find_assessment(query: str) -> Optional[Dict[str, Any]]:
    """
    Find a catalog assessment by name using exact-then-fuzzy matching.
    Falls back to keyword search if direct name matching fails.
    """
    # Direct name lookup
    item = get_item_by_name(query)
    if item:
        return item

    # Keyword search fallback
    results = keyword_search(query, top_k=3)
    if results:
        return results[0][0]

    return None


def _format_item_for_comparison(item: Dict[str, Any]) -> str:
    """
    Render a catalog item as a structured text block for LLM comparison.
    """
    codes = [KEY_TO_CODE.get(k, k) for k in item.get("keys", [])]
    type_str = ", ".join(codes) if codes else "Unknown"

    langs = item.get("languages", [])
    lang_str = ", ".join(langs[:5])
    if len(langs) > 5:
        lang_str += f" (+{len(langs) - 5} more)"
    if not lang_str:
        lang_str = "Not specified"

    levels = item.get("job_levels", [])
    levels_str = ", ".join(levels) if levels else "Not specified"

    duration = item.get("duration") or "Not specified"
    description = item.get("description") or "No description available."
    remote = item.get("remote", "unknown")
    adaptive = item.get("adaptive", "unknown")

    return f"""Name: {item.get('name', 'Unknown')}
Type codes: {type_str}
Categories: {', '.join(item.get('keys', []))}
Duration: {duration}
Job levels: {levels_str}
Languages: {lang_str}
Remote: {remote}
Adaptive: {adaptive}
URL: {item.get('link', 'N/A')}
Description: {description}"""


def _build_grounded_comparison(
    item_a: Dict[str, Any],
    item_b: Dict[str, Any],
) -> str:
    """
    Build a structured, grounded comparison reply without LLM if needed.
    Used as fallback when LLM is unavailable.
    """
    name_a = item_a.get("name", "Assessment A")
    name_b = item_b.get("name", "Assessment B")

    codes_a = set(KEY_TO_CODE.get(k) for k in item_a.get("keys", []) if KEY_TO_CODE.get(k))
    codes_b = set(KEY_TO_CODE.get(k) for k in item_b.get("keys", []) if KEY_TO_CODE.get(k))

    dur_a = item_a.get("duration") or "unspecified"
    dur_b = item_b.get("duration") or "unspecified"

    levels_a = item_a.get("job_levels", [])
    levels_b = item_b.get("job_levels", [])

    shared_codes = codes_a & codes_b
    unique_to_a = codes_a - codes_b
    unique_to_b = codes_b - codes_a

    lines = [
        f"**{name_a}** vs **{name_b}**",
        "",
    ]

    # Type comparison
    if shared_codes:
        lines.append(f"Both measure: {', '.join(sorted(shared_codes))}")
    if unique_to_a:
        lines.append(f"{name_a} additionally measures: {', '.join(sorted(unique_to_a))}")
    if unique_to_b:
        lines.append(f"{name_b} additionally measures: {', '.join(sorted(unique_to_b))}")

    # Duration
    lines.append(f"\nDuration — {name_a}: {dur_a} | {name_b}: {dur_b}")

    # Job levels
    shared_levels = set(levels_a) & set(levels_b)
    if shared_levels:
        lines.append(f"Shared job levels: {', '.join(sorted(shared_levels))}")

    # Description snippets
    desc_a = (item_a.get("description") or "")[:200]
    desc_b = (item_b.get("description") or "")[:200]
    if desc_a:
        lines.append(f"\n{name_a}: {desc_a}{'...' if len(item_a.get('description', '')) > 200 else ''}")
    if desc_b:
        lines.append(f"\n{name_b}: {desc_b}{'...' if len(item_b.get('description', '')) > 200 else ''}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_comparison_context(name_a: str, name_b: str) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    str,
    str,
]:
    """
    Retrieve both assessments and build their formatted context strings
    for injection into the LLM comparison prompt.

    Returns:
        (item_a, item_b, context_a_str, context_b_str)
        Any of these may be None/empty if the assessment is not found.
    """
    item_a = _find_assessment(name_a)
    item_b = _find_assessment(name_b)

    ctx_a = _format_item_for_comparison(item_a) if item_a else f"Assessment '{name_a}' not found in catalog."
    ctx_b = _format_item_for_comparison(item_b) if item_b else f"Assessment '{name_b}' not found in catalog."

    return item_a, item_b, ctx_a, ctx_b


def grounded_compare_fallback(name_a: str, name_b: str) -> str:
    """
    Produce a grounded comparison without an LLM call.
    Used when LLM is unavailable or as a fast path.
    """
    item_a = _find_assessment(name_a)
    item_b = _find_assessment(name_b)

    if not item_a and not item_b:
        return (
            f"Neither '{name_a}' nor '{name_b}' could be found in the SHL catalog. "
            "Please check the names and try again."
        )
    if not item_a:
        return (
            f"'{name_a}' could not be found in the catalog. "
            f"'{name_b}' is available: {_format_item_for_comparison(item_b)}"
        )
    if not item_b:
        return (
            f"'{name_b}' could not be found in the catalog. "
            f"'{name_a}' is available: {_format_item_for_comparison(item_a)}"
        )

    return _build_grounded_comparison(item_a, item_b)


def get_current_shortlist_items(
    current_recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Look up full catalog data for all items in the current shortlist.
    Returns enriched items with full metadata.
    """
    enriched = []
    for rec in current_recommendations:
        item = get_item_by_name(rec.get("name", ""))
        if item:
            enriched.append(item)
        else:
            # Return the recommendation as-is if not found
            enriched.append(rec)
    return enriched
