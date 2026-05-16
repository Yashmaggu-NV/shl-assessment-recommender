"""
Response formatter module.

Converts raw catalog items into structured API response objects.
Ensures:
  - All URLs come exclusively from the catalog (no hallucination)
  - test_type codes are correctly derived from catalog 'keys'
  - Recommendations are capped at 1–10 items
  - end_of_conversation logic is applied correctly
"""

from typing import Any, Dict, List, Optional

from models.schemas import Recommendation, ChatResponse
from utils.helpers import keys_to_type_code, get_logger

_log = get_logger(__name__)


def format_recommendation(item: Dict[str, Any]) -> Recommendation:
    """
    Convert a single catalog item dict into a Recommendation schema object.

    Args:
        item: A dict from the SHL catalog with keys: name, link, keys, etc.

    Returns:
        Recommendation with name, url (from catalog), and test_type code.
    """
    name = item.get("name", "Unknown")
    url = item.get("link", "")
    keys = item.get("keys", [])
    test_type = keys_to_type_code(keys)

    if not url:
        _log.warning("Catalog item '%s' has no link — skipping URL.", name)

    return Recommendation(name=name, url=url, test_type=test_type)


def format_recommendations(
    items: List[Dict[str, Any]],
    max_items: int = 10,
) -> List[Recommendation]:
    """
    Convert a list of catalog items to Recommendation objects.
    Enforces the 1–10 item cap, filters items with missing URLs,
    and applies a final safety filter to strip report/guide products.

    Args:
        items: Catalog item dicts
        max_items: Hard cap (default 10, assignment max)

    Returns:
        List of Recommendation objects (0–10 items).
    """
    import re

    # Last-mile filter: strip report/guide/document/exercise products that survived
    # the retriever and ranker filters. This is the final safety net.
    _REPORT_FILTER = re.compile(
        r"\breport\b|\bguide\b|\bprofiling\b|\bplanner\b"
        r"|\bremoteworkq\b|\bdigital readiness\b|\bhipo\b"
        r"|\b360\b|\bdev tips\b|\bscenarios\b"
        r"|\bglobal skills development\b|\bvirtual assessment\b"
        r"|\bexercises?\b|\bparticipant\b"
        r"|\bdevelopment cent(?:er|re)\b|\bassessment cent(?:er|re)\b"
        r"|\bdevelopment action\b|\btalent review\b|\bsuccession\b",
        re.IGNORECASE,
    )

    recommendations = []
    for item in items[:max_items + 5]:  # over-fetch to compensate for filtered items
        if len(recommendations) >= max_items:
            break

        url = item.get("link", "")
        if not url:
            _log.warning(
                "Skipping item '%s' — no catalog URL.", item.get("name", "?")
            )
            continue

        name = item.get("name", "")
        if _REPORT_FILTER.search(name):
            _log.info("Last-mile filter removed report product: '%s'", name)
            continue

        recommendations.append(format_recommendation(item))

    return recommendations


def build_chat_response(
    reply: str,
    items: Optional[List[Dict[str, Any]]] = None,
    end_of_conversation: bool = False,
    is_clarification: bool = False,
    is_refusal: bool = False,
    is_comparison: bool = False,
) -> ChatResponse:
    """
    Build the final ChatResponse object for the API.

    Logic:
      - recommendations = [] when clarifying, refusing, or comparing
      - recommendations = 1–10 items when recommending or refining
      - end_of_conversation = True only when agent considers task done

    Args:
        reply: Natural language reply text
        items: Catalog items to recommend (used when recommending/refining)
        end_of_conversation: Set True when conversation is complete
        is_clarification: True if this is a clarification turn
        is_refusal: True if this is a refusal
        is_comparison: True if this is a comparison turn

    Returns:
        ChatResponse with validated fields.
    """
    # Comparison: always return empty recommendations per conversation traces
    # (shortlist is preserved in message history for subsequent turns)
    if is_comparison:
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    # Clarification or refusal: always empty recommendations
    if is_clarification or is_refusal:
        return ChatResponse(
            reply=reply,
            recommendations=[],
            end_of_conversation=False,
        )

    # Recommend / refine / close
    if items:
        recs = format_recommendations(items, max_items=10)
        return ChatResponse(
            reply=reply,
            recommendations=recs,
            end_of_conversation=end_of_conversation,
        )

    # Fallback: no items provided, empty list
    return ChatResponse(
        reply=reply,
        recommendations=[],
        end_of_conversation=end_of_conversation,
    )


def extract_previous_recommendations(
    messages: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """
    Parse previous assistant messages to extract any confirmed recommendations.

    This is used to reconstruct the shortlist carried across turns.
    Returns a list of {"name": ..., "url": ..., "test_type": ...} dicts
    (matching the Recommendation schema).

    Strategy: look for the most recent assistant message that contains
    a structured shortlist (identified by 'url' patterns pointing to shl.com).
    """
    import re
    import json

    SHL_URL_RE = re.compile(
        r"https://www\.shl\.com/products/product-catalog/view/[^\s\"'\]>)]+",
        re.IGNORECASE,
    )

    # Walk messages in reverse to find the most recent assistant shortlist
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]

        # Try JSON extraction first (if assistant replied with JSON)
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "recommendations" in data:
                recs = data["recommendations"]
                if isinstance(recs, list) and recs:
                    return recs
        except (json.JSONDecodeError, ValueError):
            pass

        # Fall back: parse recommendation-like structures from markdown
        # Look for lines with assessment names and shl.com URLs
        urls = SHL_URL_RE.findall(content)
        if urls:
            # Extract name-url pairs from table-formatted content
            recs = _parse_markdown_table_recs(content)
            if recs:
                return recs

    return []


def _parse_markdown_table_recs(content: str) -> List[Dict[str, str]]:
    """
    Parse SHL assessment recommendations from a markdown table in the assistant message.

    Handles the format used in the conversation traces:
    | # | Name | Test Type | ... | URL |
    | 1 | Java 8 (New) | K | ... | https://... |
    """
    import re

    TABLE_ROW_RE = re.compile(
        r"\|\s*\d+\s*\|\s*(.+?)\s*\|\s*([A-Z,]+)\s*\|.*?"
        r"(https://www\.shl\.com/products/product-catalog/view/[^\s|>)]+)",
        re.IGNORECASE,
    )

    recs = []
    for match in TABLE_ROW_RE.finditer(content):
        name = match.group(1).strip()
        test_type = match.group(2).strip()
        url = match.group(3).strip().rstrip("/|>)")
        if name and url:
            recs.append({"name": name, "url": url, "test_type": test_type})

    return recs
