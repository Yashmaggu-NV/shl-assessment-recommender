"""
Recommendation engine: assembles final shortlists from retrieved candidates.

Responsibilities:
  1. Build a rich retrieval query from conversation state
  2. Call hybrid_retrieve() for candidate pool
  3. Apply ranker for scoring and battery balance
  4. Post-ranking pruning: deduplication, relevance floor
  5. Enforce hard constraints (included/excluded names, categories)
  6. Return final ordered list of 1–10 catalog items

Also handles refinement operations:
  - Add assessment: inject into shortlist
  - Remove assessment: filter out of shortlist
  - Replace assessment: swap out one item for another
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from agent.ranker import rank_candidates
from agent.retriever import (
    hybrid_retrieve,
    get_item_by_name,
    keyword_search,
)
from agent.state import ConversationState
from utils.helpers import (
    KEY_TO_CODE,
    get_logger,
    infer_job_levels,
    keys_to_type_code,
    normalize_text,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Query builder
# ---------------------------------------------------------------------------

def build_retrieval_query(state: ConversationState, user_message: str) -> str:
    """
    Construct a rich retrieval query from user message + conversation state.

    For technical roles with skills, produces a skill-focused query to maximize
    keyword overlap with technical assessment names in the catalog.
    """
    # If we have technical skills, build a focused skill query
    if state.technical_skills:
        return _build_skill_focused_query(state, user_message)

    # Generic query construction
    parts = [user_message]

    if state.role:
        parts.append(state.role)
    if state.seniority:
        parts.append(state.seniority)
    if state.industry:
        parts.append(state.industry)
    if state.safety_critical:
        parts.extend(["safety", "dependability", "reliability"])
    if state.needs_personality:
        parts.append("personality behaviour")
    if state.needs_cognitive:
        parts.append("cognitive ability reasoning")
    if state.needs_sjt:
        parts.append("situational judgment")
    if state.needs_simulation:
        parts.append("simulation")
    if state.needs_leadership:
        parts.append("leadership executive")
    if state.purpose == "development":
        parts.append("development 360")

    return " ".join(parts)


def _build_skill_focused_query(state: ConversationState, user_message: str) -> str:
    """
    Build a retrieval query that prioritises technical skill keywords.

    For a query like "Java backend engineer with AWS", this produces:
    "Java Spring SQL AWS Docker backend engineer mid assessment test"

    This ensures keyword overlap with catalog items named
    "Core Java (Advanced Level) (New)", "Amazon Web Services (AWS)", etc.

    Also appends personality/cognitive/leadership/communication terms
    when those flags are explicitly set on state (e.g., after a
    refinement turn "add personality and teamwork assessments").
    """
    parts = []

    # Technical skills first (highest priority for keyword matching)
    parts.extend(state.technical_skills)

    # Add the role if present (but keep it brief)
    if state.role:
        parts.append(state.role)

    # Add key terms from user message that aren't already covered
    # (to capture terms like "backend", "engineer", "hiring")
    covered = set(s.lower() for s in parts)
    for word in user_message.lower().split():
        clean = re.sub(r'[^a-z0-9]', '', word)
        if clean and len(clean) > 2 and clean not in covered:
            covered.add(clean)
            parts.append(clean)

    # Append category-specific terms when explicitly requested.
    # Without these, the retriever would never surface personality or
    # cognitive assessments for a tech role.
    if state.needs_personality is True:
        parts.append("personality behaviour teamwork OPQ")
    if state.needs_cognitive is True:
        parts.append("cognitive ability reasoning verify")
    if state.needs_leadership is True:
        parts.append("leadership executive")
    if state.needs_sjt is True:
        parts.append("situational judgment")
    if state.needs_simulation is True:
        parts.append("simulation")

    # Add "assessment test" to help semantic search
    parts.append("assessment test")

    query = " ".join(parts)
    _log.info("Skill-focused query: '%s'", query)
    return query


# ---------------------------------------------------------------------------
# Shortlist assembly
# ---------------------------------------------------------------------------

def assemble_recommendations(
    user_message: str,
    state: ConversationState,
    previous_recommendations: Optional[List[Dict[str, Any]]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Build a final recommendation shortlist.

    Workflow:
      1. Build retrieval query
      2. Hybrid retrieve candidates (semantic + keyword + metadata + noise filter)
      3. Inject previously confirmed items
      4. Rank and balance
      5. Post-rank pruning (dedup, relevance floor)
      6. Return 1–10 items

    Args:
        user_message: Latest user message
        state: Reconstructed conversation state
        previous_recommendations: Items from the previous turn (for refinement)
        max_results: Cap at this many results (max 10)

    Returns:
        Ordered list of catalog item dicts
    """
    query = build_retrieval_query(state, user_message)
    job_levels = infer_job_levels(state.seniority)

    # Determine if we should allow generic products
    allow_generic = (
        state.purpose == "development"
        or state.needs_leadership is True
        or any(c in (state.included_categories or []) for c in ["D", "E", "C"])
    )

    # Retrieve candidate pool (now with noise filtering built in)
    candidates = hybrid_retrieve(
        query=query,
        state_context=state.to_context_string(),
        job_levels=job_levels if job_levels else None,
        languages=state.languages if state.languages else None,
        include_categories=state.included_categories if state.included_categories else None,
        exclude_categories=state.excluded_categories if state.excluded_categories else None,
        exclude_names=state.excluded_names if state.excluded_names else None,
        technical_skills=state.technical_skills if state.technical_skills else None,
        purpose=state.purpose,
        allow_generic=allow_generic,
        needs_personality=state.needs_personality,
        needs_leadership=state.needs_leadership,
        top_k=40,
    )

    # Inject explicitly included items that may not have ranked high
    if state.included_names:
        included_items = _fetch_included_items(state.included_names, candidates)
        candidate_ids = {c["entity_id"] for c in candidates}
        for item in included_items:
            if item["entity_id"] not in candidate_ids:
                candidates.insert(0, item)

    # Inject previous recommendations for continuity
    if previous_recommendations:
        prev_items = _fetch_previous_items(previous_recommendations, candidates)
        candidate_ids = {c["entity_id"] for c in candidates}
        for item in prev_items:
            if item["entity_id"] not in candidate_ids:
                candidates.append(item)

    if not candidates:
        _log.warning("No candidates found for query: %.80s", query)
        return []

    # Build retrieval score map for ranker
    retrieval_scores = _build_score_map(query, candidates)

    # Rank candidates
    shortlist = rank_candidates(
        candidates=candidates,
        state=state,
        retrieval_scores=retrieval_scores,
        max_results=max_results,
    )

    # Post-rank pruning: remove near-duplicates
    shortlist = _deduplicate_shortlist(shortlist)

    # Post-rank pruning: for technical queries, drop items with no skill overlap
    if state.technical_skills:
        shortlist = _prune_weak_tech_matches(shortlist, state.technical_skills, state)

    # Post-rank pruning: drop items with zero query-token overlap
    # This is a lightweight final check catching loosely-matched battery-fill items.
    shortlist = _prune_zero_query_overlap(shortlist, query, state)

    # Post-rank pruning: HARD domain-irrelevance filter for tech roles.
    shortlist = _post_rank_domain_filter(shortlist, state)

    _log.info(
        "Assembled %d recommendations for query: %.60s",
        len(shortlist),
        query,
    )

    # Debug log: final shortlist details
    for i, item in enumerate(shortlist, 1):
        codes = keys_to_type_code(item.get("keys", []))
        _log.info(
            "  [%d] %s (type=%s, duration=%s, levels=%s)",
            i,
            item.get("name", "?"),
            codes,
            item.get("duration", "?"),
            ", ".join(item.get("job_levels", [])[:3]),
        )

    return shortlist


def apply_refinement(
    action: str,  # "add" | "remove" | "replace"
    target_name: str,
    replacement_name: Optional[str],
    current_shortlist: List[Dict[str, Any]],
    state: ConversationState,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Apply a refinement operation to the current shortlist.

    Args:
        action: "add" | "remove" | "replace"
        target_name: Name of the assessment to add/remove/replace
        replacement_name: New assessment name for "replace" action
        current_shortlist: Current recommendation list
        state: Conversation state (for context in add operations)

    Returns:
        (updated_shortlist, feedback_message)
    """
    target_norm = normalize_text(target_name)
    shortlist = list(current_shortlist)  # copy

    if action == "remove":
        before_len = len(shortlist)
        shortlist = [
            item for item in shortlist
            if normalize_text(item.get("name", "")) != target_norm
        ]
        if len(shortlist) < before_len:
            return shortlist, f"Removed {target_name} from the shortlist."
        else:
            return shortlist, f"'{target_name}' was not in the shortlist."

    elif action == "add":
        item = get_item_by_name(target_name)
        if not item:
            # Try keyword search
            results = keyword_search(target_name, top_k=1)
            if results:
                item = results[0][0]
        if not item:
            return shortlist, (
                f"Could not find '{target_name}' in the SHL catalog. "
                "Please check the name."
            )
        # Check not already in list
        existing_ids = {i["entity_id"] for i in shortlist}
        if item["entity_id"] in existing_ids:
            return shortlist, f"'{item['name']}' is already in the shortlist."
        if len(shortlist) >= 10:
            return shortlist, (
                f"Cannot add '{item['name']}' — shortlist already has 10 items. "
                "Please remove one first."
            )
        shortlist.append(item)
        return shortlist, f"Added {item['name']}."

    elif action == "replace":
        # Remove old
        old_item = next(
            (i for i in shortlist if normalize_text(i.get("name", "")) == target_norm),
            None,
        )
        if not old_item:
            return shortlist, f"'{target_name}' is not in the current shortlist."
        shortlist = [i for i in shortlist if i["entity_id"] != old_item["entity_id"]]

        # Add new
        if replacement_name:
            new_item = get_item_by_name(replacement_name)
            if not new_item:
                results = keyword_search(replacement_name, top_k=1)
                if results:
                    new_item = results[0][0]
            if new_item:
                shortlist.append(new_item)
                return shortlist, f"Replaced {target_name} with {new_item['name']}."
            else:
                return shortlist, (
                    f"Removed {target_name}. Could not find '{replacement_name}' "
                    "in the catalog."
                )
        return shortlist, f"Removed {target_name}."

    return shortlist, "No changes made."


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fetch_included_items(
    included_names: List[str],
    existing_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fetch catalog items for explicitly included names not already in candidates."""
    existing_names_norm = {normalize_text(c.get("name", "")) for c in existing_candidates}
    result = []
    for name in included_names:
        if normalize_text(name) not in existing_names_norm:
            item = get_item_by_name(name)
            if item:
                result.append(item)
    return result


def _fetch_previous_items(
    previous_recommendations: List[Dict[str, Any]],
    existing_candidates: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fetch catalog items for previous recommendations not already in candidates."""
    existing_ids = {c["entity_id"] for c in existing_candidates}
    result = []
    for rec in previous_recommendations:
        item = get_item_by_name(rec.get("name", ""))
        if item and item["entity_id"] not in existing_ids:
            result.append(item)
    return result


def _build_score_map(
    query: str,
    candidates: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Build a simple retrieval score map using keyword overlap.
    Used to pass relevance signals to the ranker.
    """
    from utils.helpers import extract_keywords, build_catalog_text
    query_tokens = set(extract_keywords(query))
    score_map = {}
    for item in candidates:
        item_tokens = set(extract_keywords(build_catalog_text(item)))
        overlap = query_tokens & item_tokens
        score = len(overlap) / max(len(query_tokens), 1)
        score_map[item["entity_id"]] = score
    return score_map


def _deduplicate_shortlist(
    items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Remove near-duplicate items from the shortlist.

    Two items are near-duplicates if their normalized names share >85%
    of their tokens (e.g., "Core Java (New)" vs "Core Java (Advanced Level) (New)").
    The first occurrence (higher ranked) is kept.
    """
    if len(items) <= 1:
        return items

    deduped = []
    seen_name_tokens: List[set] = []

    for item in items:
        name_tokens = set(normalize_text(item.get("name", "")).split())
        if not name_tokens:
            deduped.append(item)
            continue

        is_dup = False
        for existing_tokens in seen_name_tokens:
            if not existing_tokens:
                continue
            overlap = name_tokens & existing_tokens
            smaller = min(len(name_tokens), len(existing_tokens))
            if smaller > 0 and len(overlap) / smaller > 0.85:
                _log.debug(
                    "Dedup: dropping '%s' (near-duplicate of existing item)",
                    item.get("name", ""),
                )
                is_dup = True
                break

        if not is_dup:
            deduped.append(item)
            seen_name_tokens.append(name_tokens)

    return deduped


def _prune_weak_tech_matches(
    items: List[Dict[str, Any]],
    technical_skills: List[str],
    state: "ConversationState",
) -> List[Dict[str, Any]]:
    """
    Final strict pruning for technical queries.

    Removes items whose name and description don't mention ANY of the
    requested technical skills.

    Exceptions (items that survive even without skill overlap):
      - Explicitly included items (state.included_names)
      - Personality (P) items when state.needs_personality is True
      - Cognitive (A) items when state.needs_cognitive is True
      - SJT (B) items when state.needs_sjt is True
      - Simulation (S) items when state.needs_simulation is True

    Also enforces seniority alignment: entry-level products are dropped
    for mid/senior queries.
    """
    import re as _re

    if not technical_skills:
        return items

    included_norms = {normalize_text(n) for n in (state.included_names or [])}

    # Build skill aliases for matching
    skill_patterns = []
    for skill in technical_skills:
        s = skill.lower()
        if s == "aws":
            skill_patterns.append(r"\b(aws|amazon web services)\b")
        elif s == "k8s":
            skill_patterns.append(r"\b(k8s|kubernetes)\b")
        elif s == "sql":
            skill_patterns.append(r"\b(sql|database|mysql|postgresql|sql server)\b")
        elif s == "docker":
            skill_patterns.append(r"\b(docker|container)\b")
        else:
            skill_patterns.append(r"\b" + _re.escape(s) + r"\b")

    combined_re = _re.compile("|".join(skill_patterns), _re.IGNORECASE)

    # Build set of category codes that are explicitly requested and
    # should be exempt from tech-skill-overlap pruning.
    exempt_codes = set()
    if state.needs_personality is True:
        exempt_codes.add("P")
    if state.needs_cognitive is True:
        exempt_codes.add("A")
    if state.needs_sjt is True:
        exempt_codes.add("B")
    if state.needs_simulation is True:
        exempt_codes.add("S")

    pruned = []
    for item in items:
        name = item.get("name", "")
        name_norm = normalize_text(name)

        # Always keep explicitly included items
        if name_norm in included_norms:
            pruned.append(item)
            continue

        # Keep items in explicitly requested categories even if they
        # don't mention a technical skill (e.g., OPQ32r for personality)
        item_codes = {
            KEY_TO_CODE.get(k) for k in item.get("keys", [])
            if KEY_TO_CODE.get(k)
        }
        if item_codes & exempt_codes:
            pruned.append(item)
            continue

        combined_text = f"{name} {item.get('description', '')}"

        if combined_re.search(combined_text):
            pruned.append(item)
        else:
            _log.info("Tech pruning dropped: '%s' (no skill overlap)", name)

    return pruned


def _prune_zero_query_overlap(
    shortlist: List[Dict[str, Any]],
    query: str,
    state: "ConversationState",
) -> List[Dict[str, Any]]:
    """
    Final lightweight relevance check: drop items with ZERO meaningful token
    overlap with the retrieval query, unless they are in an explicitly
    requested category.

    Exemptions (always kept):
    - Explicitly included items (state.included_names)
    - Items in explicitly requested categories (P when needs_personality, etc.)
    - Items in state.included_categories

    Only runs when query has >= 3 meaningful tokens.
    """
    from utils.helpers import extract_keywords

    query_tokens = set(extract_keywords(query))
    if len(query_tokens) < 3:
        return shortlist

    included_norms = {normalize_text(n) for n in (state.included_names or [])}

    exempt_codes: set = set(state.included_categories or [])
    if state.needs_personality is True:
        exempt_codes.add("P")
    if state.needs_cognitive is True:
        exempt_codes.add("A")
    if state.needs_sjt is True:
        exempt_codes.add("B")
    if state.needs_simulation is True:
        exempt_codes.add("S")
    if state.needs_leadership is True:
        exempt_codes.update({"P", "A"})

    pruned = []
    for item in shortlist:
        name = item.get("name", "")
        name_norm = normalize_text(name)

        if name_norm in included_norms:
            pruned.append(item)
            continue

        item_codes = {
            KEY_TO_CODE.get(k) for k in item.get("keys", [])
            if KEY_TO_CODE.get(k)
        }
        if item_codes & exempt_codes:
            pruned.append(item)
            continue

        item_text = f"{name} {item.get('description', '')}"
        item_tokens = set(extract_keywords(item_text))

        if query_tokens & item_tokens:
            pruned.append(item)
        else:
            _log.info("Zero query-overlap pruned: '%s'", name)

    return pruned if pruned else shortlist


# Domain-irrelevance regex — same patterns used across ranker and chat_logic
_DOMAIN_IRRELEVANT_RE = re.compile(
    r"\b(sales(?!force)|selling|sales.?transformation"
    r"|customer service|call cent|contact cent"
    r"|phone solution|phone simulation"
    r"|retail|merchandis|cashier|store|shop"
    r"|manufac|indust(?!ry)(?!rial engineering)"
    r"|mechanical.?(?:focus|vigilance)"
    r"|plant operator"
    r"|safety.?(?:and|&)?.?dependab|dependab.?(?:and|&)?.?safety"
    r"|workplace.?(?:health|safety)|safety focus"
    r"|warehouse|logistics|forklift|driver"
    r"|nursing|nurse|healthcare aide|carer"
    r"|clerical|filing|receptionist"
    r"|food service|hospitality|housekeep"
    r"|entry.?level.?customer|entry.?level.?sales"
    r"|entry.?level.?cashier|entry.?level.?hotel)\b",
    re.IGNORECASE,
)

_TECH_ROLE_KW = (
    "software", "engineer", "developer", "programmer", "coder",
    "data", "backend", "frontend", "fullstack", "devops", "sre",
    "architect", "tech", "it ", "computing", "cloud",
)


def _post_rank_domain_filter(
    shortlist: List[Dict[str, Any]],
    state: ConversationState,
) -> List[Dict[str, Any]]:
    """
    Hard domain-irrelevance filter applied AFTER ranking.

    For tech/software roles, removes any item whose name matches unrelated
    domain patterns (sales, manufacturing, safety, customer service, etc.).

    This is the deepest safety net — catches items that survived the ranker's
    zero-score (e.g., injected via explicit inclusion or battery balancing).
    """
    # Detect tech context from role OR technical_skills
    is_tech = False
    if state.role:
        role_lower = state.role.lower()
        is_tech = any(kw in role_lower for kw in _TECH_ROLE_KW)
    if not is_tech and state.technical_skills:
        is_tech = True
    if not is_tech:
        return shortlist

    filtered = []
    for item in shortlist:
        name = item.get("name", "")
        if _DOMAIN_IRRELEVANT_RE.search(name):
            _log.info("Post-rank domain filter REMOVED: '%s'", name)
            continue
        filtered.append(item)

    return filtered if filtered else shortlist


# ---------------------------------------------------------------------------
# Refinement intent detection
# ---------------------------------------------------------------------------

_REMOVE_INTENT_RE = re.compile(
    r"(?:remove|drop|exclude|skip|no\s+more|without|don't|do\s+not)\s+"
    r"(?:the\s+|any\s+)?(.+?)(?:\s+tests?|\s+assessments?|\s+from|\s+please|\s*$)",
    re.IGNORECASE,
)

_ADD_INTENT_RE = re.compile(
    r"(?:add|include|also\s+add|also\s+include|i\s+(?:also\s+)?want|we\s+(?:also\s+)?want|plus|along\s+with)\s+"
    r"(?:a\s+|an\s+|the\s+|some\s+)?(.+?)(?:\s+tests?|\s+assessments?|\s+too|\s+as\s+well|\s*$)",
    re.IGNORECASE,
)


def detect_refinement_intent(
    message: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Detect refinement intent from the user message.

    Returns:
        (action, target_name, replacement_name) or None
        action is "add" | "remove" | "replace"
    """
    remove_match = _REMOVE_INTENT_RE.search(message)
    add_match = _ADD_INTENT_RE.search(message)

    # Check for replace FIRST (remove X + add Y in same message)
    if remove_match and add_match:
        return ("replace", remove_match.group(1).strip(), add_match.group(1).strip())

    # Check for remove
    if remove_match:
        return ("remove", remove_match.group(1).strip(), None)

    # Check for add
    if add_match:
        return ("add", add_match.group(1).strip(), None)

    return None
