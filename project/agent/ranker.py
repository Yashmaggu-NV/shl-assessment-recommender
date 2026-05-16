"""
Hybrid ranker for SHL assessment recommendations.

Takes a list of retrieved catalog items and produces a ranked shortlist
using a multi-signal scoring model:
  1. Semantic relevance (from retriever score)
  2. Seniority / job-level alignment
  3. Technical skill overlap boosting
  4. Generic product noise penalty
  5. Battery balance heuristics (technical + cognitive + personality)
  6. Explicit user preference signals (included/excluded names/categories)
  7. Safety-critical role boosting
  8. Contact-centre role boosting

The ranker is deterministic and does not call an LLM.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.helpers import (
    KEY_TO_CODE,
    get_logger,
    normalize_text,
    infer_job_levels,
)
from agent.state import ConversationState

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

W_SEMANTIC = 1.0        # Base retrieval score weight
W_JOB_LEVEL = 0.3      # Job level match bonus
W_EXPLICIT_INCLUDE = 2.0  # Hard boost for explicitly included assessments
W_SAFETY_BOOST = 0.5   # Boost for safety instruments in safety-critical roles
W_CONTACT_CTR = 0.5    # Boost for contact-centre instruments
W_LEADERSHIP = 0.4     # Boost for leadership/executive instruments
W_TECH_SKILL = 0.6     # Boost per matched technical skill in item name/description

# Assessment names known to be core anchors (always high priority if relevant)
ANCHOR_ASSESSMENTS = {
    "occupational personality questionnaire opq32r": "P",
    "shl verify interactive g+": "A",
    "shl verify interactive g": "A",
    "graduate scenarios": "B",
    "dependability and safety instrument (dsi)": "P",
    "global skills assessment": "C",
}

# Safety-critical instrument names (boost for safety roles)
SAFETY_INSTRUMENTS = {
    "dependability and safety instrument (dsi)",
    "manufac. & indust. - safety & dependability 8.0",
    "safety and dependability focus 8.0",
    "workplace health and safety (new)",
}

# Contact-centre specific instruments
CONTACT_CENTRE_INSTRUMENTS = {
    "contact center call simulation (new)",
    "customer service phone simulation",
    "svar spoken english (us) (new)",
    "svar spoken english (uk) (new)",
    "svar spoken english (aus) (new)",
    "svar spoken english (india) (new)",
    "entry level customer serv - retail & contact center",
    "entry level customer service",
}

# Leadership / executive instruments
LEADERSHIP_INSTRUMENTS = {
    "occupational personality questionnaire opq32r",
    "opq leadership report",
    "opq universal competency report 2.0",
    "global skills assessment",
    "assessment and development center exercises",
    "shl verify interactive g+",
    "shl verify interactive g",
}

# Generic / noise patterns that should be penalised in technical queries
_GENERIC_NAME_RE = re.compile(
    r"\breport\b|\b360\b|\bglobal skills\b|\bvirtual assessment\b"
    r"|\bdevelopment cent(?:er|re)\b|\bassessment cent(?:er|re)\b"
    r"|\bgroup report\b|\bcandidate report\b|\bemployee report\b"
    r"|\bdev tips\b|\bperformance potential\b|\benterprise leadership\b"
    r"|\btalent audit\b|\bstandard report\b|\bcompetency report\b"
    r"|\buniversal competency\b|\bleadership report\b"
    r"|\bglobal skills development\b|\bscenarios\b"
    r"|\bparticipant report\b|\bmanager report\b|\bdevelopment report\b"
    r"|\bprofiling guide\b|\bjob profiling\b|\baction planner\b"
    r"|\bgeneric catalog\b|\bcatalog document\b"
    r"|\bremoteworkq\b|\bdigital readiness\b|\bhipo assessment\b"
    r"|\bworkplace safety\b.*\breport\b"
    r"|\bplanner report\b|\bguide\b"
    r"|\bexercises?\b|\bparticipant\b"
    r"|\bcenter exercises\b|\bcentre exercises\b"
    r"|\bdevelopment action\b|\btalent review\b"
    r"|\bsuccession\b|\bbenchmark\b.*\breport\b",
    re.IGNORECASE,
)

# Legacy / outdated product patterns — penalise heavily for modern tech queries
_LEGACY_PRODUCT_RE = re.compile(
    r"\b(j2ee|java\s*2\s*platform|java\s*ee|enterprise java beans|ejb"
    r"|1\.4\b|2\.0\s+platform|platform enterprise edition"
    r"|fundamental\b|automata)",
    re.IGNORECASE,
)

# Domain-irrelevance patterns — HARD EXCLUDE items from unrelated job families
# when the active role is tech/software/engineering
_IRRELEVANT_DOMAIN_FOR_TECH_RE = re.compile(
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

# Seniority level keywords in product names (for mismatch detection)
_ENTRY_LEVEL_NAME_RE = re.compile(
    r"\b(entry.?level|entry level|beginner|fundamental|basics?)\b",
    re.IGNORECASE,
)

# Cognitive / reasoning test patterns (should only appear if explicitly requested)
_COGNITIVE_TEST_RE = re.compile(
    r"\b(verify|deductive|inductive|numerical|verbal)\s+(reasoning|ability|g\+?)\b"
    r"|\breasoning\b|\bcognitive\b|\bgeneral ability\b",
    re.IGNORECASE,
)

# Category balance target: for general hires, aim for this mix
BALANCED_BATTERY = {
    "K": 4,  # Knowledge tests
    "A": 1,  # Cognitive
    "P": 1,  # Personality
    "B": 1,  # SJT
    "S": 1,  # Simulation
}


# ---------------------------------------------------------------------------
# Technical skill relevance scoring
# ---------------------------------------------------------------------------

def _compute_technical_relevance(
    item: Dict[str, Any],
    technical_skills: List[str],
) -> float:
    """
    Compute how many of the user's technical skills match the item's
    name and description. Returns a boost value.

    Also returns 0.0 if the item is a legacy product or has no skill overlap,
    signalling the ranker to penalise it.
    """
    if not technical_skills:
        return 0.0

    name_lower = (item.get("name") or "").lower()
    desc_lower = (item.get("description") or "").lower()
    combined = f"{name_lower} {desc_lower}"

    matches = 0
    for skill in technical_skills:
        skill_lower = skill.lower()
        # Check if the skill appears in the item text
        if skill_lower in combined:
            matches += 1
        # Also check common variations
        elif skill_lower == "aws" and "amazon web services" in combined:
            matches += 1
        elif skill_lower == "k8s" and "kubernetes" in combined:
            matches += 1
        elif skill_lower == "docker" and ("docker" in combined or "container" in combined):
            matches += 1
        elif skill_lower == "sql" and ("database" in combined or "mysql" in combined or "postgresql" in combined or "sql server" in combined):
            matches += 0.5

    return matches * W_TECH_SKILL


def _has_any_skill_overlap(
    item: Dict[str, Any],
    technical_skills: List[str],
) -> bool:
    """Check if the item's name or description mentions ANY of the technical skills."""
    if not technical_skills:
        return True  # No skills to check → everything passes
    name_lower = (item.get("name") or "").lower()
    desc_lower = (item.get("description") or "").lower()
    combined = f"{name_lower} {desc_lower}"
    for skill in technical_skills:
        s = skill.lower()
        if s in combined:
            return True
        if s == "aws" and "amazon web services" in combined:
            return True
        if s == "k8s" and "kubernetes" in combined:
            return True
        if s == "docker" and "container" in combined:
            return True
        if s == "sql" and ("database" in combined or "mysql" in combined or "postgresql" in combined or "sql server" in combined):
            return True
    return False


def rank_candidates(
    candidates: List[Dict[str, Any]],
    state: ConversationState,
    retrieval_scores: Optional[Dict[str, float]] = None,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """
    Rank a list of candidate catalog items and return the top shortlist.

    Args:
        candidates: Retrieved catalog items
        state: Reconstructed conversation state
        retrieval_scores: Optional dict of entity_id → retrieval score
        max_results: Maximum number of items to return (1–10)

    Returns:
        Ordered list of catalog items (best first), max max_results items.
    """
    if not candidates:
        return []

    retrieval_scores = retrieval_scores or {}
    scored: List[Tuple[Dict[str, Any], float]] = []

    # Pre-compute normalized name → item map for anchor lookup
    included_norms: Set[str] = {
        normalize_text(n) for n in state.included_names
    }
    excluded_norms: Set[str] = {
        normalize_text(n) for n in state.excluded_names
    }

    target_job_levels = infer_job_levels(state.seniority) if state.seniority else []
    has_tech_skills = bool(state.technical_skills)
    is_development = state.purpose == "development"

    # Score each candidate
    for item in candidates:
        eid = item["entity_id"]
        name = item.get("name", "")
        name_norm = normalize_text(name)

        # Skip explicitly excluded
        if name_norm in excluded_norms:
            continue

        # Base score from retrieval
        score = retrieval_scores.get(eid, 0.5) * W_SEMANTIC

        # Boost explicitly included items
        if name_norm in included_norms:
            score += W_EXPLICIT_INCLUDE

        # Job level alignment
        if target_job_levels and item.get("job_levels"):
            if any(jl in item["job_levels"] for jl in target_job_levels):
                score += W_JOB_LEVEL

        # Category codes for this item
        item_codes = [
            KEY_TO_CODE.get(k) for k in item.get("keys", [])
            if KEY_TO_CODE.get(k)
        ]
        item_codes_set = set(item_codes)

        # --- Technical skill relevance boost ---
        if has_tech_skills:
            tech_boost = _compute_technical_relevance(item, state.technical_skills)
            score += tech_boost
            if tech_boost > 0:
                _log.debug("Tech boost %.2f for '%s'", tech_boost, name)

            # Hard penalty for items with ZERO skill overlap in tech queries
            # (e.g., "Automata - Fix" for a Java/AWS query)
            if not _has_any_skill_overlap(item, state.technical_skills):
                # Only keep if it's cognitive/personality AND explicitly requested
                if item_codes_set <= {"A"} and state.needs_cognitive is True:
                    pass  # Explicitly requested cognitive — keep
                elif item_codes_set <= {"P"} and state.needs_personality is True:
                    pass  # Explicitly requested personality — keep
                else:
                    score *= 0.05
                    _log.debug("Zero skill overlap penalty for '%s'", name)

        # --- Legacy product penalty ---
        if has_tech_skills and _LEGACY_PRODUCT_RE.search(name):
            score *= 0.05
            _log.debug("Legacy product penalty for '%s'", name)

        # --- Seniority mismatch penalty ---
        if state.seniority in ("mid", "senior", "lead", "manager", "director", "executive"):
            if _ENTRY_LEVEL_NAME_RE.search(name):
                score *= 0.08
                _log.debug("Entry-level product penalty for mid+ query: '%s'", name)

        # --- Cognitive / reasoning test penalty (unless explicitly requested) ---
        if has_tech_skills and state.needs_cognitive is not True:
            if _COGNITIVE_TEST_RE.search(name) and "K" not in item_codes_set:
                score *= 0.08
                _log.debug("Unrequested cognitive test penalty for '%s'", name)

        # --- Generic product noise penalty ---
        if _GENERIC_NAME_RE.search(name) and not is_development:
            score *= 0.05
            _log.debug("Generic penalty applied to '%s'", name)

        # --- Domain-irrelevance HARD EXCLUSION ---
        # When the role is in a specific domain (e.g., software engineering),
        # ZERO-score items from unrelated domains (sales, customer service,
        # manufacturing, safety, etc.) so they NEVER appear in top-k.
        _is_tech = False
        if state.role:
            role_lower = (state.role or "").lower()
            _is_tech = any(kw in role_lower for kw in (
                "software", "engineer", "developer", "programmer", "coder",
                "data", "backend", "frontend", "fullstack", "devops", "sre",
                "architect", "tech", "it ", "computing", "cloud",
            ))
        if not _is_tech and has_tech_skills:
            _is_tech = True
        if _is_tech:
            if _IRRELEVANT_DOMAIN_FOR_TECH_RE.search(name):
                score = 0.0
                _log.debug("Domain HARD EXCLUDE for '%s' (tech role)", name)

        # --- Category-based noise penalty for tech queries ---
        if has_tech_skills:
            # Penalise Development & 360 products
            if "D" in item_codes_set and not is_development:
                score *= 0.05

            # Penalise Assessment Exercises
            if "E" in item_codes_set and not is_development:
                score *= 0.05

            # Penalise pure Competency products
            if item_codes_set == {"C"} and not is_development:
                score *= 0.1

            # Penalise pure personality if not explicitly requested
            if item_codes_set == {"P"} and state.needs_personality is not True:
                score *= 0.1

        # Safety-critical role boosts
        if state.safety_critical:
            if name_norm in SAFETY_INSTRUMENTS:
                score += W_SAFETY_BOOST
            if item_codes == ["K"] and name_norm not in SAFETY_INSTRUMENTS:
                score *= 0.8

        # Contact-centre boosts
        if _is_contact_centre(state):
            if name_norm in CONTACT_CENTRE_INSTRUMENTS:
                score += W_CONTACT_CTR

        # Leadership / executive boosts
        if state.needs_leadership or state.seniority in ("executive", "director"):
            if name_norm in LEADERSHIP_INSTRUMENTS:
                score += W_LEADERSHIP
            if "P" in item_codes:
                score += 0.2

        # Graduate / entry heuristics
        if state.seniority in ("graduate", "entry", "junior"):
            if "A" in item_codes or "B" in item_codes:
                score += 0.3
            if "K" in item_codes and len(item_codes) == 1:
                score *= 0.9

        scored.append((item, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Debug log: scored candidates before battery balance
    _log.info(
        "Ranked %d candidates (top-10 pre-balance): %s",
        len(scored),
        ", ".join(f"'{item.get('name','')}' ({s:.3f})"
                 for item, s in scored[:10]),
    )

    # Apply battery balance: ensure we don't return 10 items of the same type
    final = _apply_battery_balance(scored, state, max_results)

    # Debug log: final shortlist
    _log.info(
        "Final shortlist (%d items): %s",
        len(final),
        ", ".join(f"'{item.get('name','')}'({KEY_TO_CODE.get(item.get('keys',[''])[0] if item.get('keys') else '', 'K')})"
                 for item in final),
    )

    return final


def _apply_battery_balance(
    scored: List[Tuple[Dict[str, Any], float]],
    state: ConversationState,
    max_results: int,
) -> List[Dict[str, Any]]:
    """
    Select a balanced battery from scored candidates.

    Priority:
      1. Explicitly included items (always included if present)
      2. Best scoring item per category until target counts met
      3. Fill remaining slots with highest-scored items (subject to minimum relevance)

    Respects excluded_categories from state.
    """
    included_norms = {normalize_text(n) for n in state.included_names}
    excluded_categories = set(state.excluded_categories)
    has_tech_skills = bool(state.technical_skills)

    selected: List[Dict[str, Any]] = []
    selected_ids: Set[str] = set()

    # Compute minimum score threshold (items must be at least 20% of top score)
    if scored:
        max_score = scored[0][1]
        min_score_threshold = max_score * 0.15
    else:
        min_score_threshold = 0.0

    # Phase 1: Always include explicitly included assessments first
    for item, _ in scored:
        name_norm = normalize_text(item.get("name", ""))
        if name_norm in included_norms and item["entity_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["entity_id"])
            if len(selected) >= max_results:
                return selected

    # Determine category targets based on state
    category_targets = _get_category_targets(state, excluded_categories)

    # Phase 2: Fill category slots (only if item passes minimum relevance)
    category_counts: Dict[str, int] = {code: 0 for code in category_targets}

    for item, item_score in scored:
        if item["entity_id"] in selected_ids:
            continue
        if len(selected) >= max_results:
            break

        # Skip items below minimum relevance threshold
        if item_score < min_score_threshold:
            continue

        item_codes = [
            KEY_TO_CODE.get(k) for k in item.get("keys", [])
            if KEY_TO_CODE.get(k)
        ]

        # For technical queries, skip non-K category fills if the item
        # has no tech skill relevance (prevents pulling in random personality tests)
        if has_tech_skills:
            non_k_codes = [c for c in item_codes if c != "K"]
            if non_k_codes and not any(c == "K" for c in item_codes):
                # This is a non-knowledge item — only add if it has meaningful score
                if item_score < min_score_threshold * 2:
                    continue

        # Check if any category still needs filling
        added = False
        for code in item_codes:
            if code in excluded_categories:
                break
            target = category_targets.get(code, 0)
            if target > 0 and category_counts.get(code, 0) < target:
                selected.append(item)
                selected_ids.add(item["entity_id"])
                category_counts[code] = category_counts.get(code, 0) + 1
                added = True
                break

    # Phase 3: Fill remaining slots with highest-scored items
    for item, item_score in scored:
        if len(selected) >= max_results:
            break
        if item["entity_id"] in selected_ids:
            continue

        # Enforce minimum relevance for fill items too
        if item_score < min_score_threshold:
            continue

        item_codes = [
            KEY_TO_CODE.get(k) for k in item.get("keys", [])
            if KEY_TO_CODE.get(k)
        ]
        # Skip excluded categories
        if any(code in excluded_categories for code in item_codes):
            continue
        selected.append(item)
        selected_ids.add(item["entity_id"])

    return selected[:max_results]


def _get_category_targets(
    state: ConversationState,
    excluded_categories: Set[str],
) -> Dict[str, int]:
    """
    Compute target category counts for the battery based on hiring context.
    """
    targets: Dict[str, int] = {}
    has_tech_skills = bool(state.technical_skills)

    # Safety-critical: emphasise P
    if state.safety_critical:
        targets = {"P": 2, "K": 1}

    # Contact centre: sim + personality + knowledge
    elif _is_contact_centre(state):
        targets = {"S": 2, "P": 1, "K": 1, "B": 1}

    # Graduate / entry: cognitive + SJT + personality
    elif state.seniority in ("graduate", "entry"):
        targets = {"A": 1, "B": 1, "P": 1, "K": 2}

    # Executive / senior leadership: personality + cognitive
    elif state.needs_leadership or state.seniority in ("executive", "director"):
        targets = {"P": 2, "A": 1, "K": 1}

    # Development / talent audit
    elif state.purpose == "development":
        targets = {"P": 2, "C": 2, "K": 1, "D": 1}

    # Technical role with skills: K-dominant battery, but make room for
    # explicitly requested categories (personality, cognitive, etc.)
    elif has_tech_skills:
        # Start with K-dominant battery
        k_slots = 10
        # Reserve slots for explicitly requested categories
        if state.needs_personality is True:
            k_slots -= 2  # room for personality tests (e.g., OPQ32r)
            targets["P"] = 2
        if state.needs_cognitive is True:
            k_slots -= 1
            targets["A"] = 1
        if state.needs_sjt is True:
            k_slots -= 1
            targets["B"] = 1
        if state.needs_simulation is True:
            k_slots -= 1
            targets["S"] = 1
        targets["K"] = max(k_slots, 3)  # always keep at least 3 K slots

    # General senior IC: technical + cognitive + personality
    elif state.seniority in ("senior", "lead"):
        targets = {"K": 4, "A": 1, "P": 1}

    # Mid-level general
    elif state.seniority in ("mid", "manager"):
        targets = {"K": 3, "A": 1, "P": 1, "B": 1}

    # Default balanced battery
    else:
        targets = {"K": 3, "A": 1, "P": 1, "B": 1, "S": 1}

    # Override with explicit category inclusions from state
    for code in state.included_categories:
        if code not in excluded_categories:
            targets[code] = targets.get(code, 0) + 1

    # Remove excluded categories
    for code in excluded_categories:
        targets.pop(code, None)

    # If personality explicitly excluded
    if state.needs_personality is False:
        targets.pop("P", None)
    elif state.needs_personality is True and "P" not in targets:
        targets["P"] = 1

    # If cognitive explicitly requested
    if state.needs_cognitive is True and "A" not in targets:
        targets["A"] = 1
    elif state.needs_cognitive is False:
        targets.pop("A", None)

    # SJT explicitly requested
    if state.needs_sjt is True and "B" not in targets:
        targets["B"] = 1

    # Simulation explicitly requested
    if state.needs_simulation is True and "S" not in targets:
        targets["S"] = 1

    return targets


def _is_contact_centre(state: ConversationState) -> bool:
    """Detect if the role is contact-centre / customer service."""
    role = (state.role or "").lower()
    industry = (state.industry or "").lower()
    keywords = {"contact centre", "contact center", "call centre", "call center",
                "customer service", "customer support", "inbound"}
    return any(kw in role or kw in industry for kw in keywords)
