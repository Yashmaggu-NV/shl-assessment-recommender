"""
Central chat orchestration logic.

This is the brain of the agent. It:
  1. Runs guards (injection / off-topic detection)
  2. Reconstructs conversation state from full message history
  3. Classifies the turn: clarify | recommend | refine | compare | refuse | close
  4. Calls the appropriate sub-module
  5. Calls OpenRouter (Mistral) for natural-language generation
  6. Returns a structured ChatResponse

The entire pipeline is stateless — state is rebuilt from messages on every call.
All LLM calls are wrapped with timeout and exception handling.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple


from agent.comparison import (
    build_comparison_context,
    extract_comparison_names,
    grounded_compare_fallback,
    is_comparison_request,
)
from agent.formatter import (
    build_chat_response,
    extract_previous_recommendations,
    format_recommendations,
)
from agent.guards import check_guards
from agent.prompts import (
    COMPARISON_PROMPT,
    ORCHESTRATION_PROMPT,
    STATE_EXTRACTION_PROMPT,
    REFUSAL_LEGAL,
    REFUSAL_OFF_TOPIC,
)
from agent.recommendation_engine import (
    assemble_recommendations,
    detect_refinement_intent,
)
from agent.refusal import (
    build_refusal_response,
    classify_refusal,
    is_vague_request,
)
from agent.retriever import hybrid_retrieve, get_item_by_name
from agent.state import ConversationState, reconstruct_state_from_history
from models.schemas import ChatResponse, ChatRequest
from utils.helpers import (
    get_logger,
    get_env,
    keys_to_type_code,
    load_catalog,
    normalize_text,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# LLM router — cascading multi-model fallback (see agent/llm_router.py)
# ---------------------------------------------------------------------------

from agent.llm_router import (
    TurnType,
    call_llm_fast,
    call_llm_recommend,
    call_llm_refine,
    call_llm_compare,
    route_llm_call,
)

# Backward-compat alias: legacy call sites that just call _call_llm(prompt)
# are routed through the standard recommend path.
def _call_llm(prompt: str, timeout: int = 15, max_tokens: int = 1024) -> Optional[str]:
    """Backward-compat alias — delegates to the cascading router."""
    return call_llm_recommend(prompt)


# ---------------------------------------------------------------------------
# Keyword-based refinement detector (backup for when regex fails)
# ---------------------------------------------------------------------------

_REFINE_SIGNAL_WORDS = frozenset({
    # Action words
    "add", "also", "include", "plus", "refine", "narrow", "filter",
    "remove", "drop", "exclude", "replace", "swap", "without",
    # Category words that always signal a refinement in mid-conversation
    "leadership", "collaboration", "teamwork", "personality", "cognitive",
    "communication", "situational", "sjt", "behavioural", "behavioral",
})


def _is_refinement_turn(user_message: str, has_assistant_history: bool) -> bool:
    """
    Return True if this turn is a refinement of a prior conversation.

    Uses two independent signals so one can back up the other:
    1. detect_refinement_intent() — regex-based, high precision
    2. keyword intersection — catches cases the regex misses
    """
    if not has_assistant_history:
        return False
    # Signal 1: regex-based
    if detect_refinement_intent(user_message) is not None:
        return True
    # Signal 2: keyword-based fallback
    words = set(user_message.lower().split())
    if words & _REFINE_SIGNAL_WORDS:
        return True
    return False


# ---------------------------------------------------------------------------
# LLM role inference (fallback for descriptive queries)
# ---------------------------------------------------------------------------

_ROLE_INFERENCE_PROMPT = """You are an HR assessment expert. The user described a hiring need but did not use a standard job title. Infer the most likely role profile from their description.

User message: "{user_message}"

Return a JSON object with these fields (use null if you cannot infer):
{{
  "role": "inferred job title (e.g., 'executive leader', 'team manager', 'strategic planner')",
  "seniority": "junior|mid|senior|lead|manager|director|executive",
  "skills": ["list", "of", "key", "competencies"],
  "needs_leadership": true/false,
  "needs_personality": true/false,
  "needs_cognitive": true/false
}}

Return ONLY valid JSON, no markdown fences, no explanation."""


def _infer_role_from_context(user_message: str) -> Optional[Dict[str, Any]]:
    """
    Use the LLM to infer a role profile from a descriptive user message.

    Called when catalog retrieval returns few/no results because the user
    described responsibilities and skills instead of a standard job title.
    Handles queries about leadership, communication, conflict management,
    strategic thinking, team management, remote work, etc.

    Returns parsed JSON dict or None on failure.
    """
    prompt = _ROLE_INFERENCE_PROMPT.format(user_message=user_message)
    raw = call_llm_fast(prompt, timeout=6, max_tokens=256)
    if not raw:
        return None

    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    _log.debug("Role inference JSON parse failed: %.100s", raw)
    return None


# ---------------------------------------------------------------------------
# State extraction via LLM
# ---------------------------------------------------------------------------

def _extract_state_via_llm(
    messages: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to extract structured state from conversation history.
    Returns parsed JSON dict or None on failure.
    """
    history_str = _format_history_for_prompt(messages)
    prompt = STATE_EXTRACTION_PROMPT.format(conversation_history=history_str)

    raw = call_llm_fast(prompt, timeout=4, max_tokens=256)
    if not raw:
        return None

    # Strip markdown code fences if present
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        _log.debug("State extraction JSON parse failed: %.100s", raw)
        return None


# ---------------------------------------------------------------------------
# Turn classification (fast, no LLM)
# ---------------------------------------------------------------------------

def _classify_turn(
    user_message: str,
    state: ConversationState,
    has_previous_recs: bool,
    has_assistant_history: bool = False,
) -> str:
    """
    Classify the current turn into one of:
      clarify | recommend | refine | compare | refuse | close

    This is a fast, deterministic classifier — no LLM call.
    Guards have already been checked in process_chat(); here we only
    check the higher-level refusal classifier for edge cases.
    """
    # 1. Higher-level refusal (supplements guards already checked in process_chat)
    refusal_reason = classify_refusal(user_message)
    if refusal_reason:
        return "refuse"

    # 2. Comparison check
    if is_comparison_request(user_message) and extract_comparison_names(user_message):
        return "compare"

    # 3. Conversation close
    if state.conversation_complete and has_previous_recs:
        return "close"

    # 4. Refinement check — detect BEFORE clarification gating.
    #    Trigger on edit intent if we have prior recs OR prior assistant messages
    #    (because URL extraction from prior assistant messages may have failed)
    refinement_intent = detect_refinement_intent(user_message)
    if refinement_intent and (has_previous_recs or has_assistant_history):
        return "refine"

    # 5. If there is assistant history, the user is mid-conversation.
    #    Never send them back to clarification for short messages.
    if has_assistant_history and not is_vague_request(user_message):
        return "recommend"

    # 6. Clarification: is there enough context to recommend?
    if _needs_clarification(user_message, state, has_previous_recs):
        return "clarify"

    # 7. Default: recommend
    return "recommend"


def _needs_clarification(
    user_message: str,
    state: ConversationState,
    has_previous_recs: bool,
) -> bool:
    """
    Decide whether we need to ask a clarification question.

    Returns True (clarify) ONLY if:
    - Message is truly vague (< 4 meaningful words and no context signals)
    - No role, skills, responsibilities, or domain context can be inferred

    Returns False (proceed to recommend) if:
    - We have enough context (role + one more signal)
    - User provided a full job description
    - Already in mid-conversation with previous recs
    - User message has refinement intent (add/remove/replace)
    - User mentions soft skills, responsibilities, or domain context
    """
    if has_previous_recs:
        return False

    # Never clarify if message has refinement intent
    if detect_refinement_intent(user_message):
        return False

    # If message is moderately long (8+ words), there's likely enough context
    # for the LLM to infer a role. Previously this was 30, which was too strict.
    if len(user_message.split()) > 8:
        return False

    # Definitely vague
    if is_vague_request(user_message):
        return True

    # Have role and at least one more signal → enough to recommend
    has_role = bool(state.role or state.technical_skills)
    has_extra = bool(
        state.seniority
        or state.industry
        or state.needs_personality is not None
        or state.needs_cognitive is not None
        or state.safety_critical
        or state.purpose
        or state.languages
    )
    if has_role and has_extra:
        return False

    # Check for rich context signals: roles, soft skills, responsibilities,
    # domain terms. If ANY of these are present, proceed to recommend and
    # let the LLM infer the role profile.
    rich_context = re.search(
        # Explicit roles / job titles
        r"(java|python|sql|excel|contact.?cent|sales|customer service|"
        r"safety|chemical|graduate|engineer|developer|analyst|manager|"
        r"nurse|teacher|accountant|financial|leadership|executive|"
        r"founder|director|coordinator|supervisor|administrator|consultant|"
        r"architect|specialist|officer|recruiter|trainer|advisor|"
        # Soft skills & competencies
        r"leadership|communication|conflict|negotiation|decision.?making|"
        r"problem.?solv|critical.?think|strategic|emotional.?intellig|"
        r"team.?manage|team.?build|collaboration|influence|coaching|"
        r"mentoring|delegation|motivation|interpersonal|presentation|"
        r"facilitat|stakeholder|change.?manage|project.?manage|"
        # Responsibilities / actions
        r"manag|lead|hire|recruit|assess|evaluat|screen|develop|"
        r"oversee|supervis|coordinat|plan|strateg|budget|report|"
        r"mentor|coach|train|onboard|"
        # Domain / industry context
        r"startup|enterprise|remote|hybrid|agile|scrum|digital|"
        r"healthcare|banking|retail|manufacturing|logistics|pharma|"
        r"technology|fintech|edtech|consulting|government|"
        # Assessment-related terms
        r"personality|cognitive|psychometric|aptitude|competenc|"
        r"behavioral|situational|360|simulation)",
        user_message,
        re.IGNORECASE,
    )

    if rich_context:
        # We found context signals, but if the message is short (≤8 words)
        # and we only have a bare role with NO seniority/extra signals,
        # still ask for clarification. E.g., "Hiring a software engineer"
        # has 'engineer' but no seniority — should clarify.
        if has_role and not has_extra and len(user_message.split()) <= 8:
            return True  # Ask for seniority/level
        return False

    return True


# ---------------------------------------------------------------------------
# Catalog context builder for prompts
# ---------------------------------------------------------------------------

def _build_catalog_context(
    candidates: List[Dict[str, Any]],
    max_items: int = 20,
) -> str:
    """
    Build a compact JSON string of candidate assessments for prompt injection.
    Includes only fields relevant to the LLM.
    """
    compact = []
    for item in candidates[:max_items]:
        codes = keys_to_type_code(item.get("keys", []))
        compact.append({
            "entity_id": item.get("entity_id"),
            "name": item.get("name"),
            "url": item.get("link"),
            "test_type": codes,
            "keys": item.get("keys", []),
            "duration": item.get("duration") or "—",
            "job_levels": item.get("job_levels", []),
            "languages": item.get("languages", [])[:5],
            "description": (item.get("description") or "")[:200],
        })
    return json.dumps(compact, indent=2)


# ---------------------------------------------------------------------------
# Turn handlers
# ---------------------------------------------------------------------------

def _handle_clarification(
    user_message: str,
    state: ConversationState,
    messages: List[Dict[str, str]],
) -> ChatResponse:
    """
    Ask a targeted clarification question.
    Optimized for latency: skips all LLM and retrieval steps.
    """
    reply = _deterministic_clarification(state, user_message)
    return build_chat_response(reply=reply, is_clarification=True)


def _deterministic_clarification(
    state: ConversationState,
    user_message: str,
) -> str:
    """Build a clarification question without LLM."""
    if not state.role:
        return "Happy to help. What role are you hiring for?"
    if not state.seniority:
        return f"Got it — {state.role}. What seniority level is this? (e.g., entry, mid, senior, or leadership)"
    if not state.purpose:
        return "Is this for selection (hiring new candidates) or development (existing employees)?"
    return "Could you share more about the key requirements for this role?"


def _handle_recommend(
    user_message: str,
    state: ConversationState,
    messages: List[Dict[str, str]],
    previous_recs: List[Dict[str, str]],
) -> ChatResponse:
    """Generate a fresh recommendation shortlist."""
    from agent.recommendation_engine import build_retrieval_query
    query = build_retrieval_query(state, user_message)
    candidates = hybrid_retrieve(
        query=query,
        state_context=state.to_context_string(),
        job_levels=None,
        languages=state.languages or None,
        exclude_categories=state.excluded_categories or None,
        exclude_names=state.excluded_names or None,
        technical_skills=state.technical_skills or None,
        purpose=state.purpose,
        top_k=40,
    )

    # If catalog retrieval returned few/no results and we have a descriptive
    # message, use LLM to infer role profile and retry retrieval
    if len(candidates) < 3 and len(user_message.split()) > 5:
        _log.info("Weak catalog match (%d candidates). Attempting LLM role inference.", len(candidates))
        inferred = _infer_role_from_context(user_message)
        if inferred:
            _log.info("LLM inferred role context: %s", inferred)
            # Update state with inferred fields
            if inferred.get("role") and not state.role:
                state.role = inferred["role"]
            if inferred.get("seniority") and not state.seniority:
                state.seniority = inferred["seniority"]
            if inferred.get("skills"):
                for skill in inferred["skills"]:
                    if skill not in state.technical_skills:
                        state.technical_skills.append(skill)
            if inferred.get("needs_leadership"):
                state.needs_leadership = True
            if inferred.get("needs_personality"):
                state.needs_personality = True
            if inferred.get("needs_cognitive"):
                state.needs_cognitive = True

            # Retry retrieval with enriched state
            query = build_retrieval_query(state, user_message)
            candidates = hybrid_retrieve(
                query=query,
                state_context=state.to_context_string(),
                job_levels=None,
                languages=state.languages or None,
                exclude_categories=state.excluded_categories or None,
                exclude_names=state.excluded_names or None,
                technical_skills=state.technical_skills or None,
                purpose=state.purpose,
                top_k=40,
            )

    catalog_ctx = _build_catalog_context(candidates, max_items=25)
    history_str = _format_history_for_prompt(messages)

    prompt = ORCHESTRATION_PROMPT.format(
        catalog_context=catalog_ctx,
        state_context=state.to_context_string(),
        conversation_history=history_str,
    )

    raw = _call_llm(prompt)
    parsed = _parse_llm_response(raw)

    if parsed and parsed.get("recommendations"):
        items = _resolve_llm_recommendations(parsed["recommendations"])
        if items:
            # Apply domain-irrelevance filtering to LLM-resolved items
            items = _filter_domain_irrelevant(items, state)
            reply = parsed.get("reply", "Here are my recommended assessments.")
            eoc = parsed.get("end_of_conversation", False)
            return build_chat_response(reply=reply, items=items, end_of_conversation=eoc)

    # Fallback: use pure retrieval + ranker
    items = assemble_recommendations(
        user_message=user_message,
        state=state,
        previous_recommendations=None,
        max_results=10,
    )
    items = _filter_domain_irrelevant(items, state)
    reply = _build_recommendation_reply(state, items)
    return build_chat_response(reply=reply, items=items, end_of_conversation=False)


def _handle_refine(
    user_message: str,
    state: ConversationState,
    messages: List[Dict[str, str]],
    previous_recs: List[Dict[str, str]],
) -> ChatResponse:
    """Apply refinement to the existing shortlist.

    Key design rule: refinement UPDATES the shortlist, it does not restart
    retrieval from scratch.  The original role context (state.role,
    state.technical_skills, etc.) is preserved and used as the retrieval
    anchor so that results never drift into unrelated domains.
    """
    # Reconstruct previous items from rec dicts
    prev_items = _recs_to_items(previous_recs)

    # --- 1. Parse category-level additions from the message ---------------
    msg_lower = user_message.lower()
    if "personality" in msg_lower:
        state.needs_personality = True
        if "P" not in state.included_categories:
            state.included_categories.append("P")
    if "teamwork" in msg_lower or "team" in msg_lower:
        state.needs_personality = True  # teamwork measured via personality
        if "P" not in state.included_categories:
            state.included_categories.append("P")
    if "cognitive" in msg_lower or "reasoning" in msg_lower:
        state.needs_cognitive = True
    if "leadership" in msg_lower:
        state.needs_leadership = True
    if "sjt" in msg_lower or "situational" in msg_lower:
        state.needs_sjt = True
    if "communication" in msg_lower:
        # Communication is typically a knowledge test in the SHL catalog
        if "K" not in state.included_categories:
            state.included_categories.append("K")

    # --- 2. Apply structural refinement intent ---------------------------
    intent = detect_refinement_intent(user_message)
    if intent:
        action, target, replacement = intent
        target_lower = target.lower()
        if action == "add":
            # Interpret category-level additions (don't push raw category
            # descriptions into included_names — that pollutes retrieval)
            _is_category_add = False
            if "personality" in target_lower:
                state.needs_personality = True
                _is_category_add = True
            if "teamwork" in target_lower or "team" in target_lower:
                state.needs_personality = True
                _is_category_add = True
            if "cognitive" in target_lower or "reasoning" in target_lower:
                state.needs_cognitive = True
                _is_category_add = True
            if "sjt" in target_lower or "situational" in target_lower:
                state.needs_sjt = True
                _is_category_add = True
            if "communication" in target_lower:
                _is_category_add = True
            # Only push into included_names if it looks like a specific
            # assessment name (not a category description)
            if not _is_category_add and target not in state.included_names:
                state.included_names.append(target)
        elif action == "remove":
            if target not in state.excluded_names:
                state.excluded_names.append(target)

    # --- 3. Ensure state.role is always populated ----------------------------
    # Inline fallback: if regex state reconstruction missed the role, scan
    # each user message directly (handles cold-start or LLM failure cases).
    from agent.recommendation_engine import build_retrieval_query
    from agent.state import _extract_role as _state_extract_role
    if not state.role:
        for _m in messages:
            if _m["role"] == "user":
                _r = _state_extract_role(_m["content"])
                if _r:
                    state.role = _r
                    _log.info("Inline role fallback extracted: '%s'", state.role)
                    break

    # role_query: used for LLM orchestration prompt (includes refinement context)
    role_query = build_retrieval_query(state, user_message)

    # role_anchor_query: used for RETRIEVAL when no prior shortlist exists.
    # Anchored to role+state flags ONLY — deliberately excludes the raw
    # refinement message ("Add communication and strategic thinking...") so
    # that semantic search does not drift into unrelated catalog families
    # (e.g. Workplace Administration, BizTalk, Multitasking Ability).
    role_anchor_query = build_retrieval_query(state, state.role or "")

    # If we have no previous items to refine, generate a fresh shortlist
    # from state context (which now includes the refinement additions).
    # This path is PURE CATALOG — no LLM call — so it's always fast (<1s).
    if not prev_items:
        _log.info(
            "No previous shortlist. Pure-catalog refinement "
            "(role=%s, needs_leadership=%s, needs_personality=%s).",
            state.role, state.needs_leadership, state.needs_personality,
        )
        items = assemble_recommendations(
            user_message=role_anchor_query,  # Role-anchored, not refinement text
            state=state,
            previous_recommendations=None,
            max_results=10,
        )
        if items:
            items = _filter_domain_irrelevant(items, state)
            reply = _build_recommendation_reply(state, items)
            return build_chat_response(reply=reply, items=items, end_of_conversation=False)

        # Role known but no catalog matches — try with bare role string
        if state.role:
            items = assemble_recommendations(
                user_message=state.role,
                state=state,
                previous_recommendations=None,
                max_results=10,
            )
            items = _filter_domain_irrelevant(items, state)
            if items:
                reply = _build_recommendation_reply(state, items)
                return build_chat_response(reply=reply, items=items, end_of_conversation=False)

        # No context at all — ask clarification
        return build_chat_response(
            reply="Happy to help. Could you share the role and any key requirements?",
            is_clarification=True,
        )

    # --- 4. Retrieve new candidates anchored to original role context -----
    new_candidates = hybrid_retrieve(
        query=role_query,
        state_context=state.to_context_string(),
        technical_skills=state.technical_skills or None,
        languages=state.languages or None,
        exclude_categories=state.excluded_categories or None,
        exclude_names=state.excluded_names or None,
        purpose=state.purpose,
        needs_personality=state.needs_personality,
        needs_leadership=state.needs_leadership,
        top_k=30,
    )
    catalog_ctx = _build_catalog_context(new_candidates + prev_items, max_items=30)
    history_str = _format_history_for_prompt(messages)

    prompt = ORCHESTRATION_PROMPT.format(
        catalog_context=catalog_ctx,
        state_context=state.to_context_string(),
        conversation_history=history_str,
    )

    raw = call_llm_refine(prompt)
    parsed = _parse_llm_response(raw)

    if parsed and parsed.get("recommendations"):
        items = _resolve_llm_recommendations(parsed["recommendations"])
        if items:
            # Critical: apply domain-irrelevance filtering AFTER LLM
            # resolution so that sales/customer-service/manufacturing
            # items never survive when the role is tech/software.
            items = _filter_domain_irrelevant(items, state)
            if items:
                reply = parsed.get("reply", "Updated shortlist:")
                eoc = parsed.get("end_of_conversation", False)
                return build_chat_response(reply=reply, items=items, end_of_conversation=eoc)

    # Fallback: detect refinement intent and apply mechanically
    if intent:
        action, target, replacement = intent
        from agent.recommendation_engine import apply_refinement
        updated_items, msg = apply_refinement(
            action=action,
            target_name=target,
            replacement_name=replacement,
            current_shortlist=prev_items,
            state=state,
        )
        updated_items = _filter_domain_irrelevant(updated_items, state)
        reply = f"{msg} Updated shortlist:"
        return build_chat_response(reply=reply, items=updated_items, end_of_conversation=False)

    # No refinement detected — re-recommend using role context with
    # previous recommendations for continuity
    items = assemble_recommendations(
        user_message=role_query,  # Use role-based query, not refinement text
        state=state,
        previous_recommendations=previous_recs,
        max_results=10,
    )
    items = _filter_domain_irrelevant(items, state)
    reply = "Updated recommendations based on your request:"
    return build_chat_response(reply=reply, items=items, end_of_conversation=False)


def _handle_compare(
    user_message: str,
    state: ConversationState,
    messages: List[Dict[str, str]],
    previous_recs: List[Dict[str, str]],
) -> ChatResponse:
    """
    Handle a comparison request between two assessments.

    Per conversation traces (C3, C5, C6): comparison turns return
    recommendations=[] (empty) — the shortlist is NOT echoed.
    The user's shortlist is preserved in history for subsequent turns.
    """
    names = extract_comparison_names(user_message)

    if not names:
        reply = "I'd be happy to compare assessments. Could you name the two assessments you'd like to compare?"
        return build_chat_response(reply=reply, is_comparison=True)

    name_a, name_b = names
    item_a, item_b, ctx_a, ctx_b = build_comparison_context(name_a, name_b)

    # Try LLM comparison
    prompt = COMPARISON_PROMPT.format(assessment_a=ctx_a, assessment_b=ctx_b)
    raw = call_llm_compare(prompt)

    if raw and len(raw.strip()) > 50:
        reply = raw.strip()
    else:
        # Deterministic fallback
        reply = grounded_compare_fallback(name_a, name_b)

    # Comparison turns always return empty recommendations per spec
    return build_chat_response(reply=reply, is_comparison=True)


def _handle_close(
    user_message: str,
    state: ConversationState,
    previous_recs: List[Dict[str, str]],
) -> ChatResponse:
    """Confirm and close the conversation, repeating the final shortlist."""
    items = _recs_to_items(previous_recs)

    if items:
        reply = "Confirmed — shortlist locked in."
    else:
        reply = "Happy to help further whenever you need assessments."

    return build_chat_response(
        reply=reply,
        items=items if items else None,
        end_of_conversation=True,
    )


def _handle_refuse(user_message: str) -> ChatResponse:
    """Return a polite refusal for out-of-scope requests."""
    guard = check_guards(user_message)
    if guard.should_refuse:
        return build_chat_response(reply=guard.response, is_refusal=True)

    reason = classify_refusal(user_message) or "off_topic"
    response = build_refusal_response(reason=reason, original_message=user_message)
    return build_chat_response(reply=response, is_refusal=True)


# ---------------------------------------------------------------------------
# LLM response parsers
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse LLM JSON response from ORCHESTRATION_PROMPT.
    Returns dict or None on parse failure.
    """
    if not raw:
        return None

    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON object from mixed content
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    _log.debug("LLM response JSON parse failed: %.100s", raw)
    return None


def _parse_llm_reply(raw: Optional[str]) -> Optional[str]:
    """
    Extract the 'reply' field from an LLM JSON response,
    or return the raw text if it's a plain text response.
    """
    if not raw:
        return None

    parsed = _parse_llm_response(raw)
    if parsed and "reply" in parsed:
        return parsed["reply"]

    # If LLM returned plain text (not JSON), return it directly
    if raw and len(raw.strip()) > 10:
        # Remove any JSON artifacts
        cleaned = re.sub(r"^\s*\{.*?\}\s*$", "", raw.strip(), flags=re.DOTALL)
        if cleaned.strip():
            return cleaned.strip()

    return None


def _resolve_llm_recommendations(
    recs: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Resolve LLM-returned recommendation names/URLs to actual catalog items.
    Ensures no hallucinated items slip through.
    Also filters out generic report/guide products the LLM may have selected.
    """
    # Report/exercise filter for LLM-resolved items
    _REPORT_RE = re.compile(
        r"\breport\b|\bguide\b|\bprofiling\b|\bplanner\b"
        r"|\bremoteworkq\b|\bdigital readiness\b|\bhipo\b"
        r"|\b360\b|\bscenarios\b|\bglobal skills development\b"
        r"|\bexercises?\b|\bparticipant\b"
        r"|\bdevelopment cent(?:er|re)\b|\bassessment cent(?:er|re)\b"
        r"|\bdevelopment action\b|\btalent review\b|\bsuccession\b",
        re.IGNORECASE,
    )

    catalog = load_catalog()
    catalog_by_name = {normalize_text(item["name"]): item for item in catalog}
    catalog_by_url = {item.get("link", "").lower(): item for item in catalog}

    resolved = []
    for rec in recs:
        name = rec.get("name", "")
        url = rec.get("url", "")

        # Try exact name match first
        item = catalog_by_name.get(normalize_text(name))
        if item:
            # Filter out report products
            if _REPORT_RE.search(item.get("name", "")):
                _log.info("LLM recommended report product '%s' — filtering out.", item["name"])
                continue
            resolved.append(item)
            continue

        # Try URL match
        if url:
            item = catalog_by_url.get(url.lower().rstrip("/"))
            if item:
                if _REPORT_RE.search(item.get("name", "")):
                    _log.info("LLM recommended report product '%s' — filtering out.", item["name"])
                    continue
                resolved.append(item)
                continue

        # Try fuzzy name match
        item = get_item_by_name(name)
        if item:
            if _REPORT_RE.search(item.get("name", "")):
                _log.info("LLM recommended report product '%s' — filtering out.", item["name"])
                continue
            resolved.append(item)
            continue

        _log.warning(
            "LLM hallucinated assessment '%s' — not in catalog. Dropping.", name
        )

    return resolved


# Domain-irrelevance regex for tech roles — matches items from unrelated
# job families that should NEVER appear when the original role is in
# software / engineering / data / IT.
_IRRELEVANT_DOMAIN_RE = re.compile(
    r"\b(sales(?!force)|selling|sales.?transformation"
    r"|customer service|call cent|contact cent"
    r"|phone solution|phone simulation"
    r"|retail|merchandis|cashier|store|shop"
    r"|manufac|indust(?!ry)(?!rial engineering)"
    r"|mechanical.?(?:focus|vigilance)"
    r"|plant operator"
    r"|safety.?(?:and|&)?.?dependab|dependab.?(?:and|&)?.?safety"
    r"|workplace.?(?:health|safety|admin)|safety focus"
    r"|warehouse|logistics|forklift|driver"
    r"|nursing|nurse|healthcare aide|carer"
    r"|clerical|filing|receptionist"
    r"|food service|hospitality|housekeep"
    r"|entry.?level.?customer|entry.?level.?sales"
    r"|entry.?level.?cashier|entry.?level.?hotel"
    r"|\bjob.?control.?lang|\bjcl\b"
    r"|\bbiztalk\b"
    r"|salesforce.?develop"
    r"|multitask(?:ing)?.?abilit"
    r"|software.?business.?anal"
    r"|office.?admin|general.?admin(?:istration)?"
    r"|switchboard|mail.?clerk|data.?entry.?clerk)\b",
    re.IGNORECASE,
)

_TECH_ROLE_KEYWORDS = (
    "software", "engineer", "developer", "programmer", "coder",
    "data", "backend", "frontend", "fullstack", "devops", "sre",
    "architect", "tech", "it ", "computing", "cloud",
)


def _is_tech_context(state: ConversationState) -> bool:
    """
    Determine if the current conversation is about a tech/software role.
    Delegates to state.is_tech_domain() as the single source of truth.
    """
    return state.is_tech_domain()



def _filter_domain_irrelevant(
    items: List[Dict[str, Any]],
    state: ConversationState,
) -> List[Dict[str, Any]]:
    """
    Post-resolution domain-irrelevance HARD filter.

    When the conversation is about a tech/software role, REMOVE items whose
    name belongs to unrelated domains (sales, customer service, manufacturing,
    safety/dependability, phone solutions, etc.).

    This is the critical safety net that prevents domain drift during
    refinement turns where broad queries like "add personality assessments"
    pull in items from every job family.

    Domain-neutral items (OPQ32r, Verify G+, Business Communication) pass
    through because their names don't match the irrelevant-domain regex.
    """
    if not _is_tech_context(state):
        return items

    filtered = []
    for item in items:
        name = item.get("name", "")
        # Check name ONLY (not description) to avoid false positives on
        # generic descriptions that mention "customer" or "service" in passing
        if _IRRELEVANT_DOMAIN_RE.search(name):
            _log.info(
                "Domain filter REMOVED '%s' (tech context: role=%s, skills=%s)",
                name, state.role, state.technical_skills[:3] if state.technical_skills else [],
            )
            continue
        filtered.append(item)

    # Guard: if everything was filtered, return original to avoid empty list
    return filtered if filtered else items


def _recs_to_items(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert recommendation dicts (name, url, test_type) back to catalog items."""
    items = []
    seen_ids = set()
    for rec in recs:
        try:
            name = rec.get("name") if isinstance(rec, dict) else getattr(rec, "name", "")
        except (AttributeError, TypeError):
            continue
        if not name:
            continue
        item = get_item_by_name(name)
        if item and item["entity_id"] not in seen_ids:
            items.append(item)
            seen_ids.add(item["entity_id"])
    return items


def _build_recommendation_reply(
    state: ConversationState,
    items: List[Dict[str, Any]],
) -> str:
    """Build a natural language introduction for a recommendation list."""
    role = state.role or "this role"
    seniority = state.seniority or ""
    n = len(items)

    if not items:
        return "I wasn't able to find matching assessments in the catalog. Could you provide more details about the role?"

    intro = f"Here {'is' if n == 1 else 'are'} {n} assessment{'s' if n > 1 else ''}"
    if seniority and role != "this role":
        intro += f" for a {seniority}-level {role}"
    elif role != "this role":
        intro += f" for {role}"
    intro += "."

    return intro


# ---------------------------------------------------------------------------
# History formatter
# ---------------------------------------------------------------------------

def _format_history_for_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format the message list into a readable conversation string for prompt injection.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_chat(request: ChatRequest) -> ChatResponse:
    """
    Main orchestration function. Called by the FastAPI /chat endpoint.

    Workflow:
      1. Convert messages to dicts
      2. Extract state from conversation history
      3. Classify turn
      4. Route to appropriate handler
      5. Return ChatResponse

    This function is stateless — all context comes from request.messages.
    """
    t_start = time.time()

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Find the last user message (skip system messages)
    user_message = ""
    for m in reversed(messages):
        if m["role"] == "user":
            user_message = m["content"]
            break
    if not user_message:
        user_message = messages[-1]["content"]

    _log.info(
        "Processing chat. Turns: %d | Latest: %.60s",
        len(messages),
        user_message,
    )

    # Fast guard check (injection / off-topic)
    guard = check_guards(user_message)
    if guard.should_refuse:
        _log.info("Guard fired: %s", guard.reason)
        return build_chat_response(reply=guard.response, is_refusal=True)

    # Reconstruct conversation state (fast, regex-only pass first)
    state = reconstruct_state_from_history(messages, llm_state=None)

    # Extract previous history
    previous_recs = extract_previous_recommendations(messages[:-1])  # exclude current user msg
    has_previous_recs = bool(previous_recs)
    has_assistant_history = any(m["role"] == "assistant" for m in messages[:-1])

    # -----------------------------------------------------------------------
    # FAST-PATH DECISION TREE
    # ORDER MATTERS: refinement > clarification > slow-path LLM extraction
    # -----------------------------------------------------------------------

    # 1. Refusal check
    refusal_reason = classify_refusal(user_message)
    if refusal_reason:
        _log.info("Fast path: Refusal (%s)", refusal_reason)
        return _handle_refuse(user_message)

    # 2. Comparison check
    if is_comparison_request(user_message) and extract_comparison_names(user_message):
        _log.info("Fast path: Comparison")
        return _handle_compare(user_message, state, messages, previous_recs)

    # 3. Refinement check — uses DUAL detection (regex + keyword backup).
    #    Runs BEFORE clarification so mid-conversation adds/removes are never
    #    re-routed to the clarification path.
    #    Pure catalog path — no LLM state extraction call.
    if _is_refinement_turn(user_message, has_assistant_history):
        _log.info(
            "Fast path: Refinement (role=%s, skills=%s, keywords=%s)",
            state.role, state.technical_skills,
            list(set(user_message.lower().split()) & _REFINE_SIGNAL_WORDS)[:4],
        )
        return _handle_refine(user_message, state, messages, previous_recs)

    # 4. Clarification / Vague check
    # Only fires on the FIRST turn (no prior assistant message) when there
    # is insufficient context to generate a recommendation.
    if not has_assistant_history and _needs_clarification(user_message, state, has_previous_recs):
        _log.info("Fast path: Clarification")
        return _handle_clarification(user_message, state, messages)

    # -----------------------------------------------------------------------
    # SLOW PATH: Requires accurate state for recommendation/refinement
    # -----------------------------------------------------------------------
    _log.info("Fast path skipped. Escalating to LLM state extraction.")
    llm_state = _extract_state_via_llm(messages)
    if llm_state:
        # Merge LLM findings into our existing state object
        state = reconstruct_state_from_history(messages, llm_state=llm_state)

    # Re-classify the turn now that we have full LLM state
    turn_type = _classify_turn(user_message, state, has_previous_recs, has_assistant_history)
    _log.info("Turn classified as: %s (has_recs=%s, has_asst=%s)", turn_type, has_previous_recs, has_assistant_history)

    # Route to handler
    try:
        if turn_type == "close":
            response = _handle_close(user_message, state, previous_recs)
        elif turn_type == "refine":
            response = _handle_refine(user_message, state, messages, previous_recs)
        elif turn_type == "refuse":
            # Just in case classification caught something the fast path missed
            response = _handle_refuse(user_message)
        elif turn_type == "clarify":
            response = _handle_clarification(user_message, state, messages)
        elif turn_type == "compare":
            response = _handle_compare(user_message, state, messages, previous_recs)
        else:  # "recommend"
            response = _handle_recommend(user_message, state, messages, previous_recs)

    except Exception as e:
        _log.error("Handler error in turn '%s': %s", turn_type, e, exc_info=True)
        # Safe fallback: clarification response
        response = build_chat_response(
            reply="I encountered an issue processing your request. Could you rephrase it?",
            is_clarification=True,
        )

    elapsed = time.time() - t_start
    _log.info(
        "Response: action=%s, recs=%d, eoc=%s, elapsed=%.2fs",
        turn_type,
        len(response.recommendations),
        response.end_of_conversation,
        elapsed,
    )
    return response
