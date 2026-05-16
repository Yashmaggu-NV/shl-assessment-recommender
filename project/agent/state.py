"""
Conversation state reconstruction module.

Reconstructs the complete hiring context from the full conversation history
on every stateless request. No server-side session storage is used.

State includes:
  - Role, seniority, industry
  - Language requirements
  - Category preferences (include / exclude)
  - Technical skills mentioned
  - Explicit assessment inclusions / exclusions
  - Assessment purpose (selection vs development)
  - Conversation completeness

Uses an LLM call to parse the history into structured JSON, with a
robust fallback regex extractor for common patterns.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from utils.helpers import (
    CATEGORY_ALIASES,
    get_logger,
    normalize_text,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class ConversationState:
    """Reconstructed hiring context from full message history."""

    role: Optional[str] = None
    seniority: Optional[str] = None           # entry/graduate/junior/mid/senior/lead/manager/director/executive
    industry: Optional[str] = None
    languages: List[str] = field(default_factory=list)

    needs_personality: Optional[bool] = None
    needs_cognitive: Optional[bool] = None
    needs_simulation: Optional[bool] = None
    needs_sjt: Optional[bool] = None
    needs_leadership: Optional[bool] = None
    safety_critical: Optional[bool] = None

    purpose: Optional[str] = None             # "selection" | "development"
    volume: Optional[str] = None              # "high" | "low"

    included_names: List[str] = field(default_factory=list)
    excluded_names: List[str] = field(default_factory=list)
    included_categories: List[str] = field(default_factory=list)
    excluded_categories: List[str] = field(default_factory=list)
    technical_skills: List[str] = field(default_factory=list)

    conversation_complete: bool = False

    # Canonical keywords for tech-role detection — ClassVar so dataclass ignores it
    _TECH_ROLE_KW: ClassVar[Tuple[str, ...]] = (
        "software", "engineer", "developer", "programmer", "coder",
        "backend", "frontend", "fullstack", "devops", "sre", "architect",
        "data scientist", "data engineer", "cloud", "ai engineer",
        "ml engineer", "machine learning", "tech lead", "computing",
    )

    def is_tech_domain(self) -> bool:
        """
        Single authoritative check: is this a technical/software role?

        Returns True if:
        - state.role contains any tech keyword, OR
        - state.technical_skills is non-empty (any specific tech skill mentioned)

        Used by query builder, retriever, ranker, and domain filter to ensure
        consistent domain classification across all pipeline stages.
        """
        if self.technical_skills:
            return True
        if self.role:
            role_lower = self.role.lower()
            return any(kw in role_lower for kw in self._TECH_ROLE_KW)
        return False

    def to_context_string(self) -> str:
        """Render state as a readable string for injection into LLM prompts."""
        parts = []
        if self.role:
            parts.append(f"Role: {self.role}")
        if self.seniority:
            parts.append(f"Seniority: {self.seniority}")
        if self.industry:
            parts.append(f"Industry: {self.industry}")
        if self.languages:
            parts.append(f"Required languages: {', '.join(self.languages)}")
        if self.purpose:
            parts.append(f"Purpose: {self.purpose}")
        if self.volume:
            parts.append(f"Volume: {self.volume}")
        if self.technical_skills:
            parts.append(f"Technical skills: {', '.join(self.technical_skills)}")
        flags = []
        if self.needs_personality is True:
            flags.append("personality required")
        elif self.needs_personality is False:
            flags.append("NO personality")
        if self.needs_cognitive is True:
            flags.append("cognitive required")
        elif self.needs_cognitive is False:
            flags.append("NO cognitive")
        if self.needs_simulation is True:
            flags.append("simulation required")
        if self.needs_sjt is True:
            flags.append("SJT required")
        if self.needs_leadership is True:
            flags.append("leadership assessment required")
        if self.safety_critical is True:
            flags.append("SAFETY-CRITICAL role")
        if flags:
            parts.append(f"Requirements: {'; '.join(flags)}")
        if self.included_names:
            parts.append(f"Confirmed in shortlist: {', '.join(self.included_names)}")
        if self.excluded_names:
            parts.append(f"Explicitly removed: {', '.join(self.excluded_names)}")
        if self.included_categories:
            parts.append(f"Included categories: {', '.join(self.included_categories)}")
        if self.excluded_categories:
            parts.append(f"Excluded categories: {', '.join(self.excluded_categories)}")
        return "\n".join(parts) if parts else "No context established yet."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "seniority": self.seniority,
            "industry": self.industry,
            "languages": self.languages,
            "needs_personality": self.needs_personality,
            "needs_cognitive": self.needs_cognitive,
            "needs_simulation": self.needs_simulation,
            "needs_sjt": self.needs_sjt,
            "needs_leadership": self.needs_leadership,
            "safety_critical": self.safety_critical,
            "purpose": self.purpose,
            "volume": self.volume,
            "included_names": self.included_names,
            "excluded_names": self.excluded_names,
            "included_categories": self.included_categories,
            "excluded_categories": self.excluded_categories,
            "technical_skills": self.technical_skills,
            "conversation_complete": self.conversation_complete,
        }


# ---------------------------------------------------------------------------
# Regex-based state extractor (fast, deterministic fallback / supplement)
# ---------------------------------------------------------------------------

# Seniority keywords
_SENIORITY_MAP = {
    r"\bentry.?level\b|\bjunior\b|\bjr\.?\b": "junior",
    r"\bgraduate\b|\bfresh(er)?\b|\bnew grad\b|\brecent grad\b": "graduate",
    r"\bmid.?level\b|\b4.?6 years?\b|\b3.?5 years?\b": "mid",
    r"\bsenior\b|\bsr\.?\b|\b7\+? years?\b|\b5\+? years?\b": "senior",
    r"\btech lead\b|\blead engineer\b|\bstaff engineer\b": "lead",
    r"\bmanager\b|\bteam lead\b|\bfront.?line manager\b": "manager",
    r"\bdirector\b": "director",
    r"\bcxo\b|\bc-suite\b|\bceo\b|\bcto\b|\bcoo\b|\bcfo\b|\bexecutive\b|\bvp\b|\bsvp\b": "executive",
    r"\bsupervisor\b": "supervisor",
}

# Purpose keywords
_PURPOSE_MAP = {
    r"\bselect\b|\bselection\b|\bhiring\b|\bscreening\b|\brecruit": "selection",
    r"\bdevelop\b|\bdevelopment\b|\bcoach\b|\bupskill\b|\breskill\b|\bauditing\b|\baudits?\b": "development",
}

# Safety-critical indicators
_SAFETY_RE = re.compile(
    r"\bsafety.?critical\b|\bhazard\b|\bchemical\b|\bnuclear\b|\bmanufact\b"
    r"|\bindustrial\b|\boil.?gas\b|\bmine\b|\bplant operator\b|\bfrontline worker\b"
    r"|\bdependability\b|\bprocedure compliance\b",
    re.IGNORECASE,
)

# Contact-centre indicators
_CONTACT_CENTRE_RE = re.compile(
    r"\bcontact.?cent(er|re)\b|\bcall cent(er|re)\b|\bcustomer service\b|\binbound call\b",
    re.IGNORECASE,
)

# Leadership indicators
_LEADERSHIP_RE = re.compile(
    r"\bleadership\b|\bcxo\b|\bc-suite\b|\bexecutive\b|\bdirector\b",
    re.IGNORECASE,
)

# High-volume indicators
_HIGHVOL_RE = re.compile(
    r"\b(high.?volume|mass hiring|\d{3,} candidates?|hundreds? of|thousands? of)\b",
    re.IGNORECASE,
)

# Language patterns
_LANG_RE = re.compile(
    r"\b(english|spanish|french|german|dutch|portuguese|arabic|chinese|japanese"
    r"|korean|italian|hindi|russian|turkish|polish|swedish|danish|norwegian|finnish)\b",
    re.IGNORECASE,
)

# Explicit removal patterns
_REMOVE_RE = re.compile(
    r"(?:remove|drop|exclude|skip|no|without|don't include|do not include|leave out)\s+(?:the\s+)?(.+?)(?:\s+test|\s+assessment|\s+from|\s+please|$)",
    re.IGNORECASE,
)

# Addition patterns
_ADD_RE = re.compile(
    r"(?:add|include|also add|also include|plus|with)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+test|\s+assessment|$)",
    re.IGNORECASE,
)

# Technical skill keywords
_TECH_SKILLS = {
    "java", "python", "javascript", "typescript", "rust", "go", "golang",
    "c++", "c#", "ruby", "php", "swift", "kotlin", "scala", "r",
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
    "spring", "django", "flask", "react", "angular", "vue", "node",
    "rest", "graphql", "kafka", "spark", "hadoop", "linux", "devops",
    "cicd", "ci/cd", "networking", "excel", "word", "powerpoint", "sap",
    "salesforce", "tableau", "power bi", "machine learning", "ml", "ai",
    "data science", "data engineering", "microservices",
}

# Role extraction patterns — ordered by specificity (more specific first)
_ROLE_PATTERNS = [
    # "hiring a/an X" or "need a/an X"
    (r"(?:hiring|recruit(?:ing)?|need(?:ing)?|looking for)\s+(?:a|an)\s+([\w\s/]+?(?:engineer|developer|programmer|architect|analyst|manager|lead|director|executive|scientist|specialist|consultant|designer|administrator|officer|devops|sre))", 1),
    # "for a/an X role/position"
    (r"(?:for|fill(?:ing)?)\s+(?:a|an)\s+([\w\s/]+?(?:engineer|developer|programmer|architect|analyst|manager|lead|director|executive|scientist|specialist|consultant|designer|administrator|officer|devops|sre))\s+(?:role|position|post|opening)", 1),
    # "X role" at start of sentence
    (r"^([\w\s/]+?(?:engineer|developer|programmer|architect|analyst|manager|lead|director|executive|scientist|specialist|consultant|designer))\s+(?:role|position|assessment|test)", 1),
    # Direct job titles
    (r"\b(software engineer|java developer|python developer|data scientist|data engineer|devops engineer|cloud engineer|frontend developer|backend developer|fullstack developer|ml engineer|ai engineer|platform engineer|site reliability engineer|solutions architect|product manager|engineering manager|tech lead|cto|vp of engineering)\b", 0),
]


def reconstruct_state_from_history(
    messages: List[Dict[str, str]],
    llm_state: Optional[Dict[str, Any]] = None,
) -> ConversationState:
    """
    Build a ConversationState from conversation history.

    If llm_state (from an LLM extraction call) is provided, it takes priority.
    Regex extraction supplements / fills gaps.

    Args:
        messages: Full conversation history (list of {role, content} dicts)
        llm_state: Optional structured JSON from LLM state extraction

    Returns:
        ConversationState populated with all inferred context
    """
    state = ConversationState()

    # Merge LLM extraction if available
    if llm_state:
        state = _merge_llm_state(state, llm_state)

    # Supplement with regex extraction over all user messages
    full_user_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    )
    full_all_text = " ".join(m["content"] for m in messages)

    # Seniority
    if not state.seniority:
        state.seniority = _extract_seniority(full_user_text)

    # Purpose
    if not state.purpose:
        state.purpose = _extract_purpose(full_user_text)

    # Safety critical
    if state.safety_critical is None and _SAFETY_RE.search(full_user_text):
        state.safety_critical = True

    # Leadership
    if state.needs_leadership is None and _LEADERSHIP_RE.search(full_user_text):
        state.needs_leadership = True

    # Volume
    if state.volume is None and _HIGHVOL_RE.search(full_user_text):
        state.volume = "high"

    # Languages
    if not state.languages:
        state.languages = _extract_languages(full_user_text)

    # Technical skills — scan all messages so skills mentioned in assistant
    # confirmation ("So you need Kubernetes experience...") are preserved.
    if not state.technical_skills:
        state.technical_skills = _extract_tech_skills(full_all_text)

    # Role — scan ALL messages so role is never lost when it was stated
    # in the first user turn and subsequent turns are refinements.
    if not state.role:
        # Prefer user-stated role; fall back to full conversation if not found
        state.role = _extract_role(full_user_text) or _extract_role(full_all_text)

    # Category exclusions from user messages
    _apply_category_edits(state, messages)

    # Category inclusions: scan user messages for explicit preference signals
    _apply_preference_signals(state, messages)

    # Conversation completion check (look for confirmation signals in last assistant message)
    if not state.conversation_complete:
        state.conversation_complete = _check_completion(messages)

    _log.debug("Reconstructed state: %s", state.to_dict())
    return state


def _merge_llm_state(state: ConversationState, llm: Dict[str, Any]) -> ConversationState:
    """Merge LLM-extracted JSON into the state object."""
    if llm.get("role"):
        state.role = llm["role"]
    if llm.get("seniority"):
        state.seniority = llm["seniority"]
    if llm.get("industry"):
        state.industry = llm["industry"]
    if llm.get("languages"):
        state.languages = llm["languages"]
    if llm.get("needs_personality") is not None:
        state.needs_personality = llm["needs_personality"]
    if llm.get("needs_cognitive") is not None:
        state.needs_cognitive = llm["needs_cognitive"]
    if llm.get("needs_simulation") is not None:
        state.needs_simulation = llm["needs_simulation"]
    if llm.get("needs_sjt") is not None:
        state.needs_sjt = llm["needs_sjt"]
    if llm.get("needs_leadership") is not None:
        state.needs_leadership = llm["needs_leadership"]
    if llm.get("safety_critical") is not None:
        state.safety_critical = llm["safety_critical"]
    if llm.get("purpose"):
        state.purpose = llm["purpose"]
    if llm.get("volume"):
        state.volume = llm["volume"]
    if llm.get("included_names"):
        state.included_names = llm["included_names"]
    if llm.get("excluded_names"):
        state.excluded_names = llm["excluded_names"]
    if llm.get("included_categories"):
        state.included_categories = llm["included_categories"]
    if llm.get("excluded_categories"):
        state.excluded_categories = llm["excluded_categories"]
    if llm.get("technical_skills"):
        state.technical_skills = llm["technical_skills"]
    if llm.get("conversation_complete") is not None:
        state.conversation_complete = llm["conversation_complete"]
    return state


def _extract_seniority(text: str) -> Optional[str]:
    """Extract seniority level from text using regex."""
    for pattern, level in _SENIORITY_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return level
    # Year-of-experience inference
    match = re.search(r"(\d+)\+?\s+years?\s+(?:of\s+)?experience", text, re.IGNORECASE)
    if match:
        years = int(match.group(1))
        if years <= 2:
            return "junior"
        elif years <= 5:
            return "mid"
        elif years <= 10:
            return "senior"
        else:
            return "lead"
    return None


def _extract_purpose(text: str) -> Optional[str]:
    """Extract selection vs development purpose from text."""
    for pattern, purpose in _PURPOSE_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            return purpose
    return None


def _extract_languages(text: str) -> List[str]:
    """Extract mentioned language requirements."""
    found = _LANG_RE.findall(text)
    # Normalize and deduplicate
    seen = set()
    result = []
    for lang in found:
        norm = lang.lower().capitalize()
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
    return result


def _extract_tech_skills(text: str) -> List[str]:
    """Extract mentioned technical skills."""
    text_lower = text.lower()
    found = []
    for skill in _TECH_SKILLS:
        # Use word boundary matching
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def _extract_role(text: str) -> Optional[str]:
    """
    Extract job role from free-form user text using regex patterns.

    Tries each pattern in _ROLE_PATTERNS in order, returning the first match.
    Cleans up common stopwords from the captured group.
    """
    for pattern, group_idx in _ROLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            role = match.group(group_idx).strip()
            # Clean trailing stopwords
            role = re.sub(
                r"\s+(role|position|post|opening|test|assessment|please|too|also)$",
                "", role, flags=re.IGNORECASE
            ).strip()
            if role and len(role) > 2:
                return role.lower()
    return None


def _apply_preference_signals(
    state: ConversationState,
    messages: List[Dict[str, str]],
) -> None:
    """
    Scan full message history for preference keywords and set state flags.

    This ensures that preferences expressed in any turn (not just the current
    turn) are preserved. For example, if the user said "add personality" in
    turn 3, and turn 5 is asking for refinement, state.needs_personality
    will still be True when turn 5 is processed.
    """
    full_user_text = " ".join(
        m["content"] for m in messages if m["role"] == "user"
    ).lower()

    if "personality" in full_user_text and state.needs_personality is None:
        state.needs_personality = True
    if ("teamwork" in full_user_text or "team work" in full_user_text) and state.needs_personality is None:
        state.needs_personality = True  # teamwork is measured via OPQ personality
    if ("cognitive" in full_user_text or "reasoning" in full_user_text) and state.needs_cognitive is None:
        state.needs_cognitive = True
    if "leadership" in full_user_text and state.needs_leadership is None:
        state.needs_leadership = True
    if ("communication" in full_user_text or "interpersonal" in full_user_text) and state.needs_personality is None:
        state.needs_personality = True  # Communication skills often overlap with personality


def _apply_category_edits(state: ConversationState, messages: List[Dict[str, str]]) -> None:
    """
    Scan message history for explicit category inclusion/exclusion commands.
    Updates state.included_categories and state.excluded_categories.
    Also handles needs_personality / needs_cognitive flags.
    """
    for msg in messages:
        if msg["role"] != "user":
            continue
        text = msg["content"].lower()

        # Removal patterns
        remove_match = _REMOVE_RE.search(text)
        if remove_match:
            removed_text = remove_match.group(1).lower()
            code = CATEGORY_ALIASES.get(removed_text)
            if code:
                if code not in state.excluded_categories:
                    state.excluded_categories.append(code)
                if code in state.included_categories:
                    state.included_categories.remove(code)
                # Update boolean flags
                if code == "P":
                    state.needs_personality = False
                elif code == "A":
                    state.needs_cognitive = False
                elif code == "S":
                    state.needs_simulation = False
                elif code == "B":
                    state.needs_sjt = False

        # Addition patterns
        add_match = _ADD_RE.search(text)
        if add_match:
            added_text = add_match.group(1).lower().strip()
            code = CATEGORY_ALIASES.get(added_text)
            if code:
                if code not in state.included_categories:
                    state.included_categories.append(code)
                if code in state.excluded_categories:
                    state.excluded_categories.remove(code)
                # Update boolean flags
                if code == "P":
                    state.needs_personality = True
                elif code == "A":
                    state.needs_cognitive = True
                elif code == "S":
                    state.needs_simulation = True
                elif code == "B":
                    state.needs_sjt = True


def _check_completion(messages: List[Dict[str, str]]) -> bool:
    """
    Infer if the conversation is complete by checking for confirmatory
    language in the most recent user message.
    """
    if not messages:
        return False
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    completion_signals = [
        r"\bperfect\b", r"\bthat.?s (it|all|good|great|fine|what we need)\b",
        r"\bconfirmed?\b", r"\blocking (it|this) in\b", r"\bkeep (it|this|them|the list)\b",
        r"\bgood (to go|choice|two.?stage|design)\b", r"\bthank(s| you)\b",
        r"\bthat works?\b", r"\bno (more )?changes?\b", r"\bfinal( list| shortlist)?\b",
    ]
    for signal in completion_signals:
        if re.search(signal, last_user, re.IGNORECASE):
            return True
    return False
