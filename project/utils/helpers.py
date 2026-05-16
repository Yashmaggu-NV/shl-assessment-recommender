"""
Utility helpers used across agent modules.
Provides catalog loading, type-mapping, text normalization, and logging setup.
"""

import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "catalog.json"
FAISS_INDEX_DIR = PROJECT_ROOT / "embeddings" / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_PATH = FAISS_INDEX_DIR / "meta.json"

# ---------------------------------------------------------------------------
# Category / test-type mapping
# ---------------------------------------------------------------------------

# Maps catalog "keys" strings → single-letter codes used in API responses
KEY_TO_CODE: Dict[str, str] = {
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Biodata & Situational Judgment": "B",
    "Simulations": "S",
    "Competencies": "C",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

CODE_TO_KEY: Dict[str, str] = {v: k for k, v in KEY_TO_CODE.items()}

# Human-readable aliases for code → user-facing label
CODE_LABEL: Dict[str, str] = {
    "K": "Knowledge & Skills",
    "A": "Ability & Aptitude",
    "P": "Personality & Behavior",
    "B": "Biodata & Situational Judgment",
    "S": "Simulations",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
}

# Aliases that users might say when referring to a category
CATEGORY_ALIASES: Dict[str, str] = {
    # personality
    "personality": "P",
    "personality test": "P",
    "personality tests": "P",
    "personality questionnaire": "P",
    "behaviour": "P",
    "behavioral": "P",
    "behavioural": "P",
    "opq": "P",
    # cognitive / ability
    "cognitive": "A",
    "cognitive test": "A",
    "aptitude": "A",
    "ability": "A",
    "reasoning": "A",
    "numerical reasoning": "A",
    "verbal reasoning": "A",
    "inductive reasoning": "A",
    "verify": "A",
    # knowledge
    "knowledge": "K",
    "skills test": "K",
    "technical test": "K",
    "knowledge test": "K",
    # situational judgment
    "situational judgment": "B",
    "situational judgement": "B",
    "sjt": "B",
    "biodata": "B",
    # simulation
    "simulation": "S",
    "simulations": "S",
    "sim": "S",
    # competency
    "competency": "C",
    "competencies": "C",
    # development
    "development": "D",
    "360": "D",
    "360 feedback": "D",
    # soft-skill aliases (map to closest SHL category)
    "teamwork": "P",
    "teamwork assessment": "P",
    "teamwork assessments": "P",
    "team": "P",
    "leadership": "P",
    "leadership assessment": "P",
    "leadership assessments": "P",
    "communication": "K",
    "communication assessment": "K",
    "communication skills": "K",
    "problem solving": "A",
    "problem-solving": "A",
    "conflict management": "P",
    "strategic thinking": "A",
    "interpersonal": "P",
    "emotional intelligence": "P",
}


def keys_to_type_code(keys: List[str]) -> str:
    """Convert a list of catalog 'keys' into a comma-separated type-code string."""
    codes = []
    for k in keys:
        code = KEY_TO_CODE.get(k)
        if code and code not in codes:
            codes.append(code)
    return ",".join(codes) if codes else "K"


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def load_catalog() -> List[Dict[str, Any]]:
    """
    Load and return the full SHL catalog from disk.
    Result is cached after first load.
    """
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = json.loads(f.read(), strict=False)
    _log.info("Loaded %d assessments from catalog.", len(data))
    return data


def get_catalog_by_id() -> Dict[str, Dict[str, Any]]:
    """Return catalog as a dict keyed by entity_id for O(1) lookup."""
    return {item["entity_id"]: item for item in load_catalog()}


def get_catalog_by_name() -> Dict[str, Dict[str, Any]]:
    """Return catalog as a dict keyed by normalized lowercase name."""
    return {normalize_text(item["name"]): item for item in load_catalog()}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, strip extra whitespace, remove punctuation for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str) -> List[str]:
    """Return meaningful tokens from a text string (stop-word light filter)."""
    STOP_WORDS = {
        "a", "an", "the", "and", "or", "but", "for", "in", "on", "at",
        "to", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might", "can", "could",
        "not", "no", "nor", "so", "yet", "both", "either", "neither",
        "i", "we", "you", "they", "he", "she", "it", "this", "that",
        "these", "those", "my", "our", "your", "their",
    }
    tokens = normalize_text(text).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def build_catalog_text(item: Dict[str, Any]) -> str:
    """
    Build a single searchable text blob for a catalog item.
    This is what gets embedded into the FAISS index.
    """
    parts = [
        item.get("name", ""),
        item.get("description", ""),
        " ".join(item.get("keys", [])),
        " ".join(item.get("job_levels", [])),
        " ".join(item.get("languages", [])),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Seniority helpers
# ---------------------------------------------------------------------------

SENIORITY_TO_JOB_LEVELS: Dict[str, List[str]] = {
    "entry": ["Entry-Level", "Graduate"],
    "graduate": ["Graduate", "Entry-Level"],
    "junior": ["Entry-Level", "Graduate", "Mid-Professional"],
    "mid": ["Mid-Professional", "Professional Individual Contributor"],
    "senior": ["Professional Individual Contributor", "Mid-Professional"],
    "lead": ["Professional Individual Contributor", "Manager", "Front Line Manager"],
    "manager": ["Manager", "Front Line Manager", "Mid-Professional"],
    "director": ["Director", "Manager"],
    "executive": ["Executive", "Director"],
    "cxo": ["Executive", "Director"],
    "c-suite": ["Executive", "Director"],
    "frontline": ["Front Line Manager", "Supervisor"],
    "supervisor": ["Supervisor", "Front Line Manager"],
}


def infer_job_levels(seniority_hint: Optional[str]) -> List[str]:
    """Map a free-text seniority hint to catalog job level strings."""
    if not seniority_hint:
        return []
    hint = seniority_hint.lower()
    for key, levels in SENIORITY_TO_JOB_LEVELS.items():
        if key in hint:
            return levels
    return []


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """Fetch an environment variable with an optional default."""
    val = os.environ.get(key, default)
    return val


def require_env(key: str) -> str:
    """Fetch a required environment variable; raise if missing."""
    val = os.environ.get(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val
