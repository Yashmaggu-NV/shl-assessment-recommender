"""
FAISS-based semantic retriever for SHL catalog assessments.

Provides:
  - Semantic similarity search using sentence-transformers embeddings
  - Metadata-aware filtering (job level, language, category)
  - Keyword overlap scoring for hybrid retrieval
  - Post-retrieval noise filtering (generic reports, 360s, exercises)
  - A combined hybrid_retrieve() method that fuses semantic + keyword signals

Index must be pre-built via embeddings/build_index.py before use.
Falls back to keyword-only search if FAISS index is not available.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from utils.helpers import (
    KEY_TO_CODE,
    FAISS_INDEX_PATH,
    FAISS_META_PATH,
    build_catalog_text,
    extract_keywords,
    get_logger,
    infer_job_levels,
    load_catalog,
    normalize_text,
)

_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports for heavy dependencies
# ---------------------------------------------------------------------------

_faiss = None
_model = None
_index = None
_meta: Optional[List[Dict[str, Any]]] = None
_catalog_lookup: Optional[Dict[str, Dict[str, Any]]] = None


def _get_faiss():
    global _faiss
    if _faiss is None:
        try:
            # pyrefly: ignore [missing-import]
            import faiss
            _faiss = faiss
        except ImportError:
            _log.warning("FAISS not installed. Falling back to keyword-only search.")
    return _faiss


def _get_model():
    global _model
    if _model is None:
        try:
            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
            _log.info("Loaded SentenceTransformer model.")
        except Exception as e:
            _log.warning("Could not load SentenceTransformer: %s", e)
    return _model


def _load_index():
    """Load the FAISS index and metadata from disk."""
    global _index, _meta, _catalog_lookup
    if _index is not None:
        return True

    faiss = _get_faiss()
    if faiss is None:
        return False

    if not FAISS_INDEX_PATH.exists() or not FAISS_META_PATH.exists():
        _log.warning("FAISS index not found at %s. Run build_index.py first.", FAISS_INDEX_PATH)
        return False

    try:
        _index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)
        # Build catalog lookup by entity_id
        catalog = load_catalog()
        _catalog_lookup = {item["entity_id"]: item for item in catalog}
        _log.info("Loaded FAISS index with %d vectors.", _index.ntotal)
        return True
    except Exception as e:
        _log.error("Failed to load FAISS index: %s", e)
        return False


# ---------------------------------------------------------------------------
# Generic / noise pattern definitions
# ---------------------------------------------------------------------------

# Name patterns that indicate generic report / non-assessment products
_GENERIC_NAME_PATTERNS = re.compile(
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

# Categories that are typically noise for technical hiring queries
_NOISE_CATEGORIES_FOR_TECH = {"D", "E", "C"}  # Development/360, Assessment Exercises, Competencies


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------

def semantic_search(
    query: str,
    top_k: int = 30,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Retrieve top-k catalog items by semantic similarity.

    Returns list of (catalog_item, score) tuples, score in [0, 1].
    Falls back to empty list if index unavailable.
    """
    model = _get_model()
    if not _load_index() or model is None:
        _log.debug("Semantic search unavailable — using keyword fallback.")
        return []

    try:
        vec = model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = _index.search(vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(_meta):
                continue
            entity_id = _meta[idx]["entity_id"]
            item = _catalog_lookup.get(entity_id)
            if item:
                # Convert L2 distance to similarity score in [0, 1]
                score = float(1.0 / (1.0 + dist))
                results.append((item, score))
        return results
    except Exception as e:
        _log.error("Semantic search error: %s", e)
        return []


def keyword_search(
    query: str,
    catalog: Optional[List[Dict[str, Any]]] = None,
    top_k: int = 50,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Score catalog items by keyword overlap with the query.
    Returns list of (catalog_item, score) tuples.
    """
    if catalog is None:
        catalog = load_catalog()

    query_tokens = set(extract_keywords(query))
    if not query_tokens:
        return []

    scored = []
    for item in catalog:
        item_text = build_catalog_text(item)
        item_tokens = set(extract_keywords(item_text))
        if not item_tokens:
            continue
        overlap = query_tokens & item_tokens
        # Jaccard-like score weighted by query coverage
        score = len(overlap) / (len(query_tokens) + 1e-9)
        if score > 0:
            scored.append((item, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def metadata_filter(
    candidates: List[Tuple[Dict[str, Any], float]],
    job_levels: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Apply metadata filters to a candidate list.
    Soft filters: items not matching get a penalty, not hard removal,
    to preserve recall when catalog data is incomplete.

    Args:
        candidates: List of (item, score) tuples
        job_levels: Preferred job levels to boost
        languages: Required languages (items missing all of them get penalised)
        include_categories: Only include items in these categories
        exclude_categories: Exclude items in these categories
        exclude_names: Explicitly excluded assessment names (exact match, case-insensitive)

    Returns:
        Filtered and re-scored candidate list (sorted by score, desc)
    """
    results = []
    exclude_names_norm = {normalize_text(n) for n in (exclude_names or [])}

    for item, score in candidates:
        item_name_norm = normalize_text(item.get("name", ""))

        # Hard exclude: explicitly removed names
        if item_name_norm in exclude_names_norm:
            continue

        item_keys = item.get("keys", [])
        item_codes = [KEY_TO_CODE.get(k) for k in item_keys if KEY_TO_CODE.get(k)]

        # Hard exclude: excluded categories
        if exclude_categories:
            if any(code in exclude_categories for code in item_codes):
                continue

        # Hard include: if include_categories set, only keep matching items
        if include_categories:
            if not any(code in include_categories for code in item_codes):
                continue

        # Soft boost: job level match
        if job_levels and item.get("job_levels"):
            if any(jl in item["job_levels"] for jl in job_levels):
                score *= 1.2

        # Soft boost: language match
        if languages and item.get("languages"):
            item_langs_lower = [l.lower() for l in item["languages"]]
            requested_langs_lower = [l.lower() for l in languages]
            if any(rl in " ".join(item_langs_lower) for rl in requested_langs_lower):
                score *= 1.1

        results.append((item, score))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def _post_filter_candidates(
    candidates: List[Tuple[Dict[str, Any], float]],
    technical_skills: Optional[List[str]] = None,
    purpose: Optional[str] = None,
    allow_generic: bool = False,
    needs_personality: Optional[bool] = None,
    needs_leadership: Optional[bool] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    Post-retrieval noise filter. Removes generic reports, development products,
    and assessment-centre exercises unless explicitly requested.

    Relevance floor: items scoring below 40% of the max score are dropped.
    """
    if not candidates:
        return []

    has_tech_skills = bool(technical_skills and len(technical_skills) > 0)
    is_development = purpose == "development"
    is_leadership = bool(needs_leadership)

    # Relevance floor: 40% of max score — stricter than before
    max_score = max(score for _, score in candidates)
    relevance_floor = max_score * 0.40

    filtered = []
    excluded_reasons = []

    for item, score in candidates:
        name = item.get("name", "")
        name_lower = name.lower()
        item_keys = item.get("keys", [])
        item_codes = set(KEY_TO_CODE.get(k) for k in item_keys if KEY_TO_CODE.get(k))

        # --- Relevance floor ---
        if score < relevance_floor:
            excluded_reasons.append((name, f"below relevance floor ({score:.3f} < {relevance_floor:.3f})"))
            continue

        # --- Skip generic filtering if explicitly allowed ---
        if allow_generic:
            filtered.append((item, score))
            continue

        # --- Generic report exclusion ---
        if _GENERIC_NAME_PATTERNS.search(name) and not is_development:
            has_relevant_code = bool(item_codes & {"K", "A", "S"})
            if has_tech_skills and has_relevant_code:
                desc_lower = (item.get("description") or "").lower()
                if any(skill.lower() in name_lower or skill.lower() in desc_lower
                       for skill in technical_skills):
                    filtered.append((item, score))
                    continue
            excluded_reasons.append((name, "generic report/360/exercise product"))
            continue

        # --- Development & 360 exclusion (unless purpose is development) ---
        if "D" in item_codes and not is_development:
            excluded_reasons.append((name, "Development & 360 product (purpose != development)"))
            continue

        # --- Assessment Exercises exclusion ---
        if "E" in item_codes and not is_development:
            excluded_reasons.append((name, "Assessment Exercise product"))
            continue

        # --- Personality penalty: only for pure tech-skill queries, NOT leadership ---
        # For leadership or explicitly requested personality, skip the penalty
        if has_tech_skills and not is_leadership and needs_personality is not True:
            if item_codes == {"P"}:
                score *= 0.4
                _log.debug("Penalised pure personality item '%s' for tech query", name)

        if has_tech_skills and item_codes == {"C"} and not is_leadership:
            score *= 0.3
            _log.debug("Penalised pure competency item '%s' for tech query", name)

        filtered.append((item, score))

    # Log exclusions
    if excluded_reasons:
        _log.info(
            "Post-filter excluded %d items: %s",
            len(excluded_reasons),
            "; ".join(f"'{n}' ({r})" for n, r in excluded_reasons[:10]),
        )

    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered


def hybrid_retrieve(
    query: str,
    state_context: Optional[str] = None,
    job_levels: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    include_categories: Optional[List[str]] = None,
    exclude_categories: Optional[List[str]] = None,
    exclude_names: Optional[List[str]] = None,
    technical_skills: Optional[List[str]] = None,
    purpose: Optional[str] = None,
    allow_generic: bool = False,
    needs_personality: Optional[bool] = None,
    needs_leadership: Optional[bool] = None,
    top_k: int = 40,
) -> List[Dict[str, Any]]:
    """
    Full hybrid retrieval pipeline:
      1. Semantic search (embedding similarity)
      2. Keyword search
      3. Score fusion
      4. Metadata filtering / scoring
      5. Post-retrieval noise filtering
      6. Return top-k items

    Args:
        query: The search query (user message or constructed retrieval query)
        state_context: Optional text context for enriching query
        job_levels: Job levels to filter/boost
        languages: Required languages
        include_categories: Category whitelist (codes)
        exclude_categories: Category blacklist (codes)
        exclude_names: Explicitly excluded assessment names
        technical_skills: Technical skills from state (for noise filtering)
        purpose: "selection" or "development" (for noise filtering)
        allow_generic: If True, don't filter out generic products
        top_k: Maximum results to return

    Returns:
        List of catalog item dicts, ranked by relevance.
    """
    # Enrich query with state context
    full_query = query
    if state_context:
        full_query = f"{query} {state_context}"

    catalog = load_catalog()

    # Run both search strategies
    semantic_results = semantic_search(full_query, top_k=50)
    keyword_results = keyword_search(full_query, catalog=catalog, top_k=50)

    # Fuse scores: 60% semantic, 40% keyword
    score_map: Dict[str, float] = {}
    item_map: Dict[str, Dict[str, Any]] = {}

    for item, score in semantic_results:
        eid = item["entity_id"]
        score_map[eid] = 0.6 * score
        item_map[eid] = item

    for item, score in keyword_results:
        eid = item["entity_id"]
        kw_contribution = 0.4 * score
        if eid in score_map:
            score_map[eid] += kw_contribution
        else:
            score_map[eid] = kw_contribution
            item_map[eid] = item

    # If no semantic results (index not ready), use keyword only
    if not semantic_results:
        score_map = {item["entity_id"]: score for item, score in keyword_results}
        item_map = {item["entity_id"]: item for item, _ in keyword_results}

    # Sort fused scores
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    candidates = [(item_map[eid], score) for eid, score in ranked if eid in item_map]

    # Debug log: raw retrieved candidates
    _log.info(
        "Raw retrieved: %d candidates (top-10: %s)",
        len(candidates),
        ", ".join(f"'{item_map[eid].get('name','')}' ({score:.3f})"
                 for eid, score in ranked[:10] if eid in item_map),
    )

    # Apply metadata filters
    filtered = metadata_filter(
        candidates,
        job_levels=job_levels,
        languages=languages,
        include_categories=include_categories,
        exclude_categories=exclude_categories,
        exclude_names=exclude_names,
    )

    # Apply post-retrieval noise filtering
    filtered = _post_filter_candidates(
        filtered,
        technical_skills=technical_skills,
        purpose=purpose,
        allow_generic=allow_generic,
        needs_personality=needs_personality,
        needs_leadership=needs_leadership,
    )

    # Debug log: filtered candidates
    _log.info(
        "After filtering: %d candidates (top-10: %s)",
        len(filtered),
        ", ".join(f"'{item.get('name','')}' ({score:.3f})"
                 for item, score in filtered[:10]),
    )

    return [item for item, _ in filtered[:top_k]]


def get_item_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Find a catalog item by exact or fuzzy name match."""
    catalog = load_catalog()
    name_norm = normalize_text(name)
    # Exact match first
    for item in catalog:
        if normalize_text(item["name"]) == name_norm:
            return item
    # Partial match
    for item in catalog:
        if name_norm in normalize_text(item["name"]):
            return item
    return None
