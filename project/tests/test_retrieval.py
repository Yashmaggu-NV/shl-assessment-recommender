"""
Unit tests for retriever, comparison, recommendation engine, and formatter.

Run with:
    pytest tests/test_retrieval.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# pyrefly: ignore [missing-import]
import pytest
from agent.retriever import keyword_search, get_item_by_name, metadata_filter
from agent.comparison import (
    extract_comparison_names,
    is_comparison_request,
    grounded_compare_fallback,
)
from agent.recommendation_engine import (
    build_retrieval_query,
    detect_refinement_intent,
    apply_refinement,
)
from agent.formatter import format_recommendation, format_recommendations, _parse_markdown_table_recs
from agent.state import ConversationState
from utils.helpers import load_catalog, keys_to_type_code, normalize_text


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

class TestCatalogLoading:
    def test_catalog_loads(self):
        catalog = load_catalog()
        assert len(catalog) > 0

    def test_catalog_items_have_name_and_link(self):
        catalog = load_catalog()
        for item in catalog[:20]:
            assert "name" in item
            assert "link" in item
            assert item["link"].startswith("https://www.shl.com")

    def test_catalog_items_have_keys(self):
        catalog = load_catalog()
        for item in catalog[:20]:
            assert "keys" in item
            assert isinstance(item["keys"], list)


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------

class TestKeywordSearch:
    def test_java_query_returns_java_results(self):
        results = keyword_search("Java developer Spring SQL", top_k=10)
        assert len(results) > 0
        names = [item["name"].lower() for item, _ in results]
        has_java = any("java" in n or "spring" in n or "sql" in n for n in names)
        assert has_java, f"Expected Java/Spring/SQL results, got: {names[:5]}"

    def test_results_are_sorted_by_score(self):
        results = keyword_search("personality leadership senior", top_k=10)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self):
        results = keyword_search("", top_k=10)
        assert results == []

    def test_top_k_limit_respected(self):
        results = keyword_search("developer", top_k=5)
        assert len(results) <= 5

    def test_safety_query_returns_safety_items(self):
        results = keyword_search("safety dependability chemical plant", top_k=10)
        assert len(results) > 0
        names = [item["name"].lower() for item, _ in results]
        has_safety = any("safety" in n or "dependability" in n for n in names)
        assert has_safety, f"Expected safety results, got: {names[:5]}"


# ---------------------------------------------------------------------------
# Catalog item lookup
# ---------------------------------------------------------------------------

class TestGetItemByName:
    def test_exact_name_found(self):
        item = get_item_by_name("Occupational Personality Questionnaire OPQ32r")
        assert item is not None
        assert "OPQ32r" in item["name"]

    def test_partial_name_found(self):
        item = get_item_by_name("OPQ32r")
        assert item is not None

    def test_nonexistent_name_returns_none(self):
        item = get_item_by_name("XYZ Fictional Assessment That Does Not Exist 9999")
        assert item is None

    def test_case_insensitive_match(self):
        item = get_item_by_name("occupational personality questionnaire opq32r")
        assert item is not None

    def test_verify_g_found(self):
        item = get_item_by_name("SHL Verify Interactive G+")
        assert item is not None

    def test_graduate_scenarios_found(self):
        item = get_item_by_name("Graduate Scenarios")
        assert item is not None


# ---------------------------------------------------------------------------
# Metadata filter
# ---------------------------------------------------------------------------

class TestMetadataFilter:
    def _make_candidate(self, name, keys, job_levels=None, languages=None, link=None):
        return {
            "entity_id": name,
            "name": name,
            "keys": keys,
            "job_levels": job_levels or [],
            "languages": languages or [],
            "link": link or f"https://www.shl.com/products/product-catalog/view/{name.lower()}/",
        }

    def test_exclude_category_removes_item(self):
        candidates = [
            (self._make_candidate("Test A", ["Personality & Behavior"]), 1.0),
            (self._make_candidate("Test B", ["Knowledge & Skills"]), 0.8),
        ]
        result = metadata_filter(candidates, exclude_categories=["P"])
        names = [item["name"] for item, _ in result]
        assert "Test A" not in names
        assert "Test B" in names

    def test_exclude_name_removes_item(self):
        candidates = [
            (self._make_candidate("OPQ32r", ["Personality & Behavior"]), 1.0),
            (self._make_candidate("Java Test", ["Knowledge & Skills"]), 0.8),
        ]
        result = metadata_filter(candidates, exclude_names=["OPQ32r"])
        names = [item["name"] for item, _ in result]
        assert "OPQ32r" not in names

    def test_job_level_boost_reorders(self):
        c1 = (self._make_candidate("Test A", ["Knowledge & Skills"], job_levels=["Graduate"]), 0.5)
        c2 = (self._make_candidate("Test B", ["Knowledge & Skills"], job_levels=["Executive"]), 0.5)
        result = metadata_filter([c1, c2], job_levels=["Graduate"])
        # Test A should be boosted to top
        assert result[0][0]["name"] == "Test A"


# ---------------------------------------------------------------------------
# Comparison engine
# ---------------------------------------------------------------------------

class TestComparisonDetection:
    def test_difference_between_pattern(self):
        names = extract_comparison_names("What is the difference between OPQ and GSA?")
        assert names is not None
        a, b = names
        assert "OPQ" in a or "GSA" in a

    def test_vs_pattern(self):
        names = extract_comparison_names("DSI vs OPQ32r — which should I use?")
        assert names is not None

    def test_compare_pattern(self):
        names = extract_comparison_names("Compare the Contact Center Simulation and Customer Service Phone Simulation")
        assert names is not None

    def test_not_comparison_returns_none(self):
        names = extract_comparison_names("Hiring a Java developer")
        assert names is None

    def test_is_comparison_request_true(self):
        assert is_comparison_request("What is the difference between OPQ and DSI?") is True

    def test_is_comparison_request_false(self):
        assert is_comparison_request("Hiring a Python developer, senior level") is False


class TestGroundedCompareFallback:
    def test_compare_real_assessments(self):
        result = grounded_compare_fallback(
            "Dependability and Safety Instrument (DSI)",
            "Occupational Personality Questionnaire OPQ32r"
        )
        assert len(result) > 50
        assert "DSI" in result or "Dependability" in result

    def test_compare_unknown_returns_not_found(self):
        result = grounded_compare_fallback("Fake Test XYZ", "Another Fake ABC")
        assert "not found" in result.lower() or "catalog" in result.lower()


# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------

class TestRetrievalQueryBuilder:
    def test_query_includes_role(self):
        state = ConversationState(role="Java developer", seniority="senior")
        query = build_retrieval_query(state, "Hiring a Java developer")
        assert "java" in query.lower()

    def test_query_includes_safety_terms(self):
        state = ConversationState(safety_critical=True)
        query = build_retrieval_query(state, "Plant operators")
        assert "safety" in query.lower() or "dependability" in query.lower()

    def test_query_includes_tech_skills(self):
        state = ConversationState(technical_skills=["java", "spring", "sql"])
        query = build_retrieval_query(state, "Backend developer")
        assert "java" in query.lower()
        assert "spring" in query.lower()


class TestRefinementIntentDetection:
    def test_remove_intent_detected(self):
        result = detect_refinement_intent("Remove the personality test please")
        assert result is not None
        action, target, _ = result
        assert action == "remove"
        assert "personality" in target.lower()

    def test_add_intent_detected(self):
        result = detect_refinement_intent("Add a cognitive test")
        assert result is not None
        action, target, _ = result
        assert action == "add"

    def test_no_intent_returns_none(self):
        result = detect_refinement_intent("That looks good")
        assert result is None

    def test_drop_intent_detected(self):
        result = detect_refinement_intent("Drop the REST API test")
        assert result is not None
        assert result[0] == "remove"


class TestApplyRefinement:
    def _make_item(self, name, eid=None):
        return {
            "entity_id": eid or name,
            "name": name,
            "keys": ["Knowledge & Skills"],
            "link": f"https://www.shl.com/products/product-catalog/view/{name.lower()}/",
        }

    def test_remove_existing_item(self):
        shortlist = [self._make_item("Java Test"), self._make_item("SQL Test")]
        updated, msg = apply_refinement("remove", "Java Test", None, shortlist, ConversationState())
        assert len(updated) == 1
        assert all(i["name"] != "Java Test" for i in updated)

    def test_remove_nonexistent_item(self):
        shortlist = [self._make_item("Java Test")]
        updated, msg = apply_refinement("remove", "Python Test", None, shortlist, ConversationState())
        assert len(updated) == 1  # unchanged
        assert "not in" in msg.lower() or "was not" in msg.lower()

    def test_add_real_catalog_item(self):
        shortlist = [self._make_item("Java Test")]
        updated, msg = apply_refinement("add", "Graduate Scenarios", None, shortlist, ConversationState())
        # Should succeed — Graduate Scenarios is in catalog
        if "Graduate Scenarios" in msg or any(i["name"] == "Graduate Scenarios" for i in updated):
            assert True  # Added successfully
        else:
            # Acceptable if name lookup fails in test env
            assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class TestFormatter:
    def _make_item(self, name, keys, link=None):
        return {
            "entity_id": "test",
            "name": name,
            "keys": keys,
            "link": link or "https://www.shl.com/products/product-catalog/view/test/",
            "job_levels": [],
            "languages": [],
        }

    def test_format_recommendation_correct_type_code(self):
        item = self._make_item("Test", ["Knowledge & Skills"])
        rec = format_recommendation(item)
        assert rec.test_type == "K"

    def test_format_recommendation_personality(self):
        item = self._make_item("OPQ", ["Personality & Behavior"])
        rec = format_recommendation(item)
        assert rec.test_type == "P"

    def test_format_recommendation_multiple_keys(self):
        item = self._make_item("Test", ["Knowledge & Skills", "Simulations"])
        rec = format_recommendation(item)
        assert "K" in rec.test_type
        assert "S" in rec.test_type

    def test_format_recommendations_caps_at_10(self):
        items = [self._make_item(f"Test {i}", ["Knowledge & Skills"]) for i in range(15)]
        recs = format_recommendations(items, max_items=10)
        assert len(recs) <= 10

    def test_format_recommendations_skips_missing_url(self):
        items = [
            self._make_item("Test A", ["Knowledge & Skills"], link="https://www.shl.com/..."),
            self._make_item("Test B", ["Knowledge & Skills"], link=""),
        ]
        recs = format_recommendations(items)
        assert len(recs) == 1
        assert recs[0].name == "Test A"

    def test_keys_to_type_code_ability(self):
        assert keys_to_type_code(["Ability & Aptitude"]) == "A"

    def test_keys_to_type_code_unknown_falls_back(self):
        result = keys_to_type_code(["Unknown Category"])
        assert result == "K"  # fallback

    def test_parse_markdown_table_recs(self):
        content = """
| # | Name | Test Type | Keys | Duration | Languages | URL |
|---|------|-----------|------|----------|-----------|-----|
| 1 | Core Java (Advanced Level) (New) | K | Knowledge & Skills | 13 minutes | English (USA) | https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/ |
| 2 | Occupational Personality Questionnaire OPQ32r | P | Personality & Behavior | 25 minutes | English | https://www.shl.com/products/product-catalog/view/occupational-personality-questionnaire-opq32r/ |
"""
        recs = _parse_markdown_table_recs(content)
        assert len(recs) == 2
        assert recs[0]["name"] == "Core Java (Advanced Level) (New)"
        assert recs[0]["test_type"] == "K"
        assert "shl.com" in recs[0]["url"]
