"""
Unit tests for the state reconstruction module.

Run with:
    pytest tests/test_state.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore [missing-import]
import pytest
from agent.state import (
    ConversationState,
    reconstruct_state_from_history,
    _extract_seniority,
    _extract_purpose,
    _extract_languages,
    _extract_tech_skills,
    _check_completion,
)


class TestSeniorityExtraction:
    def test_senior_keyword(self):
        assert _extract_seniority("Hiring a senior Java developer") == "senior"

    def test_junior_keyword(self):
        assert _extract_seniority("We need a junior developer") == "junior"

    def test_graduate_keyword(self):
        assert _extract_seniority("Hiring final-year graduates") == "graduate"

    def test_executive_keyword(self):
        assert _extract_seniority("CXO level candidates with 15+ years") == "executive"

    def test_years_experience_senior(self):
        assert _extract_seniority("7+ years of experience") == "senior"

    def test_years_experience_junior(self):
        assert _extract_seniority("1-2 years of experience") == "junior"

    def test_director_keyword(self):
        assert _extract_seniority("director level positions") == "director"

    def test_no_seniority_returns_none(self):
        assert _extract_seniority("Java developer") is None


class TestPurposeExtraction:
    def test_selection_keyword(self):
        assert _extract_purpose("This is for selection purposes") == "selection"

    def test_hiring_keyword(self):
        assert _extract_purpose("We are hiring 50 candidates") == "selection"

    def test_development_keyword(self):
        assert _extract_purpose("We want to develop our existing team") == "development"

    def test_reskilling_keyword(self):
        assert _extract_purpose("Annual talent audit to re-skill our Sales org") == "development"

    def test_no_purpose_returns_none(self):
        assert _extract_purpose("Java engineer") is None


class TestLanguageExtraction:
    def test_english_detected(self):
        langs = _extract_languages("All candidates speak English")
        assert "English" in langs

    def test_spanish_detected(self):
        langs = _extract_languages("Candidates need to be assessed in Spanish")
        assert "Spanish" in langs

    def test_multiple_languages(self):
        langs = _extract_languages("We need French and German assessments")
        assert "French" in langs
        assert "German" in langs

    def test_no_language_returns_empty(self):
        langs = _extract_languages("Hiring a Java developer")
        assert langs == []


class TestTechSkillExtraction:
    def test_java_detected(self):
        skills = _extract_tech_skills("Hiring a Java Spring developer")
        assert "java" in skills

    def test_multiple_skills(self):
        skills = _extract_tech_skills("Core Java, Spring, REST API design, Angular, SQL, AWS, Docker")
        for expected in ["java", "spring", "sql", "aws", "docker"]:
            assert expected in skills, f"Expected '{expected}' in {skills}"

    def test_no_skills_returns_empty(self):
        skills = _extract_tech_skills("We need a leadership assessment")
        assert skills == []

    def test_python_detected(self):
        skills = _extract_tech_skills("Python data scientist role")
        assert "python" in skills


class TestCompletionDetection:
    def _msgs(self, user_msg: str):
        return [{"role": "user", "content": user_msg}]

    def test_perfect_signals_completion(self):
        assert _check_completion(self._msgs("Perfect, that's what we need.")) is True

    def test_confirmed_signals_completion(self):
        assert _check_completion(self._msgs("Confirmed.")) is True

    def test_locking_in_signals_completion(self):
        assert _check_completion(self._msgs("Keep Verify G+. Locking it in.")) is True

    def test_thanks_signals_completion(self):
        assert _check_completion(self._msgs("That works. Thanks.")) is True

    def test_question_does_not_signal_completion(self):
        assert _check_completion(self._msgs("What about adding Python?")) is False

    def test_empty_messages(self):
        assert _check_completion([]) is False


class TestStateReconstruction:
    """Full state reconstruction from message history."""

    def _reconstruct(self, messages):
        return reconstruct_state_from_history(messages)

    def test_role_inferred_from_history(self):
        msgs = [{"role": "user", "content": "Hiring a Java developer"}]
        state = self._reconstruct(msgs)
        # Role may or may not be extracted without LLM — check it doesn't crash
        assert isinstance(state, ConversationState)

    def test_safety_critical_detected(self):
        msgs = [{"role": "user", "content": "Hiring plant operators for a chemical facility. Safety is top priority."}]
        state = self._reconstruct(msgs)
        assert state.safety_critical is True

    def test_leadership_detected(self):
        msgs = [{"role": "user", "content": "We need a leadership assessment for our executive team"}]
        state = self._reconstruct(msgs)
        assert state.needs_leadership is True

    def test_high_volume_detected(self):
        msgs = [{"role": "user", "content": "Screening 500 entry-level contact centre agents"}]
        state = self._reconstruct(msgs)
        assert state.volume == "high"

    def test_category_exclusion_from_history(self):
        msgs = [
            {"role": "user", "content": "Hiring a developer"},
            {"role": "assistant", "content": "I recommend these assessments..."},
            {"role": "user", "content": "Remove personality tests please"},
        ]
        state = self._reconstruct(msgs)
        assert "P" in state.excluded_categories

    def test_to_context_string_non_empty(self):
        msgs = [{"role": "user", "content": "Hiring a senior Java developer"}]
        state = self._reconstruct(msgs)
        ctx = state.to_context_string()
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_to_dict_has_all_keys(self):
        state = ConversationState()
        d = state.to_dict()
        expected_keys = [
            "role", "seniority", "industry", "languages",
            "needs_personality", "needs_cognitive", "needs_simulation",
            "needs_sjt", "needs_leadership", "safety_critical",
            "purpose", "volume", "included_names", "excluded_names",
            "included_categories", "excluded_categories",
            "technical_skills", "conversation_complete",
        ]
        for key in expected_keys:
            assert key in d, f"Missing key: {key}"
