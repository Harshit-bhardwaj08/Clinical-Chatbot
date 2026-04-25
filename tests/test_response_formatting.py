"""Tests for intent-aware response formatting in rag_chain."""

from types import SimpleNamespace

from src.rag_chain import _shape_answer_for_intent


def _doc(text: str):
    return SimpleNamespace(page_content=text)


def test_definition_query_returns_concise_summary_without_bullets():
    question = "what is typhoid?"
    answer = "Symptoms of This condition include:\n- Typhoid\n- Symptoms\n- High fever"

    result = _shape_answer_for_intent(question=question, answer=answer, docs=[], history=[])

    assert "**Typhoid**" in result
    assert "- " not in result


def test_symptom_followup_uses_history_condition_name():
    question = "what are the symptoms"
    history = [{"role": "user", "content": "what is typhoid fever?"}]
    answer = "- High fever\n- Headache\n- Abdominal pain"

    result = _shape_answer_for_intent(question=question, answer=answer, docs=[], history=history)

    assert "Symptoms of **Typhoid fever** include:" in result
    assert "- High fever" in result
    assert "If symptoms persist or worsen, seek medical attention." in result


def test_symptom_answer_filters_generic_bullets():
    question = "what are symptoms of typhoid fever"
    answer = "Symptoms of This condition include:\n- Typhoid\n- Symptoms\n- High fever\n- Weakness"

    result = _shape_answer_for_intent(question=question, answer=answer, docs=[], history=[])

    assert "- Typhoid" not in result
    assert "- Symptoms" not in result
    assert "- High fever" in result
    assert "- Weakness" in result


def test_symptom_answer_can_use_docs_when_llm_has_no_bullets():
    question = "what are symptoms of dengue"
    answer = "Symptoms include:"
    docs = [
        _doc("Dengue fever symptoms include high fever, headache, joint pain, and rash."),
    ]

    result = _shape_answer_for_intent(question=question, answer=answer, docs=docs, history=[])

    assert "Symptoms of **Dengue** include:" in result
    assert "- high fever".lower() in result.lower()
    assert "- headache".lower() in result.lower()
