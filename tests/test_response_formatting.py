"""Tests for intent-aware response formatting in rag_chain."""

from types import SimpleNamespace

from src.rag_chain import _final_quality_cleanup, _format_context, _shape_answer_for_intent


def _doc(text: str):
    return SimpleNamespace(page_content=text)


def test_definition_query_returns_concise_summary_without_bullets():
    question = "what is typhoid?"
    answer = "Symptoms of This condition include:\n- Typhoid\n- Symptoms\n- High fever"

    result = _shape_answer_for_intent(question=question, answer=answer, docs=[], history=[])

    assert "I do not have enough reliable information in the provided context to answer this question." in result
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

    assert "Symptoms of **Dengue fever** include:" in result
    assert "- high fever".lower() in result.lower()
    assert "- headache".lower() in result.lower()


def test_context_formatting_removes_dialogue_labels():
    docs = [
        _doc("Patient: I have chest pain. Doctor: Chest pain can be evaluated clinically."),
    ]

    result = _format_context(docs)

    assert "Patient:" not in result
    assert "Doctor:" not in result
    assert "Chest pain" in result


def test_final_cleanup_removes_duplicate_bullets_and_raw_labels():
    answer = (
        "Patient: Symptoms include:\n"
        "- Headache\n"
        "- Headache\n"
        "Doctor: This is not a diagnosis. Consult a healthcare professional."
    )

    result = _final_quality_cleanup(answer)

    assert "Patient:" not in result
    assert "Doctor:" not in result
    assert result.count("- Headache") == 1
