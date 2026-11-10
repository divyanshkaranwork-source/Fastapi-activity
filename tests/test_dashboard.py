"""Drives the Streamlit dashboard headlessly (mock LLM provider, temp database)."""

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard.py")


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("ASSESSMENT_DB", str(tmp_path / "dash.db"))
    monkeypatch.chdir(Path(DASHBOARD).parent)
    st.cache_resource.clear()
    at = AppTest.from_file(DASHBOARD, default_timeout=90)
    at.run()
    assert not at.exception
    yield at
    st.cache_resource.clear()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def textarea(at, label):
    return next(t for t in at.text_area if t.label == label)


def test_analyze_draft_and_correct_flow(run):
    at = run
    textarea(at, "Customer feedback").set_value(
        "My card was charged twice and your app keeps crashing. I'm closing my account today. Call 9876543210"
    )
    button(at, "Analyze").click().run()
    assert not at.exception and not at.error
    page = " ".join(m.value for m in at.markdown)
    assert "Negative" in page and "Retention" in page and "Recommended action" in page
    assert at.session_state["last_id"]

    # draft reply
    button(at, "Draft customer reply").click().run()
    assert not at.exception
    assert any("Dear Customer" in t.value for t in at.text_area)

    # human correction through the form
    urgency = next(s for s in at.selectbox if s.label == "Urgency")
    urgency.set_value("Critical")
    next(b for b in at.button if b.label == "Save correction").click().run()
    assert not at.exception
    store = at.session_state["last_id"]
    assert store
    page = " ".join(m.value for m in at.markdown)
    assert "Corrected by a human" in page


def test_empty_input_shows_error(run):
    button(run, "Analyze").click().run()
    assert run.error and "enter some feedback" in run.error[0].value


def test_cache_banner_on_repeat(run):
    at = run
    textarea(at, "Customer feedback").set_value("The app keeps crashing on login")
    button(at, "Analyze").click().run()
    button(at, "Analyze").click().run()
    assert "Served from cache" in " ".join(m.value for m in at.markdown)


def test_prompt_lab_compares_three_modes(run):
    at = run
    button(at, "Compare techniques").click().run()
    assert not at.exception and not at.error
    page = " ".join(m.value for m in at.markdown)
    for label in ("Zero-shot", "Few-shot", "Full (policy + examples)"):
        assert label in page


def test_analytics_and_queue_after_data(run):
    at = run
    textarea(at, "Customer feedback").set_value("Unauthorized debit of Rs 48,000 from my account. Block my card now!")
    button(at, "Analyze").click().run()
    at.run()
    assert not at.exception
    assert any(s.value == "Escalation queue (1 open)" for s in at.subheader)
    assert any(s.value == "Recent analyses" for s in at.subheader)
    button(at, "Mark resolved").click().run()
    assert not at.exception
    assert any(s.value == "Escalation queue (0 open)" for s in at.subheader)


def test_evaluation_tab_runs_with_mock(run):
    at = run
    button(at, "Run evaluation").click().run()
    assert not at.exception and not at.error
    for f in (Path(DASHBOARD).parent / "evaluation" / "results").glob("*mock*.json"):
        f.unlink()


def test_missing_key_message(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ASSESSMENT_DB", str(tmp_path / "d.db"))
    st.cache_resource.clear()
    at = AppTest.from_file(DASHBOARD, default_timeout=90).run()
    textarea(at, "Customer feedback").set_value("hello there")
    button(at, "Analyze").click().run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)
    st.cache_resource.clear()
