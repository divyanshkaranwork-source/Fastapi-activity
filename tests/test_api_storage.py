import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from app.llm import LLMConfigError
from tests.conftest import good_json


@pytest.fixture
def client_factory(make_analyzer):
    clients = []

    def _make(*replies):
        a, llm = make_analyzer(*replies)
        c = TestClient(create_app(a))
        c.__enter__()
        clients.append(c)
        return c, llm

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


def test_health(client_factory):
    c, _ = client_factory()
    body = c.get("/health").json()
    assert body["status"] == "ok" and body["database_ok"] is True


def test_analyze_ok_and_listing(client_factory):
    c, _ = client_factory(good_json(escalate_to_human=True))
    r = c.post("/analyze", json={"text": "App keeps crashing"})
    assert r.status_code == 200
    rec = r.json()
    assert rec["analysis"]["sentiment"] == "Negative" and rec["ticket"]["ticket_id"].startswith("TKT-")
    assert c.get(f"/analyses/{rec['id']}").json()["id"] == rec["id"]
    assert len(c.get("/analyses").json()) == 1
    assert c.get("/analyses/nope").status_code == 404


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": "hi", "mode": "bogus"}, {"text": "x" * 50_001}])
def test_analyze_validation_errors(client_factory, payload):
    c, llm = client_factory()
    assert c.post("/analyze", json=payload).status_code == 422
    assert llm.calls == []


def test_analyze_markup_only_is_422(client_factory):
    c, _ = client_factory()
    assert c.post("/analyze", json={"text": "<br><p></p>"}).status_code == 422


def test_config_error_is_503(client_factory):
    c, _ = client_factory(LLMConfigError("bad key"))
    r = c.post("/analyze", json={"text": "hello"})
    assert r.status_code == 503 and "bad key" in r.json()["detail"]


def test_escalation_queue_and_resolve(client_factory):
    c, _ = client_factory(good_json(urgency="Critical", category="Fraud & Security", suggested_team="Risk"), good_json())
    crit = c.post("/analyze", json={"text": "unauthorized debit"}).json()
    c.post("/analyze", json={"text": "app is slow"})
    queue = c.get("/escalation-queue").json()
    assert [q["id"] for q in queue] == [crit["id"]]
    tid = crit["ticket"]["ticket_id"]
    assert c.post(f"/tickets/{tid}/resolve").status_code == 200
    assert c.post(f"/tickets/{tid}/resolve").status_code == 404  # already resolved
    assert c.get("/escalation-queue").json() == []


def test_stats(client_factory):
    c, _ = client_factory()
    assert c.get("/stats").json()["total"] == 0 and c.get("/stats").json()["avg_confidence"] is None
    c.post("/analyze", json={"text": "app crashes"})
    s = c.get("/stats").json()
    assert s["total"] == 1 and s["sentiment"] == {"Negative": 1} and s["avg_confidence"] == 0.9


def _csv(c, content, name="f.csv", **params):
    return c.post("/analyze-batch", files={"file": (name, content, "text/csv")}, params=params)


def test_batch_csv(client_factory):
    c, _ = client_factory()
    r = _csv(c, "id,Feedback\n1,app crashes\n2,\n3,card declined\n")
    assert r.status_code == 200
    body = r.json()
    assert (body["total"], body["ok"], body["skipped"]) == (3, 2, 1)
    assert len(c.get("/analyses").json()) == 2


def test_batch_bom_and_quoted_multiline(client_factory):
    c, _ = client_factory()
    r = _csv(c, "﻿text\n\"line one\nline two, with comma\"\n".encode("utf-8"))
    assert r.status_code == 200 and r.json()["ok"] == 1


@pytest.mark.parametrize(
    "content, status",
    [
        ("a,b\n1,2\n", 400),  # no text column
        ("feedback\n", 400),  # no data rows
        (b"feedback\n\xff\xfe\n", 400),  # not UTF-8
        ("", 400),
    ],
)
def test_batch_bad_files(client_factory, content, status):
    c, llm = client_factory()
    assert _csv(c, content).status_code == status
    assert llm.calls == []


def test_batch_limits(client_factory, settings, make_analyzer):
    from dataclasses import replace

    from app.analyzer import Analyzer
    from app.storage import Store

    s = replace(settings, max_batch_rows=2, max_upload_bytes=1000)
    c = TestClient(create_app(Analyzer(make_analyzer()[1], s, Store(s.db_path))))
    with c:
        assert _csv(c, "feedback\na\nb\nc\n").status_code == 413
        assert _csv(c, "feedback\n" + "x" * 2000).status_code == 413


def test_batch_config_error_is_503(client_factory):
    c, _ = client_factory(LLMConfigError("bad key"))
    assert _csv(c, "feedback\none\ntwo\n").status_code == 503


def test_batch_invalid_mode(client_factory):
    c, _ = client_factory()
    assert _csv(c, "feedback\none\n", mode="nope").status_code == 422
