from evaluation.run_eval import compute_metrics, load_dataset, macro_f1
from feedbackiq.schema import CATEGORIES, SENTIMENTS, URGENCIES


def test_dataset_is_well_formed():
    rows = load_dataset()
    assert len(rows) >= 50
    assert len({r["text"] for r in rows}) == len(rows)
    for r in rows:
        assert r["sentiment"] in SENTIMENTS and r["urgency"] in URGENCIES and r["category"] in CATEGORIES
    assert sum(r["urgency"] == "Critical" for r in rows) >= 5


def test_dataset_does_not_leak_few_shot_examples():
    from feedbackiq.prompts import FEW_SHOT_EXAMPLES

    texts = {r["text"] for r in load_dataset()}
    assert not texts & {t for t, _ in FEW_SHOT_EXAMPLES}


def _res(valid=True, first=True, **pred):
    return {"valid": valid, "first_try_valid": first, "pred": pred, "latency_ms": 100, "cost_usd": 0.001, "tokens": 1}


def test_metrics_math():
    rows = [
        {"sentiment": "Negative", "urgency": "Critical", "category": "Fraud & Security"},
        {"sentiment": "Positive", "urgency": "Low", "category": "Other"},
        {"sentiment": "Neutral", "urgency": "Critical", "category": "Other"},
        {"sentiment": "Neutral", "urgency": "Low", "category": "Other"},
    ]
    res = [
        _res(sentiment="Negative", urgency="Critical", category="Fraud & Security"),
        _res(sentiment="Positive", urgency="Low", category="Other"),
        _res(sentiment="Neutral", urgency="High", category="Other"),  # missed Critical
        _res(valid=False, first=False),  # invalid output counts as wrong
    ]
    m = compute_metrics(rows, res)
    assert m["schema_valid_rate"] == 0.75 and m["first_try_valid_rate"] == 0.75
    assert m["sentiment_accuracy"] == 0.75 and m["urgency_accuracy"] == 0.5
    assert m["critical_recall"] == 0.5 and m["critical_missed"] == 1
    assert abs(m["urgency_weighted_accuracy"] - (3 + 1) / 8) < 1e-9


def test_macro_f1():
    assert macro_f1(["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "a"], ["b", "b"]) == 0.0
