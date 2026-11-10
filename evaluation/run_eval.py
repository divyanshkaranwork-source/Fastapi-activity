"""Evaluate the three prompt modes against the labelled banking dataset.

    python -m evaluation.run_eval                 # all modes, live LLM (needs OPENAI_API_KEY)
    python -m evaluation.run_eval --modes full    # a single mode
    python -m evaluation.run_eval --limit 10      # quick smoke run

Metrics are computed on the validated MODEL output, before business rules, so the table
isolates the effect of prompt design.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from feedbackiq.analyzer import Analyzer
from feedbackiq.config import get_settings
from feedbackiq.llm import LLMConfigError, estimate_cost, get_llm_client
from feedbackiq.preprocess import preprocess
from feedbackiq.schema import CATEGORIES, MODES, URGENCIES

HERE = Path(__file__).parent
DATASET = HERE / "dataset.jsonl"
CRITICAL_WEIGHT = 3  # a missed/mislabelled Critical counts 3x in weighted urgency accuracy


def load_dataset(path: Path = DATASET) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        if r["urgency"] not in URGENCIES or r["category"] not in CATEGORIES:
            raise ValueError(f"Bad label in dataset row {r.get('id')}")
    return rows


def macro_f1(truth: list[str], pred: list[str]) -> float:
    labels = sorted(set(truth) | set(pred))
    scores = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(truth, pred))
        fp = sum(t != lab and p == lab for t, p in zip(truth, pred))
        fn = sum(t == lab and p != lab for t, p in zip(truth, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def compute_metrics(rows: list[dict], results: list[dict]) -> dict:
    """rows: ground truth; results: per-row dict(valid, first_try_valid, pred{...}, latency_ms, cost_usd, tokens)."""
    n = len(rows)
    valid = [(r, x) for r, x in zip(rows, results) if x["valid"]]
    out: dict = {
        "n": n,
        "schema_valid_rate": len(valid) / n if n else 0.0,
        "first_try_valid_rate": sum(x["first_try_valid"] for x in results) / n if n else 0.0,
    }
    # Invalid outputs count as wrong answers (denominator stays n).
    def acc(field: str) -> float:
        return sum(x["pred"][field] == r[field] for r, x in valid) / n if n else 0.0

    out["sentiment_accuracy"] = acc("sentiment")
    out["urgency_accuracy"] = acc("urgency")
    out["category_accuracy"] = acc("category")
    truth_c = [r["category"] for r, _ in valid]
    pred_c = [x["pred"]["category"] for _, x in valid]
    out["category_macro_f1"] = macro_f1(truth_c, pred_c)

    weights = [CRITICAL_WEIGHT if r["urgency"] == "Critical" else 1 for r in rows]
    correct = sum(w for (r, x), w in zip(zip(rows, results), weights) if x["valid"] and x["pred"]["urgency"] == r["urgency"])
    out["urgency_weighted_accuracy"] = correct / sum(weights) if weights else 0.0

    crit = [(r, x) for r, x in zip(rows, results) if r["urgency"] == "Critical"]
    caught = sum(1 for r, x in crit if x["valid"] and x["pred"]["urgency"] == "Critical")
    out["critical_total"] = len(crit)
    out["critical_recall"] = caught / len(crit) if crit else 1.0
    out["critical_missed"] = len(crit) - caught

    lat = [x["latency_ms"] for x in results if x["valid"]]
    out["avg_latency_ms"] = statistics.fmean(lat) if lat else 0.0
    out["p95_latency_ms"] = sorted(lat)[max(0, int(len(lat) * 0.95) - 1)] if lat else 0.0
    out["avg_cost_usd"] = statistics.fmean(x["cost_usd"] for x in results) if results else 0.0
    out["total_cost_usd"] = sum(x["cost_usd"] for x in results)
    return out


def run_mode(analyzer: Analyzer, rows: list[dict], mode: str, workers: int) -> list[dict]:
    def one(row: dict) -> dict:
        pre = preprocess(row["text"], analyzer.settings.max_input_chars)
        res = analyzer.classify(pre, mode)  # type: ignore[arg-type]
        ok = res.analysis is not None
        return {
            "id": row["id"],
            "valid": ok,
            "first_try_valid": ok and res.retries == 0,
            "pred": res.analysis.model_dump() if ok else {},
            "latency_ms": res.latency_ms,
            "cost_usd": estimate_cost(analyzer.settings, res.prompt_tokens, res.completion_tokens),
            "tokens": res.prompt_tokens + res.completion_tokens,
            "error": res.error,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, rows))


def render_table(metrics_by_mode: dict[str, dict]) -> str:
    cols = [
        ("Schema valid", "schema_valid_rate", "{:.1%}"),
        ("Valid 1st try", "first_try_valid_rate", "{:.1%}"),
        ("Sentiment acc", "sentiment_accuracy", "{:.1%}"),
        ("Urgency acc", "urgency_accuracy", "{:.1%}"),
        ("Urgency acc (Critical x3)", "urgency_weighted_accuracy", "{:.1%}"),
        ("Critical recall", "critical_recall", "{:.1%}"),
        ("Category acc", "category_accuracy", "{:.1%}"),
        ("Category macro-F1", "category_macro_f1", "{:.3f}"),
        ("Avg latency (ms)", "avg_latency_ms", "{:.0f}"),
        ("Avg cost ($)", "avg_cost_usd", "{:.5f}"),
    ]
    header = "| Mode | " + " | ".join(c[0] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [header, sep]
    for mode, m in metrics_by_mode.items():
        lines.append(f"| {mode} | " + " | ".join(fmt.format(m[key]) for _, key, fmt in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--limit", type=int, default=0, help="only the first N rows")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--allow-mock", action="store_true", help="permit LLM_PROVIDER=mock (pipeline test only)")
    args = ap.parse_args(argv)

    settings = get_settings()
    if settings.provider == "mock" and not args.allow_mock:
        print("Refusing to evaluate the offline mock: its numbers say nothing about GPT-4.1 mini.\n"
              "Set LLM_PROVIDER=openai and OPENAI_API_KEY, or pass --allow-mock to test the harness.", file=sys.stderr)
        return 2

    rows = load_dataset()
    if args.limit:
        rows = rows[: args.limit]
    analyzer = Analyzer(get_llm_client(settings), settings, store=None)

    metrics_by_mode: dict[str, dict] = {}
    raw: dict[str, list[dict]] = {}
    try:
        for mode in args.modes:
            print(f"Running {mode} on {len(rows)} messages ...", file=sys.stderr)
            raw[mode] = run_mode(analyzer, rows, mode, args.workers)
            metrics_by_mode[mode] = compute_metrics(rows, raw[mode])
    except LLMConfigError as exc:
        print(f"LLM configuration error: {exc}", file=sys.stderr)
        return 3

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = HERE / "results" / f"{stamp}_{settings.provider}_{analyzer.client.model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(
        json.dumps({"provider": settings.provider, "model": analyzer.client.model, "n": len(rows),
                    "metrics": metrics_by_mode, "per_item": raw}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nProvider: {settings.provider} | model: {analyzer.client.model} | messages: {len(rows)}\n")
    print(render_table(metrics_by_mode))
    print(f"\nSaved details to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
