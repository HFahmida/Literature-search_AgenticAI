from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import httpx


PROMPT = """You are a strict factuality evaluator for biomedical literature summaries.

SOURCE ABSTRACT:
{abstract}

GENERATED SUMMARY:
{summary}

For each sentence in the GENERATED SUMMARY, label it as:
SUPPORTED - fully supported by the source abstract
PARTIAL - partly supported, but some details are not verifiable
UNSUPPORTED - not supported by the source abstract or contradicted by it

Return only valid JSON with this shape:
{{
  "sentence_labels": [
    {{"sentence": "text", "label": "SUPPORTED", "reason": "brief reason"}}
  ],
  "overall_faithfulness": "high, medium, or low"
}}
"""


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def call_ollama(prompt: str, model: str, base_url: str) -> dict[str, Any]:
    response = httpx.post(
        f"{base_url.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 2000},
        },
        timeout=600,
    )
    response.raise_for_status()
    text = response.json().get("response", "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def summary_text(extraction: dict[str, Any]) -> str:
    parts = [
        extraction.get("detailed_summary"),
        extraction.get("methods_summary"),
        extraction.get("results_summary"),
        extraction.get("conclusion_summary"),
    ]
    return "\n".join(str(part).strip() for part in parts if part)


def evaluate_file(path: Path, model: str, base_url: str) -> dict[str, Any]:
    extraction = load_json(path)
    abstract = str(extraction.get("abstract") or "").strip()
    summary = summary_text(extraction)
    result = {
        "file": str(path),
        "paper_id": extraction.get("paper_id"),
        "source": extraction.get("source"),
        "source_id": extraction.get("source_id"),
        "title": extraction.get("title"),
        "has_abstract": bool(abstract),
        "has_summary": bool(summary),
    }
    if not abstract or not summary:
        return {**result, "status": "skipped_missing_abstract_or_summary"}

    prompt = PROMPT.format(abstract=abstract, summary=summary)
    try:
        judged = call_ollama(prompt, model=model, base_url=base_url)
    except Exception as exc:
        return {**result, "status": "judge_failed", "error": repr(exc)}

    labels = judged.get("sentence_labels") or []
    counts = {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0}
    for item in labels:
        label = str(item.get("label") or "UNSUPPORTED").upper()
        counts[label] = counts.get(label, 0) + 1
    total = sum(counts.values())
    supported_ratio = counts["SUPPORTED"] / total if total else 0.0
    unsupported_ratio = counts["UNSUPPORTED"] / total if total else 0.0

    return {
        **result,
        "status": "ok",
        "supported": counts["SUPPORTED"],
        "partial": counts["PARTIAL"],
        "unsupported": counts["UNSUPPORTED"],
        "supported_ratio": round(supported_ratio, 4),
        "unsupported_ratio": round(unsupported_ratio, 4),
        "overall_faithfulness": judged.get("overall_faithfulness"),
        "sentence_labels": labels,
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in results if row.get("status") == "ok"]
    if not ok:
        return {"evaluated": 0, "skipped_or_failed": len(results)}
    return {
        "evaluated": len(ok),
        "skipped_or_failed": len(results) - len(ok),
        "mean_supported_ratio": round(sum(row["supported_ratio"] for row in ok) / len(ok), 4),
        "mean_unsupported_ratio": round(sum(row["unsupported_ratio"] for row in ok) / len(ok), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction-summary faithfulness against abstracts.")
    parser.add_argument("--extractions", required=True, help="Path to a run's extractions folder.")
    parser.add_argument("--model", default="qwen2.5:7b", help="Ollama judge model.")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL.")
    parser.add_argument("--out", default="faithfulness_results.json", help="Output JSON path.")
    args = parser.parse_args()

    extraction_paths = sorted(Path(args.extractions).glob("*.json"))
    results = [evaluate_file(path, args.model, args.base_url) for path in extraction_paths]
    output = {"aggregate": aggregate(results), "papers": results}
    text = json.dumps(output, indent=2, ensure_ascii=False)
    print(json.dumps(output["aggregate"], indent=2))
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
