"""
Evaluates faithfulness of Qwen-generated outputs (detailed_summary,
results_summary, manuscript sections) against their source abstracts.

Rationale: Your system is abstract-limited by design. Any claim in
a generated summary not supported by the source abstract is a
hallucination. We quantify this using an LLM-as-judge pipeline,
which is standard in 2024-2025 summarization evaluation literature
(G-Eval, FaithEval, FaithBench EMNLP 2025).

Two layers:
  1. ROUGE-L + BERTScore: fast, lexical/semantic reference-based.
  2. LLM-as-judge faithfulness: claim-level hallucination detection.
"""

import json
import re
from pathlib import Path

import evaluate   # pip install evaluate
import ollama     # pip install ollama (reuse your local Qwen)


# --- Load ROUGE and BERTScore evaluators once ---
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")


# --- ROUGE-L and BERTScore ---
# Use the abstract as the reference text.
# The hypothesis is the detailed_summary or results_summary from extraction JSON.

def compute_reference_metrics(predictions: list[str],
                               references: list[str]) -> dict:
    """
    predictions: list of generated summaries (detailed_summary field)
    references:  list of source abstracts (used as gold)
    """
    rouge_out = rouge.compute(predictions=predictions,
                               references=references,
                               rouge_types=["rouge1", "rougeL"])
    bert_out = bertscore.compute(predictions=predictions,
                                  references=references,
                                  lang="en")
    return {
        "ROUGE-1":    round(rouge_out["rouge1"], 4),
        "ROUGE-L":    round(rouge_out["rougeL"], 4),
        "BERTScore-F1": round(sum(bert_out["f1"]) / len(bert_out["f1"]), 4),
    }


# --- LLM-as-judge faithfulness (via your local Qwen through Ollama) ---
# Rationale: We reuse the same qwen2.5:7b model already in your pipeline.
# The judge receives (source abstract, generated summary) and returns a
# sentence-level faithfulness label. This avoids paying for GPT-4o.
# Note: a 7B judge may miss subtle hallucinations. Report this limitation.

FAITHFULNESS_PROMPT = """\
You are a strict factuality evaluator for biomedical text.

SOURCE (abstract from paper):
{source}

GENERATED SUMMARY:
{summary}

Task: For each sentence in the GENERATED SUMMARY, decide if it is:
  SUPPORTED   - fully supported by the SOURCE
  PARTIAL     - partially supported (some facts correct, some not verifiable)
  HALLUCINATED - contains claims not present or contradicted by the SOURCE

Return a JSON list like:
[
  {{"sentence": "...", "label": "SUPPORTED"}},
  {{"sentence": "...", "label": "HALLUCINATED", "issue": "brief reason"}}
]
Return only valid JSON. No markdown fences."""


def judge_faithfulness(source_abstract: str,
                        generated_summary: str,
                        model: str = "qwen2.5:7b") -> dict:
    """
    Asks the local Qwen model to label each sentence in the summary.
    Returns a dict with per-label counts and an overall faithfulness ratio.
    """
    prompt = FAITHFULNESS_PROMPT.format(
        source=source_abstract,
        summary=generated_summary
    )
    response = ollama.generate(model=model, prompt=prompt)
    raw = response["response"].strip()

    # Strip markdown fences if present (Qwen sometimes adds them)
    raw = re.sub(r"```(?:json)?", "", raw).strip("` \n")

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "parse_failed", "raw": raw[:200]}

    counts = {"SUPPORTED": 0, "PARTIAL": 0, "HALLUCINATED": 0}
    for item in items:
        label = item.get("label", "HALLUCINATED").upper()
        counts[label] = counts.get(label, 0) + 1

    total = sum(counts.values())
    return {
        **counts,
        "faithfulness_ratio": counts["SUPPORTED"] / total if total > 0 else 0.0,
        "hallucination_rate": counts["HALLUCINATED"] / total if total > 0 else 0.0,
        "total_sentences": total,
    }


# --- Batch evaluation across all extraction files ---

def evaluate_all_summaries(extractions_dir: str,
                             model: str = "qwen2.5:7b") -> list[dict]:
    """
    Iterates over all extraction JSONs.
    Each file must have: abstract, detailed_summary, results_summary.
    Returns per-paper metrics and a dataset-level aggregate.
    """
    results = []
    files = list(Path(extractions_dir).glob("*.json"))

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        abstract = data.get("abstract", "")
        summary  = data.get("detailed_summary", "")
        if not abstract or not summary:
            continue

        # Reference-based metrics
        ref_metrics = compute_reference_metrics([summary], [abstract])

        # Faithfulness judge
        faith = judge_faithfulness(abstract, summary, model=model)

        results.append({
            "paper_id": data.get("paper_id"),
            "confidence": data.get("extraction_confidence"),
            **ref_metrics,
            **faith,
        })

    return results


def print_aggregate(results: list[dict]):
    """Prints mean metrics across all evaluated papers."""
    if not results:
        print("No results to aggregate.")
        return

    keys = ["ROUGE-1", "ROUGE-L", "BERTScore-F1",
            "faithfulness_ratio", "hallucination_rate"]
    for k in keys:
        vals = [r[k] for r in results if k in r and isinstance(r[k], float)]
        if vals:
            print(f"{k:25s}: {sum(vals)/len(vals):.4f}  (n={len(vals)})")

    # Stratify by extraction_confidence (high/medium/low)
    # Rationale: abstract-limited extractions should show lower faithfulness.
    for conf in ["high", "medium", "low"]:
        subset = [r for r in results if r.get("confidence") == conf]
        if subset:
            fr = [r["faithfulness_ratio"] for r in subset
                  if "faithfulness_ratio" in r]
            print(f"  confidence={conf}: faithfulness = "
                  f"{sum(fr)/len(fr):.3f}  (n={len(fr)})")


if __name__ == "__main__":
    import sys
    extractions_dir = sys.argv[1] if len(sys.argv) > 1 \
        else "runs/YOUR_RUN_FOLDER/extractions/"
    model = sys.argv[2] if len(sys.argv) > 2 else "qwen2.5:7b"

    print(f"Evaluating summaries in: {extractions_dir}\n")
    results = evaluate_all_summaries(extractions_dir, model=model)
    print_aggregate(results)

    with open("faithfulness_results.json", "w") as out:
        json.dump(results, out, indent=2)
    print("\nSaved to faithfulness_results.json")
