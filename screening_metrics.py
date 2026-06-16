"""
Evaluates the Qwen screening component (include/maybe/exclude)
against a gold-standard dataset such as CLEF TAR 2019 or CSMeD.

Rationale: WSS@95 is the standard metric for systematic review
screening automation. It answers: "what fraction of papers can
reviewers skip, while still catching 95% of truly relevant ones?"
This is what Research Synthesis Methods and CLEF TAR papers report.

Inputs expected:
  - predictions: list of dicts from your extractions/*.json files
  - gold: list of dicts from CLEF TAR 2019 or CSMeD relevance judgments
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, cohen_kappa_score
)


# --- Load your system's screening output ---
# Each extraction JSON has: paper_id, decision (include/maybe/exclude),
# relevance_score (0-5), ranked position from PubMedBERT

def load_extractions(extractions_dir: str) -> list[dict]:
    """Reads all extraction JSON files from one run folder."""
    records = []
    for f in Path(extractions_dir).glob("*.json"):
        with open(f) as fh:
            records.append(json.load(fh))
    return records


def load_gold_labels(gold_path: str) -> dict[str, int]:
    """
    Loads gold-standard labels.
    Format: {paper_id: 1 (included) or 0 (excluded)}
    For CLEF TAR: use the qrels files (PubMedID -> relevance).
    For CSMeD: use the included column from the dataset CSV.
    """
    with open(gold_path) as f:
        return json.load(f)


# --- Convert Qwen decisions to binary (include/maybe=1, exclude=0) ---
# Rationale: 'maybe' is treated as positive for high-recall evaluation.
# This is intentional — in SLR screening, false negatives (missing a
# relevant paper) are far costlier than false positives.

def decision_to_binary(decision: str) -> int:
    return 1 if decision in ("include", "maybe") else 0


# --- WSS@95 (Work Saved over Sampling at 95% recall) ---
# Formula from Cohen et al. (2006), standard in TAR literature:
#   WSS@R = (TN + FN) / N  -  (1 - R)
# where R = 0.95 target recall, N = total papers screened.

def wss_at_recall(y_true, y_scores, target_recall=0.95) -> float:
    """
    Computes WSS at a target recall level.
    y_scores: ranked confidence scores (higher = more likely relevant).
    Papers are screened in descending score order until target_recall is met.
    """
    n = len(y_true)
    # Sort by score descending (simulate human reviewing top-ranked first)
    order = np.argsort(y_scores)[::-1]
    y_true_sorted = np.array(y_true)[order]

    total_pos = sum(y_true)
    found, screened = 0, 0

    for label in y_true_sorted:
        screened += 1
        if label == 1:
            found += 1
        if total_pos > 0 and found / total_pos >= target_recall:
            break

    tn = n - screened - (total_pos - found)  # unscreened negatives
    fn = total_pos - found
    wss = (tn + fn) / n - (1 - target_recall)
    return max(wss, 0.0)


# --- Recall at rank cutoff ---
# Measures what fraction of relevant papers appear in top-k ranked results.
# Critical for validating the PubMedBERT ranking step.

def recall_at_k(y_true, y_scores, k: int) -> float:
    order = np.argsort(y_scores)[::-1]
    top_k = np.array(y_true)[order][:k]
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0
    return sum(top_k) / total_pos


# --- nDCG@k for PubMedBERT ranking ---
# Uses graded relevance (your 0-5 relevance_score) rather than binary,
# which is more informative than binary Precision@k alone.

def ndcg_at_k(y_true_graded, y_scores, k: int) -> float:
    order = np.argsort(y_scores)[::-1]
    y_sorted = np.array(y_true_graded, dtype=float)[order][:k]
    ideal = np.sort(y_true_graded)[::-1][:k]

    def dcg(scores):
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))

    idcg = dcg(ideal)
    return dcg(y_sorted) / idcg if idcg > 0 else 0.0


# --- Per-field extraction F1 ---
# Evaluates each structured field (e.g. study_design, population_or_sample)
# against a gold annotation using token-level overlap.

def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted and gold text for one field."""
    pred_toks = set(str(pred).lower().split())
    gold_toks = set(str(gold).lower().split())
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = pred_toks & gold_toks
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_extraction_fields(predictions: list[dict],
                                gold_annotations: list[dict],
                                fields: list[str]) -> dict:
    """
    Computes mean token F1 per extraction field.
    predictions: list of your extractions/*.json dicts
    gold_annotations: list of human-annotated dicts with same paper_id
    fields: list of field names to evaluate (e.g. ['study_design', 'objective'])
    """
    gold_map = {g["paper_id"]: g for g in gold_annotations}
    results = {f: [] for f in fields}

    for pred in predictions:
        pid = pred.get("paper_id")
        if pid not in gold_map:
            continue
        gold = gold_map[pid]
        for field in fields:
            p_val = pred.get(field, "")
            g_val = gold.get(field, "")
            results[field].append(token_f1(p_val, g_val))

    return {f: np.mean(scores) for f, scores in results.items() if scores}


# --- Main evaluation runner ---

def run_evaluation(extractions_dir: str, gold_path: str):
    preds = load_extractions(extractions_dir)
    gold_map = load_gold_labels(gold_path)

    # Match predictions to gold by paper_id / DOI
    y_true, y_pred_bin, y_scores = [], [], []
    for p in preds:
        pid = p.get("paper_id") or p.get("DOI")
        if pid not in gold_map:
            continue
        y_true.append(gold_map[pid])          # 1 or 0
        y_pred_bin.append(decision_to_binary(p.get("decision", "exclude")))
        y_scores.append(float(p.get("relevance_score", 0)))

    print("=== Screening metrics ===")
    print(classification_report(y_true, y_pred_bin,
                                 target_names=["exclude", "include"]))
    print(f"Cohen's Kappa:  {cohen_kappa_score(y_true, y_pred_bin):.3f}")
    print(f"WSS@95:         {wss_at_recall(y_true, y_scores, 0.95):.3f}")
    print(f"Recall@10:      {recall_at_k(y_true, y_scores, 10):.3f}")
    print(f"Recall@20:      {recall_at_k(y_true, y_scores, 20):.3f}")
    print(f"nDCG@10:        {ndcg_at_k(y_true, y_scores, 10):.3f}")
    print(f"nDCG@20:        {ndcg_at_k(y_true, y_scores, 20):.3f}")

    # Extraction field evaluation (requires separate gold annotations)
    fields_to_eval = [
        "study_design", "population_or_sample",
        "intervention_or_exposure", "outcomes_measured",
        "main_results", "objective"
    ]
    print("\n=== Extraction field token-F1 ===")
    print("(Requires human-annotated gold JSON with same paper_ids)")
    print("Pass gold_annotations list to evaluate_extraction_fields()")

    # Confidence distribution — report this as a transparency metric
    conf_counts = {"high": 0, "medium": 0, "low": 0}
    for p in preds:
        c = p.get("extraction_confidence", "low")
        conf_counts[c] = conf_counts.get(c, 0) + 1
    total = len(preds)
    print("\n=== Extraction confidence distribution ===")
    for k, v in conf_counts.items():
        print(f"  {k:8s}: {v:4d}  ({100*v/total:.1f}%)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractions", required=True,
                    help="Path to extractions/ folder from a pipeline run")
    ap.add_argument("--gold", required=True,
                    help="Path to gold/clef_tar_2019_labels.json")
    args = ap.parse_args()
    run_evaluation(args.extractions, args.gold)
