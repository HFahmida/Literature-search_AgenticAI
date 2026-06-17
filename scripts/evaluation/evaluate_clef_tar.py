from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_K = [10, 20, 50, 100]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_topic_id(run_dir: Path, explicit_topic_id: str | None) -> str | None:
    if explicit_topic_id:
        return explicit_topic_id.upper()
    config_path = run_dir / "review_config.json"
    if not config_path.exists():
        return None
    config = load_json(config_path)
    label = str(config.get("run_label") or "").upper().strip()
    if label:
        return label
    topic = str(config.get("topic") or "")
    match = re.search(r"\bCD\d+\b", topic, flags=re.IGNORECASE)
    return match.group(0).upper() if match else None


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    files = [path] if path.is_file() else [p for p in path.rglob("*") if p.is_file()]
    qrels: dict[str, dict[str, int]] = {}
    for file_path in files:
        with file_path.open(encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                topic_id, doc_id, rel = parts[0].upper(), parts[2], parts[3]
                if not re.match(r"^-?\d+$", rel):
                    continue
                qrels.setdefault(topic_id, {})[normalize_doc_id(doc_id)] = 1 if int(rel) > 0 else 0
    return qrels


def normalize_doc_id(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"^pmid:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"https?://pubmed\.ncbi\.nlm\.nih\.gov/", "", text, flags=re.IGNORECASE)
    return text.strip("/").strip()


def candidate_doc_id(candidate: dict[str, Any]) -> str:
    source = str(candidate.get("source") or "").lower()
    source_id = normalize_doc_id(candidate.get("source_id"))
    if source == "pubmed" and source_id:
        return source_id
    if source_id and source_id.isdigit():
        return source_id
    raw = candidate.get("raw") or {}
    for key in ["pmid", "PMID", "pubmed_id"]:
        value = normalize_doc_id(raw.get(key))
        if value:
            return value
    return normalize_doc_id(candidate.get("paper_id") or candidate.get("doi") or candidate.get("title"))


def extraction_doc_id(extraction: dict[str, Any]) -> str:
    source = str(extraction.get("source") or "").lower()
    source_id = normalize_doc_id(extraction.get("source_id"))
    if source == "pubmed" and source_id:
        return source_id
    if source_id and source_id.isdigit():
        return source_id
    return normalize_doc_id(extraction.get("pubmed_id") or extraction.get("paper_id") or extraction.get("doi"))


def load_ranked_candidates(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "ranked_candidates.json"
    if not path.exists():
        return []
    rows = load_json(path)
    ranked = []
    for index, row in enumerate(rows, start=1):
        candidate = row.get("candidate", row)
        ranked.append(
            {
                "rank": index,
                "score": float(row.get("score", 0.0)),
                "candidate": candidate,
                "doc_id": candidate_doc_id(candidate),
            }
        )
    return ranked


def load_extractions(run_dir: Path) -> list[dict[str, Any]]:
    extractions_dir = run_dir / "extractions"
    if not extractions_dir.exists():
        return []
    rows = []
    for path in sorted(extractions_dir.glob("*.json")):
        extraction = load_json(path)
        screening = extraction.get("screening") or {}
        rows.append(
            {
                "doc_id": extraction_doc_id(extraction),
                "decision": str(screening.get("decision") or "exclude").lower().strip(),
                "relevance_score": float(screening.get("relevance_score") or 0),
                "title": extraction.get("title"),
                "source": extraction.get("source"),
                "source_id": extraction.get("source_id"),
            }
        )
    return rows


def binary_decision(decision: str) -> int:
    return 1 if decision in {"include", "maybe"} else 0


def precision_recall_f1(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def cohen_kappa(y_true: list[int], y_pred: list[int]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    observed = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    true_pos = sum(y_true) / n
    pred_pos = sum(y_pred) / n
    expected = true_pos * pred_pos + (1 - true_pos) * (1 - pred_pos)
    return (observed - expected) / (1 - expected) if expected < 1 else 0.0


def recall_at_k(labels: list[int], k: int) -> float:
    total_pos = sum(labels)
    return sum(labels[:k]) / total_pos if total_pos else 0.0


def ndcg_at_k(labels: list[int], k: int) -> float:
    actual = labels[:k]
    ideal = sorted(labels, reverse=True)[:k]

    def dcg(values: list[int]) -> float:
        return sum(value / math.log2(index + 2) for index, value in enumerate(values))

    ideal_dcg = dcg(ideal)
    return dcg(actual) / ideal_dcg if ideal_dcg else 0.0


def average_precision(labels: list[int]) -> float:
    total_pos = sum(labels)
    if not total_pos:
        return 0.0
    hits = 0
    precisions = []
    for index, label in enumerate(labels, start=1):
        if label:
            hits += 1
            precisions.append(hits / index)
    return mean(precisions) if precisions else 0.0


def wss_at_recall(labels: list[int], target_recall: float = 0.95) -> float:
    n = len(labels)
    total_pos = sum(labels)
    if n == 0 or total_pos == 0:
        return 0.0
    found = 0
    screened = 0
    for label in labels:
        screened += 1
        found += label
        if found / total_pos >= target_recall:
            break
    false_negatives = total_pos - found
    true_negatives_unseen = n - screened - false_negatives
    return max((true_negatives_unseen + false_negatives) / n - (1 - target_recall), 0.0)


def evaluate(run_dir: Path, qrels_path: Path, topic_id: str | None, k_values: list[int]) -> dict[str, Any]:
    qrels = load_qrels(qrels_path)
    resolved_topic_id = load_topic_id(run_dir, topic_id)
    if not resolved_topic_id:
        raise RuntimeError("Could not determine topic ID. Pass --topic-id CDxxxxxx.")
    if resolved_topic_id not in qrels:
        raise RuntimeError(f"Topic {resolved_topic_id} was not found in qrels: {qrels_path}")

    gold = qrels[resolved_topic_id]
    gold_positive = {doc_id for doc_id, label in gold.items() if label == 1}
    ranked = load_ranked_candidates(run_dir)
    extracted = load_extractions(run_dir)

    ranked_labels = [gold.get(row["doc_id"], 0) for row in ranked]
    retrieved_gold = {row["doc_id"] for row in ranked if row["doc_id"] in gold_positive}

    search_metrics = {
        "topic_id": resolved_topic_id,
        "gold_relevant_total": len(gold_positive),
        "ranked_candidate_total": len(ranked),
        "gold_relevant_retrieved": len(retrieved_gold),
        "search_recall": len(retrieved_gold) / len(gold_positive) if gold_positive else 0.0,
    }

    ranking_metrics = {
        "average_precision": average_precision(ranked_labels),
        "wss_at_95": wss_at_recall(ranked_labels, 0.95),
    }
    for k in k_values:
        ranking_metrics[f"recall_at_{k}"] = recall_at_k(ranked_labels, k)
        ranking_metrics[f"ndcg_at_{k}"] = ndcg_at_k(ranked_labels, k)

    matched_extractions = [row for row in extracted if row["doc_id"] in gold]
    y_true = [gold[row["doc_id"]] for row in matched_extractions]
    y_pred = [binary_decision(row["decision"]) for row in matched_extractions]
    y_scores = [row["relevance_score"] for row in matched_extractions]
    ranked_by_screen_score = [
        label for _, label in sorted(zip(y_scores, y_true), key=lambda item: item[0], reverse=True)
    ]
    screening_metrics = {
        "extractions_total": len(extracted),
        "extractions_matched_to_qrels": len(matched_extractions),
        "screening_wss_at_95": wss_at_recall(ranked_by_screen_score, 0.95),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        **precision_recall_f1(y_true, y_pred),
    }
    false_negatives = [
        row for row, truth, pred in zip(matched_extractions, y_true, y_pred) if truth == 1 and pred == 0
    ]

    return {
        "run_dir": str(run_dir),
        "qrels_path": str(qrels_path),
        "search": round_floats(search_metrics),
        "ranking": round_floats(ranking_metrics),
        "screening": round_floats(screening_metrics),
        "false_negative_titles": [row.get("title") for row in false_negatives[:25]],
        "notes": [
            "Ranking metrics are computed from ranked_candidates.json.",
            "Screening metrics are computed from extractions/*.json using screening.decision and screening.relevance_score.",
            "For systematic review screening, include and maybe are treated as positive decisions.",
        ],
    }


def round_floats(data: dict[str, Any]) -> dict[str, Any]:
    return {key: round(value, 4) if isinstance(value, float) else value for key, value in data.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one pipeline run against CLEF TAR qrels.")
    parser.add_argument("--run-dir", required=True, help="Pipeline run folder.")
    parser.add_argument("--qrels", required=True, help="CLEF TAR qrels file or directory.")
    parser.add_argument("--topic-id", default=None, help="CLEF topic ID, e.g. CD009044.")
    parser.add_argument("--k", nargs="*", type=int, default=DEFAULT_K, help="Rank cutoffs.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = evaluate(Path(args.run_dir), Path(args.qrels), args.topic_id, args.k)
    text = json.dumps(results, indent=2)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
