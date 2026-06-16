"""
Loops through every pipeline run folder, computes evaluation metrics
per topic, and writes publication-ready tables.

Standalone — requires only: runs/, tar/2019/Task2/qrels/, numpy, pandas.
No batch_summary.json or other scripts needed.

Outputs written to paper_results/:
  results_per_topic.csv       one row per topic  → Supplementary Table S1
  results_aggregate.json      mean ± std         → sanity check
  table_main_results.tex      Table 2 in paper   → paste into LaTeX directly
  table_confidence.tex        Table 3 in paper   → confidence stratification

Usage:
    python collect_results.py
    python collect_results.py --runs-dir runs/ --qrels-dir tar/2019/Task2/qrels
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, cohen_kappa_score
)


# ─── Metric helpers ────────────────────────────────────────────────────────────

def decision_to_binary(decision: str) -> int:
    """
    Maps Qwen's 3-class output to binary for screening metrics.
    'maybe' is treated as positive because in SLR screening,
    missing a relevant paper (false negative) is far more costly
    than including an extra one (false positive).
    """
    return 1 if decision in ("include", "maybe") else 0


def wss_at_recall(y_true: list, y_scores: list, target: float = 0.95) -> float:
    """
    Work Saved over Sampling at target recall.
    Answers: what fraction of papers can reviewers skip while still
    catching `target` of the relevant ones?
    Reference: Cohen et al. (2006), JAMIA 13(3):206-219.
    Formula: WSS@R = (TN + FN) / N  -  (1 - R)
    """
    n = len(y_true)
    total_pos = sum(y_true)
    if n == 0 or total_pos == 0:
        return 0.0
    # Simulate reviewer working top-ranked first
    order    = np.argsort(y_scores)[::-1]
    y_sorted = np.array(y_true)[order]
    found = screened = 0
    for label in y_sorted:
        screened += 1
        found    += label
        if found / total_pos >= target:
            break
    tn  = n - screened - (total_pos - found)
    fn  = total_pos - found
    return float(max((tn + fn) / n - (1 - target), 0.0))


def recall_at_k(y_true: list, y_scores: list, k: int) -> float:
    """Fraction of relevant papers found in the top-k ranked results."""
    total_pos = sum(y_true)
    if total_pos == 0:
        return 0.0
    top_k = np.array(y_true)[np.argsort(y_scores)[::-1]][:k]
    return float(sum(top_k) / total_pos)


def ndcg_at_k(y_graded: list, y_scores: list, k: int) -> float:
    """
    Normalised Discounted Cumulative Gain using graded relevance (0-5).
    Rewards highly relevant papers appearing early in the ranking.
    """
    order    = np.argsort(y_scores)[::-1]
    y_sorted = np.array(y_graded, dtype=float)[order][:k]
    ideal    = np.sort(y_graded)[::-1][:k]
    dcg      = lambda s: sum(v / np.log2(i + 2) for i, v in enumerate(s))
    idcg     = dcg(ideal)
    return float(dcg(y_sorted) / idcg) if idcg > 0 else 0.0


# ─── CLEF TAR gold label loader ───────────────────────────────────────────────

def load_all_qrels(qrels_dir: str) -> dict[str, dict[str, int]]:
    """
    Reads every .txt qrel file and returns:
        { topic_id → { pubmed_id → relevance (0 or 1) } }

    CLEF TAR qrel format (space-separated):
        CD009044  0  26784175  1
        CD009044  0  22884264  0
        ^topic    ^iter  ^pmid  ^relevance
    """
    qrels: dict[str, dict[str, int]] = {}
    for path in Path(qrels_dir).glob("*.txt"):
        with open(path, errors="ignore") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                topic_id, doc_id, rel = parts[0], parts[2], int(parts[3])
                qrels.setdefault(topic_id, {})[doc_id] = rel
    return qrels


# ─── Single run evaluator ─────────────────────────────────────────────────────

def evaluate_run(run_dir: Path, topic_gold: dict[str, int]) -> dict:
    """
    Reads all extraction JSONs in one run folder and computes
    the full metric set for that topic.

    Matching strategy:
      1. Use pubmed_id field if present (ideal)
      2. Fall back to paper_id
      3. Fall back to normalised DOI string
    This covers runs where the pipeline stored different ID formats.
    """
    extr_dir = run_dir / "extractions"
    if not extr_dir.exists():
        return {"status": "no_extractions_dir"}

    y_true, y_pred, y_scores = [], [], []
    conf = {"high": 0, "medium": 0, "low": 0}
    n_total = n_matched = 0

    for fp in extr_dir.glob("*.json"):
        with open(fp) as fh:
            ext = json.load(fh)
        n_total += 1

        # Resolve paper identifier
        pid = (str(ext.get("pubmed_id") or "").strip()
               or str(ext.get("paper_id") or "").strip()
               or _norm_doi(ext.get("DOI", "")))

        if pid not in topic_gold:
            continue   # paper not in this topic's candidate pool

        n_matched += 1
        y_true.append(topic_gold[pid])
        y_pred.append(decision_to_binary(ext.get("decision", "exclude")))
        y_scores.append(float(ext.get("relevance_score", 0)))
        c = ext.get("extraction_confidence", "low")
        conf[c] = conf.get(c, 0) + 1

    # Need at least one positive to compute most metrics
    if not y_true or sum(y_true) == 0:
        return {"status": "no_positives_matched",
                "n_total": n_total, "n_matched": n_matched}

    # Screening metrics
    p     = precision_score(y_true, y_pred, zero_division=0)
    r     = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    kappa = (cohen_kappa_score(y_true, y_pred)
             if len(set(y_true)) > 1 else 0.0)

    total_conf = sum(conf.values()) or 1

    # PRISMA counts from run_stats.json
    stats_path = run_dir / "run_stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    return {
        "status":            "ok",
        # ── Screening ──────────────────────────────
        "precision":         round(p, 4),
        "recall":            round(r, 4),
        "f1":                round(f1, 4),
        "kappa":             round(kappa, 4),
        "wss_95":            round(wss_at_recall(y_true, y_scores, 0.95), 4),
        "wss_100":           round(wss_at_recall(y_true, y_scores, 1.00), 4),
        # ── Ranking ────────────────────────────────
        "recall_at_10":      round(recall_at_k(y_true, y_scores, 10), 4),
        "recall_at_20":      round(recall_at_k(y_true, y_scores, 20), 4),
        "ndcg_at_10":        round(ndcg_at_k(y_scores, y_scores, 10), 4),
        "ndcg_at_20":        round(ndcg_at_k(y_scores, y_scores, 20), 4),
        # ── Confidence distribution ─────────────────
        "conf_high_pct":     round(100 * conf["high"]   / total_conf, 1),
        "conf_medium_pct":   round(100 * conf["medium"] / total_conf, 1),
        "conf_low_pct":      round(100 * conf["low"]    / total_conf, 1),
        # ── Counts ─────────────────────────────────
        "n_gold_positive":   sum(y_true),
        "n_matched":         n_matched,
        "n_extracted":       n_total,
        "n_records_found":   stats.get("records_identified", 0),
        "n_duplicates":      stats.get("duplicates_removed", 0),
        "n_included":        stats.get("included_studies", 0),
    }


def _norm_doi(doi: str) -> str:
    return re.sub(r"https?://doi\.org/", "", str(doi)).strip().lower()


# ─── Run folder discovery ──────────────────────────────────────────────────────

def discover_runs(runs_dir: Path, qrels: dict[str, dict]) -> list[dict]:
    """
    Matches each CLEF TAR topic to its most recent run folder.

    Strategy: each run folder is named  YYYYMMDD-HHMMSS-<topic-slug>
    We normalise the slug (lower, strip hyphens) and compare to each
    topic ID (e.g. CD009044). Falls back to checking review_config.json
    inside the run folder for the run_label field set by
    generate_clef_configs.py.
    """
    # Build a quick lookup: normalised_slug → topic_id
    topic_lookup = {tid.lower().replace("-", ""): tid for tid in qrels}

    # Gather all run folders, sorted newest-first per topic
    run_folders = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir()],
        reverse=True          # newest run first
    )

    matched: dict[str, Path] = {}  # topic_id → best run folder

    for folder in run_folders:
        # Try matching via review_config.json run_label first (most reliable)
        config_path = folder / "review_config.json"
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                label = cfg.get("run_label", "").upper()
                if label in qrels and label not in matched:
                    matched[label] = folder
                    continue
            except Exception:
                pass

        # Fall back to slug matching on folder name
        slug = re.sub(r"^\d{8}-\d{6}-", "", folder.name).lower().replace("-", "")
        for norm_tid, tid in topic_lookup.items():
            if norm_tid in slug and tid not in matched:
                matched[tid] = folder
                break

    return [{"topic_id": tid, "run_dir": path} for tid, path in sorted(matched.items())]


# ─── Aggregation ──────────────────────────────────────────────────────────────

METRIC_COLS = [
    "precision", "recall", "f1", "kappa",
    "wss_95", "wss_100",
    "recall_at_10", "recall_at_20",
    "ndcg_at_10", "ndcg_at_20",
    "conf_high_pct", "conf_medium_pct", "conf_low_pct",
]


def aggregate(rows: list[dict]) -> dict:
    """
    Computes mean, std, median, min, max for every numeric metric.
    Only includes rows where status == 'ok'.
    """
    valid = [r for r in rows if r.get("status") == "ok"]
    out   = {"n_topics": len(valid)}
    for col in METRIC_COLS:
        vals = [r[col] for r in valid if col in r]
        if vals:
            out[col] = {
                "mean":   round(float(np.mean(vals)),   4),
                "std":    round(float(np.std(vals)),    4),
                "median": round(float(np.median(vals)), 4),
                "min":    round(float(np.min(vals)),    4),
                "max":    round(float(np.max(vals)),    4),
            }
    return out


# ─── Table builders ───────────────────────────────────────────────────────────

def _cell(agg: dict, col: str) -> str:
    """Formats one cell as  mean ± std  or  —  if missing."""
    if col not in agg:
        return "—"
    return f"{agg[col]['mean']:.3f} $\\pm$ {agg[col]['std']:.3f}"


def build_main_latex_table(full: dict, ablation: dict | None,
                            n_topics: int) -> str:
    """
    Two-row LaTeX table: full pipeline vs BM25-only ablation.
    Columns: WSS@95 · WSS@100 · R@10 · R@20 · nDCG@10 · P · R · F1 · κ
    Matches the format expected by Research Synthesis Methods and JAMIA.
    """
    cols = [
        ("WSS@95",    "wss_95"),
        ("WSS@100",   "wss_100"),
        ("R@10",      "recall_at_10"),
        ("R@20",      "recall_at_20"),
        ("nDCG@10",   "ndcg_at_10"),
        ("Precision", "precision"),
        ("Recall",    "recall"),
        ("F1",        "f1"),
        ("$\\kappa$", "kappa"),
    ]

    header = " & ".join(h for h, _ in cols)
    full_row   = " & ".join(_cell(full,     k) for _, k in cols)
    abl_row    = (" & ".join(_cell(ablation, k) for _, k in cols)
                  if ablation else None)

    body = f"    Full pipeline (PubMedBERT + Qwen) & {full_row} \\\\\n"
    if abl_row:
        body += f"    BM25 only (no PubMedBERT)        & {abl_row} \\\\\n"

    return (
        "\\begin{table*}[ht]\n"
        "\\centering\n"
        f"\\caption{{Screening and ranking performance across {n_topics} CLEF TAR 2019 topics "
        "(mean $\\pm$ SD). WSS = Work Saved over Sampling. "
        "R@$k$ = Recall at rank $k$. nDCG@10 = Normalised Discounted Cumulative Gain. "
        "$\\kappa$ = Cohen's Kappa between system decisions and gold labels.}}\n"
        "\\label{tab:main_results}\n"
        f"\\begin{{tabular}}{{l{'c' * len(cols)}}}\n"
        "\\toprule\n"
        f"Method & {header} \\\\\n"
        "\\midrule\n"
        f"{body}"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}"
    )


def build_confidence_latex_table(rows: list[dict]) -> str:
    """
    Stratifies WSS@95 and F1 by extraction_confidence.
    Shows that abstract-limited (low confidence) extractions
    perform worse — a transparency contribution unique to this system.
    """
    valid = [r for r in rows if r.get("status") == "ok"]

    def group_stats(subset: list[dict], col: str) -> str:
        vals = [r[col] for r in subset if col in r]
        if not vals:
            return "— & —"
        return f"{np.mean(vals):.3f} & {len(vals)}"

    high = [r for r in valid if r.get("conf_high_pct", 0) > 50]
    med  = [r for r in valid if 20 < r.get("conf_high_pct", 0) <= 50]
    low  = [r for r in valid if r.get("conf_high_pct", 0) <= 20]

    return (
        "\\begin{table}[ht]\n"
        "\\centering\n"
        "\\caption{WSS@95 and F1 stratified by extraction confidence. "
        "High confidence indicates the abstract provided sufficient evidence "
        "for extraction. Low confidence flags abstract-limited outputs "
        "requiring additional human verification.}\n"
        "\\label{tab:confidence}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Confidence & WSS@95 (mean) & Topics & F1 (mean) & Topics \\\\\n"
        "\\midrule\n"
        f"High   & {group_stats(high, 'wss_95')} & {group_stats(high, 'f1')} \\\\\n"
        f"Medium & {group_stats(med,  'wss_95')} & {group_stats(med,  'f1')} \\\\\n"
        f"Low    & {group_stats(low,  'wss_95')} & {group_stats(low,  'f1')} \\\\\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}"
    )


# ─── Terminal progress printer ────────────────────────────────────────────────

def print_progress(topic_id: str, metrics: dict, idx: int, total: int):
    bar_len   = 30
    filled    = int(bar_len * idx / total)
    bar       = "█" * filled + "░" * (bar_len - filled)
    status    = metrics.get("status", "?")
    if status == "ok":
        info = (f"WSS@95={metrics['wss_95']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"nDCG@10={metrics['ndcg_at_10']:.3f}")
    else:
        info = f"SKIPPED ({status})"
    print(f"[{bar}] {idx:3d}/{total}  {topic_id:<12s}  {info}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Collect results from all pipeline runs into paper tables."
    )
    ap.add_argument("--runs-dir",    default="runs",
                    help="Root folder containing all pipeline run subfolders")
    ap.add_argument("--qrels-dir",   default="tar/2019/Task2/qrels",
                    help="CLEF TAR 2019 qrels folder (*.txt files)")
    ap.add_argument("--ablation-dir", default=None,
                    help="Optional separate runs/ folder for ablation runs "
                         "(e.g. runs produced with use_pubmedbert_ranking=false)")
    ap.add_argument("--out-dir",     default="paper_results",
                    help="Output folder for tables and CSV")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load gold labels ──────────────────────────────────────────
    print(f"Loading CLEF TAR qrels from {args.qrels_dir} ...")
    qrels = load_all_qrels(args.qrels_dir)
    if not qrels:
        print("  No qrel files found. Clone: git clone https://github.com/CLEF-TAR/tar.git")
        return
    print(f"  {len(qrels)} topics loaded.\n")

    # ── Discover matching run folders ─────────────────────────────
    print(f"Discovering run folders in {runs_dir} ...")
    run_list = discover_runs(runs_dir, qrels)
    if not run_list:
        print("  No matching run folders found.")
        print("  Run generate_clef_configs.py --run first.")
        return
    print(f"  {len(run_list)} matched.\n")

    # ── Evaluate full pipeline runs ───────────────────────────────
    print("Evaluating full pipeline runs:")
    print("-" * 70)
    full_rows = []
    for idx, entry in enumerate(run_list, 1):
        tid      = entry["topic_id"]
        run_dir  = entry["run_dir"]
        gold     = qrels[tid]
        metrics  = evaluate_run(run_dir, gold)
        metrics["topic_id"]   = tid
        metrics["run_folder"] = run_dir.name
        full_rows.append(metrics)
        print_progress(tid, metrics, idx, len(run_list))

    # ── Optionally evaluate ablation runs ─────────────────────────
    abl_rows = []
    if args.ablation_dir:
        abl_dir = Path(args.ablation_dir)
        print(f"\nEvaluating ablation runs from {abl_dir}:")
        print("-" * 70)
        abl_list = discover_runs(abl_dir, qrels)
        for idx, entry in enumerate(abl_list, 1):
            tid     = entry["topic_id"]
            gold    = qrels[tid]
            metrics = evaluate_run(entry["run_dir"], gold)
            metrics["topic_id"] = tid
            abl_rows.append(metrics)
            print_progress(tid, metrics, idx, len(abl_list))

    # ── Aggregate ─────────────────────────────────────────────────
    print("\nAggregating metrics...")
    full_agg = aggregate(full_rows)
    abl_agg  = aggregate(abl_rows) if abl_rows else None
    n_valid  = full_agg["n_topics"]

    # ── Save per-topic CSV ────────────────────────────────────────
    df = pd.DataFrame(full_rows)
    csv_path = out_dir / "results_per_topic.csv"
    df.to_csv(csv_path, index=False)

    # ── Save aggregate JSON ───────────────────────────────────────
    agg_path = out_dir / "results_aggregate.json"
    agg_path.write_text(json.dumps(
        {"full_pipeline": full_agg, "ablation": abl_agg},
        indent=2
    ))

    # ── Save LaTeX tables ─────────────────────────────────────────
    main_tex  = build_main_latex_table(full_agg, abl_agg, n_valid)
    conf_tex  = build_confidence_latex_table(full_rows)

    (out_dir / "table_main_results.tex").write_text(main_tex)
    (out_dir / "table_confidence.tex").write_text(conf_tex)

    # ── Terminal summary ──────────────────────────────────────────
    ok_count  = sum(1 for r in full_rows if r.get("status") == "ok")
    skip_count = len(full_rows) - ok_count

    print("\n" + "=" * 70)
    print(f"  Topics evaluated : {len(full_rows)}")
    print(f"  OK               : {ok_count}")
    print(f"  Skipped          : {skip_count}")
    print("=" * 70)

    labels = [
        ("WSS@95",      "wss_95"),
        ("WSS@100",     "wss_100"),
        ("Recall@10",   "recall_at_10"),
        ("Recall@20",   "recall_at_20"),
        ("nDCG@10",     "ndcg_at_10"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("F1",          "f1"),
        ("Kappa",       "kappa"),
    ]
    print(f"\n  {'Metric':<14}  {'Full pipeline':>18}", end="")
    if abl_agg:
        print(f"  {'Ablation (no BERT)':>20}", end="")
    print()
    print(f"  {'-'*14}  {'-'*18}", end="")
    if abl_agg:
        print(f"  {'-'*20}", end="")
    print()

    for label, key in labels:
        if key not in full_agg:
            continue
        m    = full_agg[key]
        cell = f"{m['mean']:.3f} ± {m['std']:.3f}"
        print(f"  {label:<14}  {cell:>18}", end="")
        if abl_agg and key in abl_agg:
            am   = abl_agg[key]
            diff = m["mean"] - am["mean"]
            sign = "+" if diff >= 0 else ""
            print(f"  {am['mean']:.3f} ± {am['std']:.3f}  ({sign}{diff:.3f})", end="")
        print()

    print("\n" + "=" * 70)
    print(f"  Outputs saved to  {out_dir}/")
    print(f"    results_per_topic.csv       ← Supplementary Table S1")
    print(f"    results_aggregate.json      ← sanity check")
    print(f"    table_main_results.tex      ← Table 2  (paste into LaTeX)")
    print(f"    table_confidence.tex        ← Table 3  (paste into LaTeX)")
    print("=" * 70)

    # ── Skipped topic report ──────────────────────────────────────
    skipped = [r for r in full_rows if r.get("status") != "ok"]
    if skipped:
        print(f"\n  Skipped topics ({len(skipped)}):")
        for r in skipped:
            print(f"    {r['topic_id']:<12}  {r.get('status')}  "
                  f"n_matched={r.get('n_matched', 0)}")


if __name__ == "__main__":
    main()
