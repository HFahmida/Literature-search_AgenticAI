"""
Reads all 123 CLEF TAR 2019 topic files and generates one pipeline
config JSON per topic, ready to pass to run_review.py.

Rationale: CLEF TAR topics contain exactly the information your
pipeline needs — a review question (Objectives), inclusion criteria
(Types of studies, Types of participants), and exclusion criteria
(implied from inclusion). We map these fields directly to your
config schema so results are comparable against CLEF TAR gold qrels.

Usage:
    # Generate configs only
    python generate_clef_configs.py --topics-dir tar/2019/Task2/topics

    # Generate AND run the pipeline on all topics (slow on CPU)
    python generate_clef_configs.py --topics-dir tar/2019/Task2/topics --run

    # Run only the first N topics (for quick testing)
    python generate_clef_configs.py --topics-dir tar/2019/Task2/topics --run --limit 5

Outputs:
    clef_configs/CD000001.json   ← one config per topic
    clef_configs/CD000001_run/   ← pipeline output per topic (if --run)
    clef_configs/batch_summary.json ← topic ID → run folder mapping
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


# ── Config template matching your pipeline's schema ───────────────────────────
# These defaults are safe for CPU-only machines running CLEF TAR topics.
# draft_manuscript is False because we only need extractions for evaluation.

BASE_CONFIG = {
    "llm_provider":           "ollama",
    "extract_model":          "qwen2.5:7b",
    "draft_model":            "qwen2.5:7b",
    "databases":              ["pubmed"],     # CLEF TAR gold labels are PubMed IDs
    "max_results_per_source": 200,            # CLEF TAR topics average ~300 candidates
    "max_ranked_candidates":  100,
    "max_papers_to_extract":  50,
    "use_pubmedbert_ranking": True,
    "draft_during_extraction": False,
    "draft_manuscript":       False,          # skip drafting — saves time
    "concurrency":            1,
    "min_relevance_score":    2,
}


# ── CLEF TAR 2019 topic file parser ──────────────────────────────────────────
# CLEF TAR 2019 topic files are plain text with labeled sections.
# Each file is named after a Cochrane review ID (e.g. CD009044.txt).
# Sections are separated by blank lines with a capitalized header.
#
# Example structure:
#   CD009044
#   Title: Exercise for depression
#   Objectives: To determine the effectiveness of exercise...
#   Types of studies: Randomised controlled trials...
#   Types of participants: Adults (18+)...
#   Types of interventions: Any exercise programme...
#   Types of outcome measures: ...

SECTION_ALIASES = {
    "title":               ["title"],
    "objectives":          ["objectives", "objective", "review question"],
    "types_of_studies":    ["types of studies", "study types", "type of studies"],
    "types_of_participants": ["types of participants", "participants", "population"],
    "types_of_interventions": ["types of interventions", "interventions", "intervention"],
    "types_of_outcomes":   ["types of outcome measures", "outcomes", "outcome measures"],
    "search_methods":      ["search methods", "search strategy", "search terms"],
    "exclusion":           ["exclusion criteria", "exclusion"],
}


def parse_topic_file(path: Path) -> dict:
    """
    Parses one CLEF TAR 2019 topic file into a structured dict.
    Handles both 'Key: Value' single-line and multi-line section formats.
    """
    text = path.read_text(errors="ignore")
    topic_id = path.stem                           # e.g. "CD009044"
    sections = {"topic_id": topic_id, "raw": text}

    # Split into lines and group into sections
    lines = text.split("\n")
    current_key, current_buf = None, []

    def flush():
        if current_key and current_buf:
            content = " ".join(" ".join(current_buf).split())  # normalise whitespace
            sections[current_key] = content

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # Detect a new section header (ends with ':' or matches known alias)
        matched_key = _match_section_header(stripped)
        if matched_key:
            flush()
            current_key = matched_key
            # Content may appear on the same line after the colon
            after_colon = re.sub(r"^[^:]+:\s*", "", stripped)
            current_buf = [after_colon] if after_colon else []
        elif current_key:
            current_buf.append(stripped)

    flush()                                        # save the last section
    return sections


def _match_section_header(line: str) -> str | None:
    """Returns the canonical section key if the line is a known header."""
    candidate = re.sub(r":.*$", "", line).strip().lower()
    for canonical, aliases in SECTION_ALIASES.items():
        if any(candidate == a for a in aliases):
            return canonical
    return None


# ── Config builder ────────────────────────────────────────────────────────────

def build_config(sections: dict) -> dict:
    """
    Maps parsed CLEF TAR sections to your pipeline config schema.

    Mapping rationale:
      objectives        → review_question  (central question for ranking)
      title             → topic            (short label for the run folder)
      types_of_studies  → first inclusion criterion (study design filter)
      types_of_participants → second inclusion criterion
      types_of_interventions → used to derive search terms
      types_of_outcomes → third inclusion criterion
    """
    topic_id = sections.get("topic_id", "UNKNOWN")
    title    = sections.get("title",    topic_id)
    obj      = sections.get("objectives", "")
    studies  = sections.get("types_of_studies", "")
    people   = sections.get("types_of_participants", "")
    interv   = sections.get("types_of_interventions", "")
    outcomes = sections.get("types_of_outcomes", "")

    # Build inclusion criteria list from PICO sections
    inclusion = [c for c in [studies, people, outcomes] if c]
    if not inclusion:
        inclusion = ["Relevant studies matching the review objectives"]

    # Exclusion criteria: CLEF TAR topics rarely state these explicitly.
    # We infer generic ones that match typical SLR exclusion practice.
    exclusion = [
        "Non-English language publications",
        "Conference abstracts without full data",
        "Studies not matching the review population or intervention",
    ]

    # Search terms: extract key noun phrases from title + objectives + intervention
    search_terms = _extract_search_terms(title, obj, interv)

    config = {
        **BASE_CONFIG,
        "topic":              f"{topic_id}: {_truncate(title, 60)}",
        "review_question":    obj or f"Review topic: {title}",
        "search_terms":       search_terms,
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion,
        "run_label":          topic_id,         # used by batch runner
    }
    return config


def _extract_search_terms(title: str, objectives: str, interventions: str,
                           max_terms: int = 6) -> list[str]:
    """
    Derives search terms from the topic text.
    Strategy: take the title as the primary term, then extract noun
    chunks from objectives and interventions as secondary terms.
    This is a simple heuristic — good enough for automated batch runs.
    """
    terms = []

    # Title is always the best primary search string
    if title and title.lower() not in ("unknown", ""):
        terms.append(title.strip())

    # Extract bracketed or quoted phrases from objectives
    for match in re.finditer(r'"([^"]{4,60})"', objectives):
        terms.append(match.group(1))

    # Extract key noun phrases from interventions (words > 4 chars)
    if interventions:
        words = [w for w in interventions.split() if len(w) > 4]
        if words:
            terms.append(" ".join(words[:6]))

    # Remove duplicates while preserving order
    seen, unique = set(), []
    for t in terms:
        norm = t.lower()
        if norm not in seen:
            seen.add(norm)
            unique.append(t)

    return unique[:max_terms] if unique else [title or "systematic review"]


def _truncate(text: str, n: int) -> str:
    return text[:n].rstrip() + ("..." if len(text) > n else "")


# ── Batch runner ──────────────────────────────────────────────────────────────

def run_pipeline(config_path: Path, pipeline_script: str = "run_review.py") -> bool:
    """
    Calls your pipeline with the generated config file.
    Returns True if the run completed without error.
    """
    cmd = [sys.executable, pipeline_script, "--config", str(config_path)]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate pipeline configs from CLEF TAR 2019 topics."
    )
    ap.add_argument("--topics-dir",  default="tar/2019/Task2/topics",
                    help="Path to CLEF TAR topics folder")
    ap.add_argument("--out-dir",     default="clef_configs",
                    help="Where to save generated config files")
    ap.add_argument("--pipeline",    default="run_review.py",
                    help="Path to your pipeline entry point")
    ap.add_argument("--run",         action="store_true",
                    help="Run the pipeline immediately after generating configs")
    ap.add_argument("--limit",       type=int, default=None,
                    help="Only process the first N topics (for quick testing)")
    ap.add_argument("--ablation-no-bert", action="store_true",
                    help="Generate a second set of configs with PubMedBERT disabled "
                         "(for ablation study)")
    args = ap.parse_args()

    topics_dir = Path(args.topics_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    topic_files = sorted(topics_dir.glob("*.txt"))
    if not topic_files:
        print(f"No .txt topic files found in {topics_dir}")
        print("Check that you cloned: git clone https://github.com/CLEF-TAR/tar.git")
        sys.exit(1)

    if args.limit:
        topic_files = topic_files[: args.limit]

    print(f"Found {len(topic_files)} topic files. Processing...\n")

    batch_summary = {}   # topic_id → config path + run status

    for topic_file in topic_files:
        topic_id = topic_file.stem
        print(f"[{topic_id}] Parsing topic file...")

        sections = parse_topic_file(topic_file)
        config   = build_config(sections)

        # Save the main config
        config_path = out_dir / f"{topic_id}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"  Saved config: {config_path}")

        # Optionally save an ablation config (PubMedBERT disabled)
        if args.ablation_no_bert:
            ablation_config = {**config, "use_pubmedbert_ranking": False}
            abl_path = out_dir / f"{topic_id}_no_bert.json"
            with open(abl_path, "w") as f:
                json.dump(ablation_config, f, indent=2)
            print(f"  Saved ablation config: {abl_path}")

        batch_summary[topic_id] = {
            "config":  str(config_path),
            "topic":   config["topic"],
            "status":  "generated",
        }

        # Run the pipeline if requested
        if args.run:
            print(f"  Running pipeline for {topic_id}...")
            success = run_pipeline(config_path, pipeline_script=args.pipeline)
            batch_summary[topic_id]["status"] = "success" if success else "failed"
            print(f"  Status: {batch_summary[topic_id]['status']}")

        print()

    # Save batch summary so the evaluation scripts can find all run folders
    summary_path = out_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    # Print overview
    total     = len(batch_summary)
    succeeded = sum(1 for v in batch_summary.values() if v["status"] == "success")
    failed    = sum(1 for v in batch_summary.values() if v["status"] == "failed")
    generated = sum(1 for v in batch_summary.values() if v["status"] == "generated")

    print("=" * 50)
    print(f"Topics processed : {total}")
    print(f"Configs generated: {generated + succeeded + failed}")
    if args.run:
        print(f"Pipeline runs OK : {succeeded}")
        print(f"Pipeline failed  : {failed}")
    print(f"Summary saved to : {summary_path}")
    print()
    if not args.run:
        print("To run the pipeline on all configs:")
        print(f"  python generate_clef_configs.py "
              f"--topics-dir {args.topics_dir} --run")
        print("For a quick 5-topic test:")
        print(f"  python generate_clef_configs.py "
              f"--topics-dir {args.topics_dir} --run --limit 5")


if __name__ == "__main__":
    main()
