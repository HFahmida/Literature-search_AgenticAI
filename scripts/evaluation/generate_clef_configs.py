from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


BASE_CONFIG = {
    "llm_provider": "ollama",
    "extract_model": "qwen2.5:7b",
    "draft_model": "qwen2.5:7b",
    "databases": ["pubmed"],
    "max_results_per_source": 200,
    "max_ranked_candidates": 100,
    "max_papers_to_extract": 50,
    "use_pubmedbert_ranking": True,
    "draft_during_extraction": False,
    "draft_manuscript": False,
    "concurrency": 1,
    "min_relevance_score": 2,
    "output_dir": "runs",
    "extraction_detail_instructions": [
        "Screen the abstract against the systematic review eligibility criteria.",
        "Extract PICO-style information where available: population, intervention or exposure, comparator, outcomes, study design, and main findings.",
        "Flag abstract-limited evidence in missing_information rather than inventing details.",
    ],
    "custom_extraction_questions": [
        "Does this article match the review objective?",
        "What population or sample is studied?",
        "What intervention, exposure, or comparison is evaluated?",
        "What outcomes are measured?",
        "What result details are available in the abstract?",
    ],
}


SECTION_ALIASES = {
    "title": ["title"],
    "objectives": ["objectives", "objective", "review question"],
    "query": ["query", "search query", "search strategy"],
    "pids": ["pids", "pubmed ids", "pmids"],
    "types_of_studies": ["types of studies", "study types", "type of studies"],
    "types_of_participants": ["types of participants", "participants", "population"],
    "types_of_interventions": ["types of interventions", "interventions", "intervention"],
    "types_of_outcomes": ["types of outcome measures", "outcomes", "outcome measures"],
    "exclusion": ["exclusion criteria", "exclusion"],
}


def match_section_header(line: str) -> str | None:
    candidate = re.sub(r":.*$", "", line).strip().lower()
    for canonical, aliases in SECTION_ALIASES.items():
        if candidate in aliases:
            return canonical
    return None


def parse_topic_file(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    sections: dict[str, str] = {"topic_id": path.stem.upper(), "raw": text}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_key and current_lines:
            sections[current_key] = " ".join(" ".join(current_lines).split())

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        matched = match_section_header(stripped)
        if matched:
            flush()
            current_key = matched
            after_colon = re.sub(r"^[^:]+:\s*", "", stripped).strip()
            current_lines = [after_colon] if after_colon else []
        elif current_key:
            current_lines.append(stripped)
    flush()
    return sections


def build_search_terms(title: str, objectives: str, interventions: str, query: str = "", max_terms: int = 6) -> list[str]:
    terms: list[str] = []
    if title:
        terms.append(title)
    for phrase in re.findall(r'"([^"]{4,80})"', objectives):
        terms.append(phrase)
    if interventions:
        words = [word.strip(".,;:()") for word in interventions.split() if len(word.strip(".,;:()")) > 4]
        if words:
            terms.append(" ".join(words[:8]))
    query_lines = [
        line.strip()
        for line in query.splitlines()
        if line.strip()
        and not re.search(r"^\(?\d+(\s+(and|or)\s+\d+)+\)?$", line.strip(), flags=re.IGNORECASE)
        and not line.lower().startswith("limit ")
        and not line.lower().startswith("exp ")
    ]
    for line in query_lines[:3]:
        cleaned = re.sub(r"\.(ti|ab|pt|fs)\b", "", line, flags=re.IGNORECASE)
        cleaned = re.sub(r"[*()/]", " ", cleaned)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) >= 4:
            terms.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.lower().strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(term.strip())
    return unique[:max_terms] or [title or objectives or query or "systematic review"]


def build_config(topic: dict[str, str], no_bert: bool = False) -> dict:
    topic_id = topic["topic_id"]
    title = topic.get("title") or topic_id
    query = topic.get("query", "")
    objectives = topic.get("objectives") or f"Identify studies relevant to this CLEF TAR topic: {title}"
    studies = topic.get("types_of_studies", "")
    participants = topic.get("types_of_participants", "")
    interventions = topic.get("types_of_interventions", "")
    outcomes = topic.get("types_of_outcomes", "")
    exclusion = topic.get("exclusion", "")

    inclusion = [value for value in [studies, participants, interventions, outcomes] if value]
    if not inclusion:
        inclusion = ["Studies matching the review objective."]

    exclusion_criteria = [exclusion] if exclusion else [
        "Studies not matching the review objective.",
        "Records without enough bibliographic information for screening.",
    ]

    config = {
        **BASE_CONFIG,
        "run_label": topic_id,
        "topic": f"{topic_id}: {title}",
        "review_question": f"{objectives}\n\nOriginal CLEF query:\n{query}".strip(),
        "search_terms": build_search_terms(title, objectives, interventions, query=query),
        "inclusion_criteria": inclusion,
        "exclusion_criteria": exclusion_criteria,
        "use_pubmedbert_ranking": not no_bert,
    }
    return config


def run_pipeline(config_path: Path, pipeline: Path) -> bool:
    command = [sys.executable, str(pipeline), "--config", str(config_path)]
    print("Running:", " ".join(command))
    return subprocess.run(command).returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review configs from CLEF TAR topic files.")
    parser.add_argument("--topics-dir", required=True, help="Folder containing CLEF TAR topic .txt files.")
    parser.add_argument("--out-dir", default="clef_configs", help="Output folder for generated configs.")
    parser.add_argument("--pipeline", default="run_review.py", help="Pipeline entry point.")
    parser.add_argument("--run", action="store_true", help="Run the pipeline after each config is generated.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N topics.")
    parser.add_argument("--no-bert", action="store_true", help="Disable PubMedBERT ranking for ablation runs.")
    args = parser.parse_args()

    topics_dir = Path(args.topics_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline = Path(args.pipeline)

    topic_files = sorted(
        path
        for path in topics_dir.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if args.limit:
        topic_files = topic_files[: args.limit]
    if not topic_files:
        raise SystemExit(f"No .txt topic files found in {topics_dir}")

    summary = []
    for topic_file in topic_files:
        topic = parse_topic_file(topic_file)
        config = build_config(topic, no_bert=args.no_bert)
        suffix = "_no_bert" if args.no_bert else ""
        config_path = out_dir / f"{topic['topic_id']}{suffix}.json"
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

        status = "generated"
        if args.run:
            status = "success" if run_pipeline(config_path, pipeline) else "failed"
        print(f"{topic['topic_id']}: {status} -> {config_path}")
        summary.append({"topic_id": topic["topic_id"], "config": str(config_path), "status": status})

    summary_path = out_dir / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
