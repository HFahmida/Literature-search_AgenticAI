from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ReviewConfig
from .schemas import PaperCandidate, RunStats, StudyExtraction


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:max_len] or "review"


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


class RunStorage:
    def __init__(self, config: ReviewConfig):
        root = Path(config.output_dir)
        if not root.is_absolute():
            root = Path.cwd() / root
        self.run_dir = root / f"{now_stamp()}-{slugify(config.topic)}"
        self.candidates_dir = self.run_dir / "candidates"
        self.extractions_dir = self.run_dir / "extractions"
        self.paper_summaries_dir = self.run_dir / "paper_summaries"
        self.manuscripts_dir = self.run_dir / "manuscripts"
        self.logs_dir = self.run_dir / "logs"
        for path in [
            self.candidates_dir,
            self.extractions_dir,
            self.paper_summaries_dir,
            self.manuscripts_dir,
            self.logs_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def save_config(self, config: ReviewConfig) -> None:
        write_json(self.run_dir / "review_config.json", config.model_dump())

    def save_stats(self, stats: RunStats) -> None:
        write_json(self.run_dir / "run_stats.json", stats.model_dump())

    def save_candidate(self, candidate: PaperCandidate) -> None:
        name = slugify(candidate.stable_id, 120) + ".json"
        write_json(self.candidates_dir / name, candidate.model_dump())
        append_jsonl(self.logs_dir / "candidates.jsonl", candidate.model_dump())

    def save_extraction(self, extraction: StudyExtraction) -> None:
        name = slugify(extraction.paper_id, 120) + ".json"
        write_json(self.extractions_dir / name, extraction.model_dump())
        summary_name = slugify(extraction.paper_id, 120) + ".md"
        (self.paper_summaries_dir / summary_name).write_text(
            extraction_to_markdown(extraction),
            encoding="utf-8",
        )
        append_jsonl(self.logs_dir / "extractions.jsonl", extraction.model_dump())

    def save_error(self, paper_id: str, error: str) -> None:
        append_jsonl(self.logs_dir / "errors.jsonl", {"paper_id": paper_id, "error": error})

    def save_manuscript(self, markdown: str, index: int, final: bool = False) -> Path:
        filename = "manuscript_final.md" if final else f"manuscript_draft_{index:03d}.md"
        path = self.manuscripts_dir / filename
        path.write_text(markdown, encoding="utf-8")
        return path


def extraction_to_markdown(extraction: StudyExtraction) -> str:
    takeaways = "\n".join(f"- {item}" for item in extraction.key_takeaways)
    methods = "\n".join(f"- {item}" for item in extraction.methods)
    outcomes = "\n".join(f"- {item}" for item in extraction.outcomes_measured)
    results = "\n".join(f"- {item}" for item in extraction.main_results)
    limitations = "\n".join(f"- {item}" for item in extraction.limitations)
    missing = "\n".join(f"- {item}" for item in extraction.missing_information)
    return f"""# {extraction.title}

## Screening

- Decision: {extraction.screening.decision}
- Relevance score: {extraction.screening.relevance_score}
- Reason: {extraction.screening.reason}

## Detailed Summary

{extraction.detailed_summary or extraction.overall_concept_summary}

## Relevance To Review

{extraction.relevance_to_review or ""}

## Key Takeaways

{takeaways}

## Methods Summary

{extraction.methods_summary or ""}

{methods}

## Outcomes

{outcomes}

## Results Summary

{extraction.results_summary or ""}

{results}

## Conclusion Summary

{extraction.conclusion_summary or ""}

## Limitations And Missing Information

{limitations}

{missing}

## Citation

- Authors: {", ".join(extraction.authors)}
- Year: {extraction.year or ""}
- Journal: {extraction.journal or ""}
- DOI: {extraction.doi or ""}
- URL: {extraction.url or ""}
"""
