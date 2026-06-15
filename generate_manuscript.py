from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from src.lit_review_agent.config import ReviewConfig
from src.lit_review_agent.local_ollama import OllamaAgent
from src.lit_review_agent.manuscript import manuscript_to_markdown
from src.lit_review_agent.pipeline import ReviewPipeline
from src.lit_review_agent.schemas import RunStats, StudyExtraction


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate only the manuscript from an existing literature-review run."
    )
    parser.add_argument(
        "--run-dir",
        help="Existing run directory containing extractions/*.json.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest folder inside the configured output directory.",
    )
    parser.add_argument(
        "--config",
        default="config.detailed.json",
        help="Config file to use for model settings and relevance threshold.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file containing optional local settings.",
    )
    return parser


def latest_run_dir(config: ReviewConfig) -> Path:
    output_dir = Path(config.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    candidates = [path for path in output_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"No run directories found in {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_included_extractions(run_dir: Path, config: ReviewConfig) -> list[StudyExtraction]:
    extractions_dir = run_dir / "extractions"
    if not extractions_dir.exists():
        raise RuntimeError(f"No extractions directory found: {extractions_dir}")

    included: list[StudyExtraction] = []
    for path in sorted(extractions_dir.glob("*.json")):
        extraction = StudyExtraction.model_validate(load_json(path))
        decision = extraction.screening.decision.lower().strip()
        score = extraction.screening.relevance_score
        if decision == "include" and score >= config.min_relevance_score:
            included.append(extraction)
    return included


def main() -> None:
    args = build_parser().parse_args()
    app_dir = Path(__file__).resolve().parent

    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = app_dir / env_path
    load_dotenv(env_path, encoding="utf-8-sig")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = app_dir / config_path
    config = ReviewConfig.load(config_path)

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.latest:
        run_dir = latest_run_dir(config)
    else:
        raise SystemExit("Use --run-dir D:\\Literature_Search\\runs\\... or --latest")

    if not run_dir.is_absolute():
        run_dir = app_dir / run_dir
    if not run_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")

    included = load_included_extractions(run_dir, config)
    if not included:
        raise RuntimeError(
            f"No included extractions found in {run_dir / 'extractions'} "
            f"with min_relevance_score={config.min_relevance_score}"
        )

    stats_path = run_dir / "run_stats.json"
    stats = RunStats.model_validate(load_json(stats_path)) if stats_path.exists() else RunStats()

    agent = OllamaAgent(config)
    agent.check_ready()

    manuscripts_dir = run_dir / "manuscripts"
    manuscripts_dir.mkdir(parents=True, exist_ok=True)
    output_path = manuscripts_dir / "manuscript_final.md"

    console.print(f"[bold]Generating manuscript only from:[/bold] {run_dir}")
    console.print(f"[bold]Included studies:[/bold] {len(included)}")

    try:
        draft = agent.draft_manuscript(included, stats.model_dump(), final=True)
        markdown = manuscript_to_markdown(draft)
    except Exception as exc:
        console.print(
            "[yellow]Structured manuscript generation failed; saving fallback draft.[/yellow]"
        )
        pipeline = ReviewPipeline(config)
        markdown = pipeline._fallback_manuscript_markdown(included, repr(exc))

    output_path.write_text(markdown, encoding="utf-8")
    console.print(f"[bold green]Saved manuscript:[/bold green] {output_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
