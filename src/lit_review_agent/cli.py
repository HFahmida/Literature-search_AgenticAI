from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

from .config import ReviewConfig
from .pipeline import run_review


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an agentic literature search, extraction, and manuscript draft workflow."
    )
    parser.add_argument(
        "--config",
        default="config.example.json",
        help="Path to the review JSON config.",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Path to .env file containing optional local settings.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    app_dir = Path(__file__).resolve().parents[2]
    env_path = Path(args.env)
    if not env_path.is_absolute():
        env_path = app_dir / env_path
    load_dotenv(env_path, encoding="utf-8-sig")
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = app_dir / config_path
    config = ReviewConfig.load(config_path)
    console.print(f"[bold]Topic:[/bold] {config.topic}")
    storage = asyncio.run(run_review(config))
    console.print(f"[bold green]Done.[/bold green] Outputs are in: {storage.run_dir}")
