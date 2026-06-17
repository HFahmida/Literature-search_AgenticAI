from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def stable_id(candidate: dict[str, Any]) -> str:
    doi = str(candidate.get("doi") or "").strip().lower()
    if doi:
        return "doi:" + doi
    title = " ".join(str(candidate.get("title") or "").lower().split())
    return "title:" + title


def title_key(title: Any) -> str:
    text = str(title or "").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def load_candidates(run_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    by_title: dict[str, dict[str, Any]] = {}
    for path in (run_dir / "candidates").glob("*.json"):
        candidate = load_json(path)
        by_id[stable_id(candidate)] = candidate
        by_title[title_key(candidate.get("title"))] = candidate
    return by_id, by_title


def backfill(run_dir: Path, dry_run: bool = False) -> dict[str, int]:
    by_id, by_title = load_candidates(run_dir)
    counts = {"seen": 0, "matched": 0, "updated": 0, "unmatched": 0}
    for path in sorted((run_dir / "extractions").glob("*.json")):
        extraction = load_json(path)
        counts["seen"] += 1
        paper_id = str(extraction.get("paper_id") or "")
        candidate = by_id.get(paper_id) or by_title.get(title_key(extraction.get("title")))
        if not candidate:
            counts["unmatched"] += 1
            continue
        counts["matched"] += 1
        changed = False
        for field in ["source", "source_id", "abstract"]:
            if not extraction.get(field) and candidate.get(field):
                extraction[field] = candidate[field]
                changed = True
        if changed:
            counts["updated"] += 1
            if not dry_run:
                write_json(path, extraction)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill source, source_id, and abstract into old extraction JSON files."
    )
    parser.add_argument("--run-dir", required=True, help="Existing pipeline run folder.")
    parser.add_argument("--dry-run", action="store_true", help="Report counts without editing files.")
    args = parser.parse_args()
    counts = backfill(Path(args.run_dir), dry_run=args.dry_run)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
