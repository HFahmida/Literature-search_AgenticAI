from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def main() -> int:
    failures = 0

    try:
        import numpy  # noqa: F401
        import pydantic  # noqa: F401
        import rich  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401

        ok("Python dependencies are importable")
    except Exception as exc:
        fail(f"Python dependency import failed: {exc}")
        failures += 1

    try:
        from lit_review_agent.config import ReviewConfig

        config = ReviewConfig.load(ROOT / "config.example.json")
        ok("config.example.json parsed successfully")
    except Exception as exc:
        fail(f"Config parsing failed: {exc}")
        return 1

    try:
        response = httpx.get(f"{config.ollama_base_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = {item.get("name") for item in models}
        ok("Ollama server is reachable")
        if config.extract_model in model_names:
            ok(f"Ollama model is installed: {config.extract_model}")
        else:
            fail(f"Ollama model is missing: {config.extract_model}. Run: ollama pull {config.extract_model}")
            failures += 1
    except Exception as exc:
        fail(f"Ollama check failed: {exc}")
        failures += 1

    try:
        from transformers import AutoTokenizer

        AutoTokenizer.from_pretrained(config.pubmedbert_model)
        ok(f"PubMedBERT tokenizer is available: {config.pubmedbert_model}")
    except Exception as exc:
        warn(f"PubMedBERT tokenizer check failed. It may download on first run. Details: {exc}")

    try:
        quick = json.loads((ROOT / "config.quick.json").read_text(encoding="utf-8"))
        if quick.get("max_papers_to_extract") == 1:
            ok("config.quick.json is configured as a one-paper smoke test")
        else:
            warn("config.quick.json is not limited to one paper")
    except Exception as exc:
        warn(f"Could not inspect config.quick.json: {exc}")

    if failures:
        print(f"\nSetup check finished with {failures} failure(s).")
        return 1
    print("\nSetup check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
