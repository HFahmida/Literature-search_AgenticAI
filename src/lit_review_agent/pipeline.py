from __future__ import annotations

import asyncio
import json
from contextlib import suppress

import httpx
from rich.console import Console

from .config import ReviewConfig
from .local_ollama import OllamaAgent
from .manuscript import manuscript_to_markdown
from .ranking import CandidateRanker
from .schemas import PaperCandidate, RunStats, StudyExtraction
from .sources import dedupe_candidates, search_database
from .storage import RunStorage


console = Console()


class ReviewPipeline:
    def __init__(self, config: ReviewConfig):
        self.config = config
        self.storage = RunStorage(config)
        self.stats = RunStats()
        self.agent = OllamaAgent(config)
        self.ranker = CandidateRanker(config)
        self.extractions: list[StudyExtraction] = []
        self._evidence_changed = asyncio.Event()
        self._stop_drafting = asyncio.Event()
        self._lock = asyncio.Lock()

    async def run(self) -> RunStorage:
        self.storage.save_config(self.config)
        self.storage.save_stats(self.stats)
        self.agent.check_ready()
        draft_task = None
        if self.config.draft_during_extraction:
            draft_task = asyncio.create_task(self._drafting_worker())
        try:
            candidates = await self._search_all()
            await self._extract_all(candidates)
        finally:
            self._stop_drafting.set()
            self._evidence_changed.set()
            if draft_task is not None:
                with suppress(asyncio.CancelledError):
                    await draft_task
        if self.config.draft_manuscript:
            await self._write_draft(final=True)
        self.storage.save_stats(self.stats)
        return self.storage

    async def _search_all(self) -> list[PaperCandidate]:
        console.print("[bold]Searching literature databases...[/bold]")
        all_candidates: list[PaperCandidate] = []
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            for database in self.config.databases:
                database_total = 0
                for term in self.config.search_terms:
                    try:
                        found = await search_database(
                            client,
                            database,
                            term,
                            self.config.max_results_per_source,
                        )
                    except Exception as exc:
                        self.storage.save_error(f"{database}:{term}", repr(exc))
                        console.print(f"[yellow]Search failed for {database}: {term}[/yellow]")
                        continue
                    database_total += len(found)
                    for candidate in found:
                        self.storage.save_candidate(candidate)
                    all_candidates.extend(found)
                    console.print(f"  {database}: {len(found)} records for {term!r}")
                self.stats.records_identified[database] = database_total
                self.storage.save_stats(self.stats)
        unique, duplicates = dedupe_candidates(all_candidates)
        self.stats.duplicate_records_removed = duplicates
        self.stats.unique_records_screened = len(unique)
        self.storage.save_stats(self.stats)
        console.print(
            f"[green]Found {len(all_candidates)} records; {len(unique)} unique after dedupe.[/green]"
        )
        ranked = self.ranker.rank(unique)
        ranked_path = self.storage.run_dir / "ranked_candidates.json"
        ranked_path.write_text(
            json.dumps(
                [
                    {"score": item.score, "candidate": item.candidate.model_dump()}
                    for item in ranked
                ],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        limit = min(self.config.max_papers_to_extract, self.config.max_ranked_candidates)
        selected = [item.candidate for item in ranked[:limit]]
        console.print(
            f"[green]Selected top {len(selected)} candidates for local extraction after PubMedBERT ranking.[/green]"
        )
        return selected

    async def _extract_all(self, candidates: list[PaperCandidate]) -> None:
        console.print("[bold]Screening and extracting papers with local Ollama model...[/bold]")
        semaphore = asyncio.Semaphore(max(1, self.config.concurrency))

        async def run_one(candidate: PaperCandidate) -> None:
            async with semaphore:
                await self._extract_one(candidate)

        await asyncio.gather(*(run_one(candidate) for candidate in candidates))

    async def _extract_one(self, candidate: PaperCandidate) -> None:
        try:
            extraction = await asyncio.to_thread(self.agent.extract_paper, candidate)
        except Exception as exc:
            self.stats.extraction_errors += 1
            self.storage.save_error(candidate.stable_id, repr(exc))
            self.storage.save_stats(self.stats)
            console.print(f"[red]Extraction failed:[/red] {candidate.title[:90]}")
            return

        self.stats.reports_retrieved_or_abstract_only += 1
        decision = extraction.screening.decision.lower().strip()
        score = extraction.screening.relevance_score
        if decision == "include" and score >= self.config.min_relevance_score:
            self.stats.studies_included += 1
        else:
            self.stats.reports_excluded += 1
        self.storage.save_extraction(extraction)
        self.storage.save_stats(self.stats)
        async with self._lock:
            self.extractions.append(extraction)
            self._evidence_changed.set()
        console.print(f"  decision={decision} score={score} {candidate.title[:90]}")

    async def _drafting_worker(self) -> None:
        draft_index = 1
        last_drafted_count = 0
        while not self._stop_drafting.is_set():
            try:
                await asyncio.wait_for(
                    self._evidence_changed.wait(),
                    timeout=max(30, self.config.draft_interval_seconds),
                )
            except asyncio.TimeoutError:
                pass
            if self._stop_drafting.is_set():
                break
            await asyncio.sleep(min(10, max(1, self.config.draft_interval_seconds // 6)))
            current_count = await self._included_count()
            if current_count <= last_drafted_count:
                self._evidence_changed.clear()
                continue
            wrote = await self._write_draft(index=draft_index)
            if wrote:
                last_drafted_count = current_count
                draft_index += 1
            self._evidence_changed.clear()

    async def _included_count(self) -> int:
        async with self._lock:
            return sum(
                1
                for item in self.extractions
                if item.screening.decision.lower().strip() == "include"
                and item.screening.relevance_score >= self.config.min_relevance_score
            )

    async def _write_draft(self, index: int = 1, final: bool = False) -> bool:
        async with self._lock:
            included = [
                item
                for item in self.extractions
                if item.screening.decision.lower().strip() == "include"
                and item.screening.relevance_score >= self.config.min_relevance_score
            ]
        if not included:
            return False
        console.print("[bold]Drafting manuscript from current extracted evidence...[/bold]")
        try:
            draft = await asyncio.to_thread(
                self.agent.draft_manuscript,
                included,
                self.stats.model_dump(),
                final,
            )
        except Exception as exc:
            self.stats.extraction_errors += 1
            self.storage.save_error("manuscript", repr(exc))
            self.storage.save_stats(self.stats)
            fallback = self._fallback_manuscript_markdown(included, repr(exc))
            path = self.storage.save_manuscript(fallback, index, final=final)
            console.print(f"[yellow]Manuscript drafting failed; saved evidence-based fallback draft:[/yellow] {path}")
            return True
        path = self.storage.save_manuscript(manuscript_to_markdown(draft), index, final=final)
        console.print(f"[green]Saved manuscript draft:[/green] {path}")
        return True

    def _fallback_manuscript_markdown(self, included: list[StudyExtraction], error: str) -> str:
        rows = [
            "| Study | Design / sample | Outcomes | Main findings |",
            "|---|---|---|---|",
        ]
        for item in included:
            study = f"{item.title} ({item.year or 'n.d.'})"
            design = " ".join(part for part in [item.study_design or "", item.population_or_sample or ""] if part)
            outcomes = "; ".join(item.outcomes_measured[:5])
            findings = item.results_summary or "; ".join(item.main_results[:3]) or item.overall_concept_summary
            rows.append(
                "| "
                + " | ".join(
                    text.replace("|", "/").replace("\n", " ")
                    for text in [study, design, outcomes, findings]
                )
                + " |"
            )
        summaries = []
        for item in included:
            summaries.append(
                f"### {item.title}\n\n"
                f"**Relevance:** {item.relevance_to_review or item.screening.reason}\n\n"
                f"**Methods:** {item.methods_summary or 'Not available from extracted metadata.'}\n\n"
                f"**Results:** {item.results_summary or '; '.join(item.main_results) or 'Not available from extracted metadata.'}\n\n"
                f"**Key takeaways:**\n"
                + "\n".join(f"- {takeaway}" for takeaway in item.key_takeaways)
            )
        return (
            "# Systematic Review Evidence Draft\n\n"
            "The local LLM manuscript writer failed to produce valid structured manuscript JSON, "
            "so this fallback draft was generated directly from the validated extraction files.\n\n"
            f"Drafting error recorded in logs: `{error}`\n\n"
            "## Evidence Table\n\n"
            + "\n".join(rows)
            + "\n\n## Study-Level Summaries\n\n"
            + "\n\n".join(summaries)
            + "\n\n## Human Reviewer Notes\n\n"
            "- Verify every extracted result against the source paper.\n"
            "- Use this fallback as a synthesis scaffold, not a final manuscript.\n"
        )


async def run_review(config: ReviewConfig) -> RunStorage:
    pipeline = ReviewPipeline(config)
    return await pipeline.run()
