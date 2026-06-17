# Evaluation Guide

This folder contains evaluation scripts for the local literature review agent.

The goal is to evaluate the pipeline with open-access benchmark data rather than relying only on manual inspection.

## Recommended Public Datasets

- **CLEF TAR**: systematic-review topic files and qrels for evaluating search, ranking, and screening.
- **EBM-NLP**: annotated biomedical abstracts for evaluating PICO-style extraction fields.
- **CSMeD**: larger systematic-review screening benchmark collection that can be added later.

## What Is Evaluated

### 1. Search Coverage

Uses `ranked_candidates.json`.

Measures whether the pipeline retrieved known relevant PubMed records from the CLEF TAR qrels.

Reported metrics:

- number of gold relevant papers
- number retrieved
- search recall

### 2. Ranking Performance

Uses `ranked_candidates.json`.

Measures whether PubMedBERT ranked relevant papers near the top.

Reported metrics:

- Recall@10, Recall@20, Recall@50, Recall@100
- nDCG@10, nDCG@20, nDCG@50, nDCG@100
- average precision
- WSS@95

### 3. Screening Performance

Uses `extractions/*.json`.

Measures whether Qwen's `include`, `maybe`, and `exclude` decisions match gold relevance labels.

For high-recall systematic-review screening, `include` and `maybe` are treated as positive.

Reported metrics:

- precision
- recall
- F1
- Cohen's kappa
- screening WSS@95
- false negative titles

### 4. Summary Faithfulness

Uses `extractions/*.json`.

The script compares generated summaries against the source abstract saved in the extraction JSON. A local Ollama model labels each generated sentence as supported, partial, or unsupported.

This is an abstract-level faithfulness check. It does not prove full-paper correctness.

## Step 1: Download CLEF TAR

From the project root:

```powershell
cd D:\Literature_Search
git clone https://github.com/CLEF-TAR/tar.git
```

If the repository is already downloaded, skip this step.

## Step 2: Generate CLEF Configs

Generate configs only:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\generate_clef_configs.py --topics-dir tar\2019-TAR\Task2\Testing\Intervention\topics --out-dir clef_configs
```

If your local CLEF folder has a different structure, point `--topics-dir` to the folder containing the topic `.txt` files.

Quick 3-topic test with immediate pipeline runs:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\generate_clef_configs.py --topics-dir tar\2019-TAR\Task2\Testing\Intervention\topics --out-dir clef_configs --run --limit 3
```

The generated configs include `run_label`, such as `CD009044`, so evaluation scripts can match each run to the right qrels.

## Step 3: Evaluate One CLEF Run

Use a specific run folder:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_clef_tar.py --run-dir "D:\Literature_Search\runs\YOUR_RUN_FOLDER" --qrels "D:\Literature_Search\tar\2019-TAR\Task2\qrels" --topic-id CD009044 --out paper_results\CD009044_eval.json
```

For CLEF 2019 Task 2, qrels are also nested by topic type. For example, Intervention test qrels are here:

```text
tar\2019-TAR\Task2\Testing\Intervention\qrels
```

If `review_config.json` contains `run_label`, `--topic-id` is optional:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_clef_tar.py --run-dir "D:\Literature_Search\runs\YOUR_RUN_FOLDER" --qrels "D:\Literature_Search\tar\2019-TAR\Task2\qrels"
```

## Optional: Backfill Older Runs

Older run folders may not contain `source`, `source_id`, or `abstract` inside `extractions/*.json`. If the matching candidate files are still present, backfill those fields before evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\backfill_extraction_provenance.py --run-dir "D:\Literature_Search\runs\YOUR_RUN_FOLDER"
```

Preview without editing:

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\backfill_extraction_provenance.py --run-dir "D:\Literature_Search\runs\YOUR_RUN_FOLDER" --dry-run
```

## Step 4: Evaluate Summary Faithfulness

```powershell
.\.venv\Scripts\python.exe scripts\evaluation\evaluate_faithfulness.py --extractions "D:\Literature_Search\runs\YOUR_RUN_FOLDER\extractions" --out paper_results\faithfulness_results.json
```

This requires Ollama to be running and `qwen2.5:7b` to be installed.

## Important Notes

- New runs save `source`, `source_id`, and `abstract` in every extraction JSON. Older runs may not have these fields, so CLEF matching and faithfulness evaluation may be incomplete for older outputs.
- CLEF TAR is PubMed-ID based, so use `databases: ["pubmed"]` for cleanest evaluation.
- Search/ranking metrics should be computed from `ranked_candidates.json`, not only from extracted papers.
- Screening metrics should be computed from `extractions/*.json`, because Qwen decisions are only available after extraction.
- Do not treat abstract-level faithfulness as full-paper validation. It only checks whether generated claims are supported by the abstract stored in the extraction JSON.
