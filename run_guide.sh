#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Full evaluation pipeline for the Local Literature Review Agent
# Run this from the root of your project folder.
# Assumes your virtual environment is already active.
#
# Structure expected:
#   your_project/
#     run_review.py           ← your pipeline
#     runs/                   ← pipeline outputs
#     evaluate_screening.py   ← from this conversation
#     evaluate_faithfulness.py
#     prepare_eval_data.py
# ─────────────────────────────────────────────────────────────────

set -e  # stop on any error

echo "=== Step 1: Install evaluation dependencies ==="
pip install \
    datasets \
    evaluate \
    rouge_score \
    bert_score \
    scikit-learn \
    numpy

# bert_score needs a transformer model on first use — it auto-downloads.
# rouge_score is the backend for the evaluate library's rouge metric.


echo ""
echo "=== Step 2: Download public gold datasets ==="

# CLEF TAR 2019 (systematic review screening gold standard)
if [ ! -d "tar" ]; then
    echo "Cloning CLEF TAR repository..."
    git clone https://github.com/CLEF-TAR/tar.git
else
    echo "CLEF TAR already downloaded."
fi

# EBM-NLP (PICO extraction annotations)
if [ ! -d "EBM-NLP" ]; then
    echo "Cloning EBM-NLP repository..."
    git clone https://github.com/bepnye/EBM-NLP.git
else
    echo "EBM-NLP already downloaded."
fi

# CSMeD and LitSearch are downloaded automatically via HuggingFace
# when prepare_eval_data.py runs. No manual download needed.


echo ""
echo "=== Step 3: Prepare gold data files ==="
# Converts all datasets to the JSON format expected by the eval scripts.
# Outputs go to gold/ folder.
python prepare_eval_data.py


echo ""
echo "=== Step 4: Run your pipeline on CLEF TAR topics ==="
# Rationale: We use the CLEF TAR 2019 review questions as inputs
# to your pipeline so its outputs can be compared to the gold qrels.
#
# For each topic in tar/2019/Task2/topics/, create a config JSON
# using the review question and criteria, then run the pipeline.
# The script below does this automatically for the first 5 topics
# (for a quick test; remove the head -5 to run all 123).

TOPICS_DIR="tar/2019/Task2/topics"
RESULTS_DIR="eval_runs"
mkdir -p "$RESULTS_DIR"

# Generate one config file per topic and run the pipeline
# (Shown here for manual execution — see generate_clef_configs.py
#  for the automated version)
echo "Topics available:"
ls "$TOPICS_DIR" | head -5
echo "(Run generate_clef_configs.py to auto-create pipeline configs)"


echo ""
echo "=== Step 5: Run screening evaluation ==="
# Point to the extractions folder from one of your runs.
# Replace YOUR_RUN_FOLDER with the actual timestamped folder name.

RUN_FOLDER=$(ls -td runs/*/ | head -1)  # uses most recent run
echo "Evaluating run: $RUN_FOLDER"

python evaluate_screening.py \
    --extractions "${RUN_FOLDER}extractions/" \
    --gold        "gold/clef_tar_2019_labels.json"

# Results printed to stdout. Redirect to save:
# python evaluate_screening.py ... > results/screening_metrics.txt


echo ""
echo "=== Step 6: Run faithfulness evaluation ==="
# Evaluates ROUGE-L, BERTScore, and LLM-as-judge faithfulness
# on all detailed_summary fields in the extraction JSONs.

python evaluate_faithfulness.py "${RUN_FOLDER}extractions/"

# Saves detailed per-paper results to faithfulness_results.json


echo ""
echo "=== Step 7: Run ablation ==="
# Four conditions to compare — run each separately and collect WSS@95:
#   Condition A: BM25 only (set use_pubmedbert_ranking: false in config)
#   Condition B: PubMedBERT only, no Qwen screening (manual top-N selection)
#   Condition C: Full pipeline (your default)
#   Condition D: Full pipeline + relevance_score threshold tuning

echo "Ablation: set use_pubmedbert_ranking to false in config and re-run"
echo "then compare WSS@95 values across conditions."


echo ""
echo "=== Done ==="
echo "Results:"
echo "  screening_metrics.txt    ← WSS@95, Recall@k, nDCG@k, Kappa, F1"
echo "  faithfulness_results.json ← ROUGE-L, BERTScore, faithfulness rate"
echo "  gold/                    ← all gold label files"
