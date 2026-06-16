# Install the two remaining dependencies if not already installed
pip install pandas scikit-learn numpy

# Basic run — full pipeline only
python collect_results.py

# With ablation comparison column in the table
python collect_results.py \
    --ablation-dir runs_no_bert/

# All paths explicit (if your folders differ)
python collect_results.py \
    --runs-dir    runs/ \
    --qrels-dir   tar/2019/Task2/qrels \
    --ablation-dir runs_no_bert/ \
    --out-dir     paper_results/