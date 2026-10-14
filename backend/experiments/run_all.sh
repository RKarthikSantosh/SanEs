#!/usr/bin/env bash
# Runs every experiment in order. Usage (from backend/): bash experiments/run_all.sh
set -u
PY=.venv/bin/python
LOGS=experiments/logs
mkdir -p "$LOGS"
for spec in "scifact" "nfcorpus" "fiqa --max-queries 200"; do
  name=${spec%% *}
  $PY -m experiments.run_retrieval $spec > "$LOGS/retrieval_$name.log" 2>&1 || echo "FAILED retrieval $name"
done
$PY -m experiments.run_e2e generate > "$LOGS/e2e_generate.log" 2>&1 || echo "FAILED e2e generate"
# judge with both Groq keys in parallel
$PY -m experiments.run_e2e judge --configs bm25,hybrid --key 0 > "$LOGS/e2e_judge_0.log" 2>&1 &
$PY -m experiments.run_e2e judge --configs dense,hybrid_rerank --key 1 > "$LOGS/e2e_judge_1.log" 2>&1 &
wait
$PY -m experiments.run_e2e report > "$LOGS/e2e_report.log" 2>&1 || echo "FAILED e2e report"
echo ALL_DONE
