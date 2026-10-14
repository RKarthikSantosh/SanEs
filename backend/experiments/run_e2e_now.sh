#!/usr/bin/env bash
# SQuAD end-to-end run, started early so Groq judging overlaps the Cohere-bound retrieval runs.
PY=.venv/bin/python
LOGS=experiments/logs
$PY -m experiments.run_e2e generate > "$LOGS/e2e_generate_early.log" 2>&1 || { echo "FAILED e2e generate"; exit 1; }
$PY -m experiments.run_e2e judge --configs bm25,hybrid --key 0 > "$LOGS/e2e_judge_early_0.log" 2>&1 &
$PY -m experiments.run_e2e judge --configs dense,hybrid_rerank --key 1 > "$LOGS/e2e_judge_early_1.log" 2>&1 &
wait
$PY -m experiments.run_e2e report > "$LOGS/e2e_report_early.log" 2>&1
echo E2E_DONE
