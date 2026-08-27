#!/usr/bin/env bash
# Article 1 Experiment 2 full matrix: 3 datasets × 3 seeds × 6 regimes.
# Requires the already-built Experiment-1 holdout_authority matrix.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-FLWR}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-OUTPUTS/experiments/article1_experiment2_v1}"
PROXY_SIZE="${PROXY_SIZE:-1000}"
AUTH_ROOT="${AUTH_ROOT:-OUTPUTS/experiments/article_1_server_expertise/experiment_1/holdout_authority}"

# Build every target first.  Inspect these cheap artifacts before enabling
# student training; set RUN_STUDENTS=1 only after that review.
find "$AUTH_ROOT" -name provenance.json -print0 | sort -z | while IFS= read -r -d '' PROVENANCE; do
  readarray -t CELL < <(conda run -n "$CONDA_ENV" python -c '
import json, sys
d=json.load(open(sys.argv[1]))
dataset={"cifar10":"cifar", "mnist":"mnist", "fmnist":"fmnist"}[d["dataset"]]
print(dataset); print(d["seed"]); print(d["regime"]); print(d["source_proxy_analysis"])
' "$PROVENANCE")
  DATASET="${CELL[0]}"; SEED="${CELL[1]}"; REGIME="${CELL[2]}"; SOURCE="${CELL[3]}"
  AUTHORITY="$(dirname "$PROVENANCE")/authority.npz"
  conda run -n "$CONDA_ENV" python -m experiments.article_1_server_expertise.experiment_2 \
    --stage targets --source-proxy-analysis "$SOURCE" --authority-npz "$AUTHORITY" \
    --output-root "$OUT" --dataset "$DATASET" --seed "$SEED" --regime "$REGIME" \
    --proxy-size "$PROXY_SIZE"
  if [[ "${RUN_STUDENTS:-0}" == "1" ]]; then
    conda run -n "$CONDA_ENV" python -m experiments.article_1_server_expertise.experiment_2 \
      --stage students --source-proxy-analysis "$SOURCE" --authority-npz "$AUTHORITY" \
      --output-root "$OUT" --dataset "$DATASET" --seed "$SEED" --regime "$REGIME" \
      --proxy-size "$PROXY_SIZE" --student-init-seed "$SEED" --batch-order-seed "$SEED" \
      --training-mode fixed_epochs --device "$DEVICE"
    EXISTING_RESULTS=("$OUT/experiment_2/$DATASET/seed_$SEED/$REGIME"/N"$PROXY_SIZE"_balanced_*/results.csv)
    if [[ -e "${EXISTING_RESULTS[0]}" ]]; then
      echo "[SKIP] immutable aggregate already exists: ${EXISTING_RESULTS[0]}"
    else
      conda run -n "$CONDA_ENV" python -m experiments.article_1_server_expertise.experiment_2 \
        --stage aggregate --source-proxy-analysis "$SOURCE" --authority-npz "$AUTHORITY" \
        --output-root "$OUT" --dataset "$DATASET" --seed "$SEED" --regime "$REGIME" \
        --proxy-size "$PROXY_SIZE"
    fi
  fi
done
