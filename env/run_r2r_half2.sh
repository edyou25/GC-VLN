#!/usr/bin/env bash
# Run the fixed 50 R2R val_unseen episodes inside the navigation container / env.
# Usage (in container or activated GC-VLN env):
#   bash scripts/run_r2r_50.sh
set -euo pipefail
cd "$(dirname "$0")/.."

EPISODES=($(seq 920 1838))

# yacs list token, e.g. [53,1685,...]
eps_csv=$(IFS=,; echo "${EPISODES[*]}")
eps_list="[${eps_csv}]"

DATASET=r2r
SPLIT_NUM=1
SPLIT_INDEX=0
GSAM2_SERVER_PORT="${GSAM2_SERVER_PORT:-7001}"
DEVICE="${DEVICE:-0}"
EXP_CONFIG=config/r2r_vlnce.yaml
EXPERIMENT_ID="r2r_half2_$(date +%Y%m%d-%H%M%S)"
export NLTK_DATA="${NLTK_DATA:-/opt/nltk_data:./data/nltk_data}"

echo "========================================="
echo "GC-VLN R2R 50-episode smoke eval"
echo "episodes: ${#EPISODES[@]}"
echo "config: $EXP_CONFIG"
echo "experiment: $EXPERIMENT_ID"
echo "========================================="

CUDA_VISIBLE_DEVICES=$DEVICE python main.py \
  --exp-config "$EXP_CONFIG" \
  --split_num "$SPLIT_NUM" \
  --split_index "$SPLIT_INDEX" \
  --GSAM2_server_port "$GSAM2_SERVER_PORT" \
  --dataset "$DATASET" \
  --experiment_id "$EXPERIMENT_ID" \
  SIMULATOR_GPU_IDS "[0]" \
  TORCH_GPU_IDS "[0]" \
  GPU_NUMBERS 1 \
  NUM_ENVIRONMENTS 1 \
  TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING True \
  TASK_CONFIG.DATASET.EPISODES_ALLOWED "$eps_list"

echo "R2R 50-episode evaluation completed: $EXPERIMENT_ID"
