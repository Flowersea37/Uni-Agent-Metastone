#!/usr/bin/env bash
# Two-node OPD: head GPUs 0-3 train Qwen3.5-9B with Megatron, head GPUs 4-7
# serve the Qwen3.8 teacher, worker GPUs 0-7 run student rollouts.
# Run inside the head node's verl_test container.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export OPD_BACKEND=megatron
# The reference uses TP=8 on eight trainer GPUs. This OPD topology has four
# trainer GPUs, so TP=2 and CP=2 use all four while splitting long sequences.
export MEGATRON_TP="${MEGATRON_TP:-4}"
export MEGATRON_CP="${MEGATRON_CP:-1}"
export MEGATRON_PP="${MEGATRON_PP:-1}"

# Megatron CP splits the long trajectory across GPUs; TP shards the model.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-122880}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + MEGATRON_CP - 1) / MEGATRON_CP))}"

# Match train_qwen3p5_dense_test.sh: one prompt, eight responses, one update.
# Separate async requires TRAIN_BATCH_SIZE = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-1}"
export PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"

# The worker has eight rollout GPUs: TP=1 creates eight replicas, as in the
# reference's default GEN_TP=1. Match its vLLM memory and agent concurrency.
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-4}"
export CONCURRENCY="${CONCURRENCY:-32}"

export PROJECT_NAME="${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-Qwen3.8-Flash-Next-FP8-OPD-megatron-separate-2node}"

exec bash "${SCRIPT_DIR}/train_opd_multi_node_separate.sh" "$@"
