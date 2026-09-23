#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent-metastone}
TRAIN_SCRIPT="${REPO_ROOT}/examples/quickstart/training/train_qwen3p5_dense_test.sh"

# ============================================================
# Model / Agent
# ============================================================
MODEL_NAME=${MODEL_NAME:-"Qwen3.5-9B"}
TASK_CONFIG=${TASK_CONFIG:-"${REPO_ROOT}/examples/quickstart/training/task_config_mini_swe_agent_blackbox_benchmark.yaml"}

# ============================================================
# V1 separate_async
# ============================================================
TRAINER_MODE=${TRAINER_MODE:-"separate_async"}
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-1}
ROLLOUT_CHECKPOINT_BACKEND=${ROLLOUT_CHECKPOINT_BACKEND:-"nccl"}

# ============================================================
# GPU resource split
# Single 8-GPU node default: 4 trainer + 4 rollout
# ============================================================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}

ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-6}

# Uni-Agent framework workers; normally follow rollout GPU count.
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-$((ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))}

# ============================================================
# Sequence length
# ============================================================
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}

# ============================================================
# Rollout
# ============================================================
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
GEN_TP=${GEN_TP:-1}

# ============================================================
# Megatron training parallelism
# ============================================================
TP=${TP:-1}
PP=${PP:-1}
CP=${CP:-2}


# ============================================================
# Batch
# separate_async requires:
# TRAIN_PROMPT_BSZ = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE
# ============================================================
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}

# ============================================================
# Training
# ============================================================
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-10}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:--1}

# ============================================================
# Quantization
# ============================================================
ACTOR_QUANTIZATION=${ACTOR_QUANTIZATION:-""}
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-""}

# ============================================================
# Experiment
# ============================================================
EXP_NAME=${EXP_NAME:-"separate_async_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S)"}

TRAINER_GPUS=$((NNODES * NGPUS_PER_NODE))
ROLLOUT_GPUS=$((ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
TOTAL_GPUS=$((TRAINER_GPUS + ROLLOUT_GPUS))

expected_train_bsz=$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))
if [[ ${TRAIN_PROMPT_BSZ} -ne ${expected_train_bsz} ]]; then
    echo "Invalid separate_async batch configuration:" >&2
    echo "  TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ}" >&2
    echo "  PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP}" >&2
    echo "  PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE}" >&2
    echo "Expected TRAIN_PROMPT_BSZ=${expected_train_bsz}" >&2
    exit 2
fi

echo "================================================="
echo "Model:                 ${MODEL_NAME}"
echo "Experiment:            ${EXP_NAME}"
echo "Trainer mode:          ${TRAINER_MODE}"
echo "-------------------------------------------------"
echo "Trainer GPUs:          ${TRAINER_GPUS}"
echo "Rollout GPUs:          ${ROLLOUT_GPUS}"
echo "Total required GPUs:   ${TOTAL_GPUS}"
echo "Agent workers:         ${AGENT_NUM_WORKERS}"
echo "Checkpoint backend:    ${ROLLOUT_CHECKPOINT_BACKEND}"
echo "-------------------------------------------------"
echo "Prompt length:         ${MAX_PROMPT_LENGTH}"
echo "Response length:       ${MAX_RESPONSE_LENGTH}"
echo "Rollout N:             ${N_RESP_PER_PROMPT}"
echo "Train TP/PP/CP:        ${TP}/${PP}/${CP}"
echo "Generation TP:         ${GEN_TP}"
echo "-------------------------------------------------"
echo "Train batch:           ${TRAIN_PROMPT_BSZ}"
echo "PPO mini batch:        ${PPO_MINI_BATCH_SIZE}"
echo "Parameter sync step:   ${PARAMETER_SYNC_STEP}"
echo "Warmup batches:        ${NUM_WARMUP_BATCHES}"
echo "Training steps:        ${TOTAL_TRAINING_STEPS}"
echo "Actor quantization:    ${ACTOR_QUANTIZATION:-default}"
echo "Rollout quantization:  ${ROLLOUT_QUANTIZATION:-default}"
echo "================================================="

env \
    MODEL_NAME="${MODEL_NAME}" \
    TASK_CONFIG="${TASK_CONFIG}" \
    EXP_NAME="${EXP_NAME}" \
    \
    TRAINER_MODE="${TRAINER_MODE}" \
    NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES}" \
    PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP}" \
    ROLLOUT_CHECKPOINT_BACKEND="${ROLLOUT_CHECKPOINT_BACKEND}" \
    \
    NNODES="${NNODES}" \
    NGPUS_PER_NODE="${NGPUS_PER_NODE}" \
    ROLLOUT_NNODES="${ROLLOUT_NNODES}" \
    ROLLOUT_NGPUS_PER_NODE="${ROLLOUT_NGPUS_PER_NODE}" \
    AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS}" \
    \
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" \
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
    N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT}" \
    GEN_TP="${GEN_TP}" \
    \
    TP="${TP}" \
    PP="${PP}" \
    CP="${CP}" \
    \
    TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ}" \
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
    TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}" \
    SAVE_FREQ="${SAVE_FREQ}" \
    TEST_FREQ="${TEST_FREQ}" \
    \
    ACTOR_QUANTIZATION="${ACTOR_QUANTIZATION}" \
    ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION}" \
    \
    bash "${TRAIN_SCRIPT}"
