#!/usr/bin/env bash
# Actor-update memory/TP/CP benchmark. No agent rollout or teacher requests.
# Run inside verl_test on the head node, with the selected trainer GPUs free.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export BENCH_GPUS="${BENCH_GPUS:-4}"
export TP="${TP:-2}"
export CP="${CP:-2}"
export PP="${PP:-1}"
export BENCH_STEPS="${BENCH_STEPS:-1}"
# The selected trajectory has 1,468 prompt + 127,264 response tokens. These
# benchmark-only limits sum to the production 128K context budget.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-3808}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-127264}"
export BENCH_SAMPLES="${BENCH_SAMPLES:-8}"
export TRAIN_PROMPT_BSZ=1 PPO_MINI_BATCH_SIZE=1 PARAMETER_SYNC_STEP=1
export N_RESP_PER_PROMPT="${BENCH_SAMPLES}" TRAINER_MODE=colocate_async
export NNODES=1 NGPUS_PER_NODE="${BENCH_GPUS}"
export GEN_TP=1 AGENT_NUM_WORKERS="${BENCH_GPUS}"
export TOTAL_TRAINING_STEPS="${BENCH_STEPS}" SAVE_FREQ=-1 TEST_FREQ=-1

for value in "${BENCH_GPUS}" "${TP}" "${CP}" "${PP}" "${BENCH_STEPS}" "${BENCH_SAMPLES}"; do
    if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "BENCH_GPUS, TP, CP, PP, BENCH_STEPS, and BENCH_SAMPLES must be positive integers" >&2
        exit 2
    fi
done
if (( BENCH_GPUS % (TP * CP * PP) != 0 )); then
    echo "TP*CP*PP must divide BENCH_GPUS" >&2
    exit 2
fi

TRAJECTORY="${TRAJECTORY:-${REPO_ROOT}/train_logs/logs/Uni-Agent-Qwen3.5-9B-swe-agent-opd-fsdp/202609040642_exp/step_7/session-sample-0-rollout-0-40e7d0ffac1f496b92e0beac1b79bd0e/trajectory.npz}"
export VERL_FIXED_BATCH_PATH="${VERL_FIXED_BATCH_PATH:-${REPO_ROOT}/train_logs/fixed_trajectory/opd_128k_k1_samples${BENCH_SAMPLES}}"
export VERL_FIXED_BATCH_AUDIT="${VERL_FIXED_BATCH_AUDIT:-${REPO_ROOT}/train_logs/fixed_trajectory/results/gpus${BENCH_GPUS}_tp${TP}_cp${CP}_pp${PP}_samples${BENCH_SAMPLES}/audit.jsonl}"
export GPU_MEMORY_CSV="${GPU_MEMORY_CSV:-$(dirname "${VERL_FIXED_BATCH_AUDIT}")/gpu_memory.csv}"
export PROJECT_NAME="${PROJECT_NAME:-OPD-fixed-long-trajectory-megatron}"
export EXP_NAME="${EXP_NAME:-gpus${BENCH_GPUS}_tp${TP}_cp${CP}_pp${PP}_$(date +%Y%m%d%H%M%S)}"

if [[ "${BENCH_DRY_RUN:-0}" == 1 ]]; then
    python3 -m tools.run_opd_fixed_trajectory_benchmark
    exit 0
fi

if [[ ! -f "${TRAJECTORY}" ]]; then
    echo "Trajectory not found: ${TRAJECTORY}" >&2
    exit 1
fi

# Do not stop, reconfigure, or share a live Ray cluster from normal training.
if ray status >/dev/null 2>&1; then
    echo "A Ray cluster is running. Stop training and its Ray cluster before benchmarking." >&2
    exit 1
fi
for ((gpu=0; gpu<BENCH_GPUS; gpu++)); do
    if [[ -n "$(nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" ]]; then
        echo "GPU ${gpu} is busy; refusing to affect an existing training process." >&2
        exit 1
    fi
done

mkdir -p "${VERL_FIXED_BATCH_PATH}" "$(dirname "${VERL_FIXED_BATCH_AUDIT}")"
python3 -m tools.prepare_opd_long_fixed_batch \
    --trajectory "${TRAJECTORY}" \
    --output-dir "${VERL_FIXED_BATCH_PATH}" \
    --max-total-tokens "$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))" \
    --max-prompt-tokens "${MAX_PROMPT_LENGTH}" \
    --max-response-tokens "${MAX_RESPONSE_LENGTH}" \
    --samples "${BENCH_SAMPLES}" \
    --steps "${BENCH_STEPS}"

echo "Fixed actor update: GPUs=${BENCH_GPUS}, TP=${TP}, CP=${CP}, PP=${PP}, samples=${BENCH_SAMPLES}, steps=${BENCH_STEPS}"
echo "Input: ${VERL_FIXED_BATCH_PATH}/metadata.json"
python3 -m tools.run_opd_fixed_trajectory_benchmark
