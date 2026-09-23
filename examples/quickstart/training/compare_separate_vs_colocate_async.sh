#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent-metastone}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-"${REPO_ROOT}/examples/quickstart/training/train_qwen3p5_dense_test.sh"}
ANALYZER=${ANALYZER:-"${REPO_ROOT}/tools/analyze_async_mode_comparison.py"}
RUNTIME_DIR=${RUNTIME_DIR:-"${REPO_ROOT}/train_logs"}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-4B}
PROJECT_NAME=${PROJECT_NAME:-"Uni-Agent-${MODEL_NAME}-megatron"}
COMPARISON_NAME=${COMPARISON_NAME:-"async_mode_comparison_$(date +%Y%m%d_%H%M%S)"}
RESULT_ROOT=${RESULT_ROOT:-"${RUNTIME_DIR}/async_mode_comparisons/${COMPARISON_NAME}"}
TASK_CONFIG=${TASK_CONFIG:-"${REPO_ROOT}/examples/quickstart/training/task_config_mini_swe_agent_blackbox_benchmark.yaml"}

# Both arms use eight GPUs in total. The separated arm dedicates four to each
# role; the colocated arm lets training and rollout time-share all eight.
TOTAL_GPUS=${TOTAL_GPUS:-8}
SEPARATE_TRAINER_GPUS=${SEPARATE_TRAINER_GPUS:-4}
SEPARATE_ROLLOUT_GPUS=${SEPARATE_ROLLOUT_GPUS:-4}
STEPS=${STEPS:-10}
# REPORT_WARMUP_STEPS only controls how many initial training steps the report
# excludes. NUM_WARMUP_BATCHES controls the rollout producer's initial prefetch.
REPORT_WARMUP_STEPS=${REPORT_WARMUP_STEPS:-1}
NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
TP=${TP:-1}
CP=${CP:-2}
GEN_TP=${GEN_TP:-1}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-1}
# MODES=${MODES:-"separate_async colocate_async"}
MODES=${MODES:-"separate_async"}

if (( SEPARATE_TRAINER_GPUS + SEPARATE_ROLLOUT_GPUS != TOTAL_GPUS )); then
    echo "Separate GPU split must sum to TOTAL_GPUS=${TOTAL_GPUS}." >&2
    exit 2
fi

mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"

run_one() {
    local mode=$1
    local output_dir="${RESULT_ROOT}/${mode}"
    local trainer_gpus rollout_gpus agent_workers
    mkdir -p "${output_dir}"

    if [[ ${mode} == separate_async ]]; then
        trainer_gpus=${SEPARATE_TRAINER_GPUS}
        rollout_gpus=${SEPARATE_ROLLOUT_GPUS}
        agent_workers=${SEPARATE_ROLLOUT_GPUS}
    else
        trainer_gpus=${TOTAL_GPUS}
        # Ignored by colocate_async, but keep the value valid for config parsing.
        rollout_gpus=${SEPARATE_ROLLOUT_GPUS}
        agent_workers=${TOTAL_GPUS}
    fi

    printf '%s\n' "$(date +%s.%N)" > "${output_dir}/wall_start.txt"
    echo "===== ${mode}: trainer=${trainer_gpus}, rollout=${rollout_gpus}, steps=${STEPS} ====="
    set +e
    env \
        REPO_ROOT="${REPO_ROOT}" RUNTIME_DIR="${RUNTIME_DIR}" \
        MODEL_NAME="${MODEL_NAME}" PROJECT_NAME="${PROJECT_NAME}" \
        EXP_NAME="${COMPARISON_NAME}_${mode}" TASK_CONFIG="${TASK_CONFIG}" \
        TRAINER_MODE="${mode}" NNODES=1 NGPUS_PER_NODE="${trainer_gpus}" \
        ROLLOUT_NNODES=1 ROLLOUT_NGPUS_PER_NODE="${rollout_gpus}" \
        AGENT_NUM_WORKERS="${agent_workers}" \
        GPU_MEMORY_CSV="${output_dir}/gpu_memory_by_step.csv" \
        ROLLOUT_GPU_MEMORY_CSV="${output_dir}/gpu_samples.csv" \
        ROLLOUT_ENGINE_MEMORY_CSV="${output_dir}/rollout_engine_memory.csv" \
        ROLLOUT_REQUEST_METRICS_CSV="${output_dir}/vllm_request_metrics.csv" \
        AGENT_LOG_DIR="${output_dir}/trajectories" \
        CKPTS_DIR="${output_dir}/checkpoints" \
        TOTAL_TRAINING_STEPS="${STEPS}" \
        TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ}" PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
        PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP}" NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES}" \
        N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT}" \
        MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
        CP="${CP}" TP="${TP}" GEN_TP="${GEN_TP}" \
        DATA_SHUFFLE=False DATA_SEED=42 ROLLOUT_SEED=42 \
        DO_SAMPLE=True VAL_DO_SAMPLE=False TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1 \
        SAVE_FREQ=-1 TEST_FREQ=-1 \
        bash "${TRAIN_SCRIPT}" 2>&1 | tee "${output_dir}/train.log"
    local status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$(date +%s.%N)" > "${output_dir}/wall_end.txt"
    printf '%s\n' "${status}" > "${output_dir}/exit_code.txt"
    return "${status}"
}

failed=()
for mode in ${MODES}; do
    case "${mode}" in
        colocate_async|separate_async)
            run_one "${mode}" || failed+=("${mode}")
            ;;
        *)
            echo "Unknown mode: ${mode}" >&2
            exit 2
            ;;
    esac
done

if (( ${#failed[@]} > 0 )); then
    echo "Failed modes: ${failed[*]}; no comparison report was generated or published." >&2
    exit 1
fi

python3 "${ANALYZER}" \
    --result-root "${RESULT_ROOT}" \
    --warmup-steps "${REPORT_WARMUP_STEPS}" \
    --expected-gpus "${TOTAL_GPUS}" \
    --output "${RESULT_ROOT}/comparison_report.md"

if [[ -s "${RESULT_ROOT}/comparison_report.md" ]]; then
    ln -sfn "${COMPARISON_NAME}" "${RUNTIME_DIR}/async_mode_comparisons/latest"
fi

echo "Report: ${RESULT_ROOT}/comparison_report.md"
