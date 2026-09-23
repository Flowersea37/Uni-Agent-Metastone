#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent}
RUNTIME_DIR=${RUNTIME_DIR:-"${REPO_ROOT}/train_logs"}
MODEL_NAME=${MODEL_NAME:-Qwen3.5-9B}
PROJECT_NAME=${PROJECT_NAME:-"Uni-Agent-${MODEL_NAME}-megatron"}
BENCHMARK_NAME=${BENCHMARK_NAME:-"actor_fixed_batch_$(date +%Y%m%d_%H%M%S)"}
RESULT_ROOT="${RUNTIME_DIR}/gpu_memory/fixed_batch/${BENCHMARK_NAME}"
FIXED_BATCH_PATH=${FIXED_BATCH_SOURCE_DIR:-"${RESULT_ROOT}/fixed_update_batches"}
SOURCE_AUDIT=${FIXED_BATCH_SOURCE_AUDIT:-"${RESULT_ROOT}/dump/fixed_batch_audit.jsonl"}
REPLAY_STEPS=${REPLAY_STEPS:-6}
WARMUP_STEPS=${WARMUP_STEPS:-3}
TP=${TP:-2}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TASK_CONFIG=${TASK_CONFIG:-"${REPO_ROOT}/examples/quickstart/training/task_config_mini_swe_agent_blackbox_benchmark.yaml"}
SOURCE_TRAIN_FILE=${SOURCE_TRAIN_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}
DATA_ROW_INDEX=${DATA_ROW_INDEX:-1}
BENCHMARK_DATA_FILE="${RESULT_ROOT}/benchmark_rows_${DATA_ROW_INDEX}_${REPLAY_STEPS}.parquet"
PHASES=${PHASES:-"dump fixed_default fixed_actor_fp8 analyze"}

mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"
python "${REPO_ROOT}/tools/select_parquet_row.py" \
    "${SOURCE_TRAIN_FILE}" "${BENCHMARK_DATA_FILE}" \
    --index "${DATA_ROW_INDEX}" --count "${REPLAY_STEPS}"

run_job() {
    local label=$1
    local mode=$2
    local actor_quantization=$3
    local steps=$4
    local output_dir="${RESULT_ROOT}/${label}"
    mkdir -p "${output_dir}"
    echo "===== ${label}: mode=${mode}, actor=${actor_quantization:-default}, steps=${steps} ====="

    env \
        REPO_ROOT="${REPO_ROOT}" RUNTIME_DIR="${RUNTIME_DIR}" \
        MODEL_NAME="${MODEL_NAME}" PROJECT_NAME="${PROJECT_NAME}" \
        TRAIN_FILE="${BENCHMARK_DATA_FILE}" TEST_FILE="${BENCHMARK_DATA_FILE}" \
        EXP_NAME="${BENCHMARK_NAME}_${label}" TASK_CONFIG="${TASK_CONFIG}" \
        VERL_FIXED_BATCH_MODE="${mode}" VERL_FIXED_BATCH_PATH="${FIXED_BATCH_PATH}" \
        VERL_FIXED_BATCH_DIRECT="$([[ ${mode} == replay ]] && echo True || echo False)" \
        VERL_FIXED_BATCH_AUDIT="${output_dir}/fixed_batch_audit.jsonl" \
        GPU_MEMORY_CSV="${output_dir}/gpu_memory_by_step.csv" \
        ROLLOUT_GPU_MEMORY_CSV="${output_dir}/rollout_gpu_samples.csv" \
        ROLLOUT_ENGINE_MEMORY_CSV="${output_dir}/rollout_engine_memory.csv" \
        ROLLOUT_REQUEST_METRICS_CSV="${output_dir}/vllm_request_metrics.csv" \
        ACTOR_QUANTIZATION="${actor_quantization}" ROLLOUT_QUANTIZATION="" \
        quanti_pa="" QUANTI_PA="" TOTAL_TRAINING_STEPS="${steps}" \
        TP="${TP}" N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT}" \
        MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}" \
        DATA_SHUFFLE=False DATA_SEED=42 ROLLOUT_SEED=42 \
        DO_SAMPLE=False VAL_DO_SAMPLE=False FULL_DETERMINISM=False \
        TEMPERATURE=1.0 TOP_P=1.0 TOP_K=-1 \
        SEQUENCE_PARALLEL=True USE_REMOVE_PADDING=True USE_FUSED_KERNELS=True \
        SAVE_FREQ=-1 TEST_FREQ=-1 \
        python "${REPO_ROOT}/tools/run_fixed_batch_train.py"

    local recorded=$(( $(wc -l < "${output_dir}/gpu_memory_by_step.csv") - 1 ))
    [[ ${recorded} -eq ${steps} ]] || {
        echo "${label}: expected ${steps} CSV rows, got ${recorded}" >&2
        return 1
    }
}

# The dump run creates one distinct update batch per step. Both timed runs start
# fresh from the same checkpoint and replay the exact same ordered batch sequence.
for phase in ${PHASES}; do
    case "${phase}" in
        dump) run_job dump dump "" "${REPLAY_STEPS}" ;;
        fixed_default) run_job fixed_default replay "" "${REPLAY_STEPS}" ;;
        fixed_actor_fp8) run_job fixed_actor_fp8 replay FP8 "${REPLAY_STEPS}" ;;
        analyze)
            python "${REPO_ROOT}/tools/analyze_fixed_batch.py" \
                --default-csv "${RESULT_ROOT}/fixed_default/gpu_memory_by_step.csv" \
                --fp8-csv "${RESULT_ROOT}/fixed_actor_fp8/gpu_memory_by_step.csv" \
                --dump-audit "${SOURCE_AUDIT}" \
                --default-audit "${RESULT_ROOT}/fixed_default/fixed_batch_audit.jsonl" \
                --fp8-audit "${RESULT_ROOT}/fixed_actor_fp8/fixed_batch_audit.jsonl" \
                --skip-first "${WARMUP_STEPS}" --output-dir "${RESULT_ROOT}/summary"
            ;;
        *) echo "Unknown phase: ${phase}" >&2; exit 2 ;;
    esac
done

if [[ -s "${RESULT_ROOT}/summary/fixed_batch_report.md" ]]; then
    ln -sfn "${BENCHMARK_NAME}" "${RUNTIME_DIR}/gpu_memory/fixed_batch/latest"
    echo "Report: ${RESULT_ROOT}/summary/fixed_batch_report.md"
fi
