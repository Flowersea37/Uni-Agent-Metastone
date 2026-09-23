#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent}
TRAIN_SCRIPT="${REPO_ROOT}/examples/quickstart/training/train_qwen3p5_dense_test.sh"
ANALYZER="${REPO_ROOT}/tools/analyze_gpu_memory.py"
ROLLOUT_ANALYZER="${REPO_ROOT}/tools/analyze_rollout_gpu_memory.py"
RUNTIME_DIR=${RUNTIME_DIR:-"${REPO_ROOT}/train_logs"}
# MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
MODEL_NAME=${MODEL_NAME:-"Qwen3.5-9B"}
PROJECT_NAME=${PROJECT_NAME:-"Uni-Agent-${MODEL_NAME}-megatron"}
COMPARISON_NAME=${COMPARISON_NAME:-"quant_memory_rollout_quantization_$(date +%Y%m%d_%H%M%S)"}
RESULT_ROOT="${RUNTIME_DIR}/gpu_memory/comparisons/${COMPARISON_NAME}"
TASK_CONFIG=${TASK_CONFIG:-"${REPO_ROOT}/examples/quickstart/training/task_config_mini_swe_agent_blackbox_benchmark.yaml"}
# RESULT_ROOT="/data/xgq/projects/RLs/uni-agent/train_logs/gpu_memory/comparisons/quant_memory_20260909_100120"
STEPS=${STEPS:-10}
RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-32768}
ROLLOUT_N=${N_RESP_PER_PROMPT:-8}
TRAIN_TP=${TP:-2}
EXPERIMENTS=${EXPERIMENTS:-"default rollout_fp8 actor_fp8 both_fp8"}
# Keep the initial dataset prompts identical across all independent Ray jobs.
DATA_SHUFFLE=${DATA_SHUFFLE:-False}
DATA_SEED=${DATA_SEED:-42}
ROLLOUT_SEED=${ROLLOUT_SEED:-42}
mkdir -p "${RESULT_ROOT}"
cd "${REPO_ROOT}"

run_experiment() {
    local label=$1
    local actor_quantization=$2
    local rollout_quantization=$3
    local experiment_name="${COMPARISON_NAME}_${label}"
    local csv_path="${RESULT_ROOT}/${label}/gpu_memory_by_step.csv"
    local rollout_samples="${RESULT_ROOT}/${label}/rollout_gpu_samples.csv"
    local rollout_summary="${RESULT_ROOT}/${label}/rollout_gpu_by_step.csv"
    local engine_memory="${RESULT_ROOT}/${label}/rollout_engine_memory.csv"
    local request_metrics="${RESULT_ROOT}/${label}/vllm_request_metrics.csv"

    echo
    echo "===== Starting ${label}: actor=${actor_quantization:-default}, rollout=${rollout_quantization:-default}, ${STEPS} steps ====="
    # Greedy decoding is controlled by DO_SAMPLE=False. The Megatron fused
    # log-prob kernel still requires a positive temperature for actor scoring.
    env \
        quanti_pa="" \
        QUANTI_PA="" \
        ACTOR_QUANTIZATION="${actor_quantization}" \
        ROLLOUT_QUANTIZATION="${rollout_quantization}" \
        EXP_NAME="${experiment_name}" \
        MODEL_NAME="${MODEL_NAME}" \
        PROJECT_NAME="${PROJECT_NAME}" \
        TASK_CONFIG="${TASK_CONFIG}" \
        RUNTIME_DIR="${RUNTIME_DIR}" \
        GPU_MEMORY_CSV="${csv_path}" \
        ROLLOUT_GPU_MEMORY_CSV="${rollout_samples}" \
        ROLLOUT_ENGINE_MEMORY_CSV="${engine_memory}" \
        ROLLOUT_REQUEST_METRICS_CSV="${request_metrics}" \
        TOTAL_TRAINING_STEPS="${STEPS}" \
        N_RESP_PER_PROMPT="${ROLLOUT_N}" \
        DATA_SHUFFLE="${DATA_SHUFFLE}" \
        DATA_SEED="${DATA_SEED}" \
        ROLLOUT_SEED="${ROLLOUT_SEED}" \
        TEMPERATURE=1.0 \
        DO_SAMPLE=False \
        VAL_DO_SAMPLE=False \
        TOP_P=1.0 \
        TOP_K=-1 \
        FULL_DETERMINISM=False \
        SAVE_FREQ=-1 \
        TEST_FREQ=-1 \
        TP="${TRAIN_TP}" \
        MAX_RESPONSE_LENGTH="${RESPONSE_LENGTH}" \
        SEQUENCE_PARALLEL=True \
        USE_REMOVE_PADDING=True \
        USE_FUSED_KERNELS=True \
        bash "${TRAIN_SCRIPT}" || return $?

    if [[ ! -s "${csv_path}" ]]; then
        echo "Experiment ${label} finished without producing ${csv_path}" >&2
        return 1
    fi

    local completed_steps
    completed_steps=$(( $(wc -l < "${csv_path}") - 1 ))
    if [[ ${completed_steps} -ne ${STEPS} ]]; then
        echo "Experiment ${label} recorded ${completed_steps} steps instead of ${STEPS}." >&2
        return 1
    fi
    python "${ROLLOUT_ANALYZER}" "${csv_path}" "${rollout_samples}" \
        --engine-memory "${engine_memory}" --request-metrics "${request_metrics}" \
        --output "${rollout_summary}" || return $?
    echo "===== Completed ${label}: ${csv_path} ====="
}

successful_experiments=()
failed_experiments=()
for experiment in ${EXPERIMENTS}; do
    status=0
    case "${experiment}" in
        default) run_experiment default "" "" || status=$? ;;
        rollout_fp8) run_experiment rollout_fp8 "" FP8 || status=$? ;;
        actor_fp8) run_experiment actor_fp8 FP8 "" || status=$? ;;
        both_fp8|fp8) run_experiment both_fp8 FP8 FP8 || status=$? ;;
        *) echo "Unknown experiment '${experiment}'" >&2; status=2 ;;
    esac
    if [[ ${status} -eq 0 ]]; then
        successful_experiments+=("${experiment}")
    else
        failed_experiments+=("${experiment}:${status}")
        echo "===== FAILED ${experiment} (exit ${status}); continuing =====" >&2
    fi
done

analysis_inputs=()
for experiment in default rollout_fp8 actor_fp8 both_fp8; do
    csv_path="${RESULT_ROOT}/${experiment}/gpu_memory_by_step.csv"
    [[ -s "${csv_path}" ]] && analysis_inputs+=("${csv_path}")
done
if [[ ${#analysis_inputs[@]} -ge 2 ]]; then
    python "${ANALYZER}" "${analysis_inputs[@]}" \
        --skip-first 1 --output-dir "${RESULT_ROOT}/summary" \
        || echo "Summary generation failed; raw run data is preserved." >&2
fi

echo
echo "Experiments attempted: ${EXPERIMENTS}"
echo "Successful: ${successful_experiments[*]:-none}"
echo "Failed: ${failed_experiments[*]:-none}"
echo "Per-run files are under: ${RESULT_ROOT}/{default,rollout_fp8,actor_fp8,both_fp8}/"
echo "Summary CSV: ${RESULT_ROOT}/summary/gpu_memory_summary.csv"
echo "Training metrics plot: ${RESULT_ROOT}/summary/training_metrics_by_step.png"
echo "Rollout engine plot: ${RESULT_ROOT}/summary/rollout_engine_metrics_by_step.png"
echo "Markdown analysis: ${RESULT_ROOT}/summary/result_analysis.md"
echo "Stable latest link: ${RUNTIME_DIR}/gpu_memory/comparisons/latest"

if [[ ${#failed_experiments[@]} -gt 0 ]]; then
    exit 1
fi

# Publish only a fully successful comparison that actually produced a report.
# A single-run smoke test is useful for validation, but must not replace the
# last comparable result behind the stable link.
if [[ -s "${RESULT_ROOT}/summary/result_analysis.md" ]]; then
    ln -sfn "${COMPARISON_NAME}" "${RUNTIME_DIR}/gpu_memory/comparisons/latest"
else
    echo "Stable latest link unchanged: no comparison report was generated."
fi
