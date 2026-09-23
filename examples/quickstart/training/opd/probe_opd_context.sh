#!/usr/bin/env bash
# Check whether the OPD teacher and/or student rollout can serve a target context.
# Usage: bash probe_opd_context.sh INPUT_TOKENS [teacher|student|both]
# Set PROBE_FULL_PREFILL=1 to also score a prompt of INPUT_TOKENS tokens.
set -euo pipefail

INPUT_TOKENS="${1:?Usage: $0 INPUT_TOKENS [teacher|student|both]}"
SIDE="${2:-both}"
if ! [[ "${INPUT_TOKENS}" =~ ^[0-9]+$ ]] || (( INPUT_TOKENS < 2 || INPUT_TOKENS > 262143 )); then
    echo "INPUT_TOKENS must be an integer from 2 to 262143." >&2
    exit 2
fi
if [[ "${SIDE}" != teacher && "${SIDE}" != student && "${SIDE}" != both ]]; then
    echo "SIDE must be teacher, student, or both." >&2
    exit 2
fi

MAX_MODEL_LEN=$((INPUT_TOKENS + 1))
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${REPO_ROOT}/train_logs/context_probes"
mkdir -p "${LOG_DIR}"
TMP_REQUEST="$(mktemp)"
TMP_RESPONSE="$(mktemp)"
STARTED=()
cleanup() {
    for name in "${STARTED[@]}"; do
        docker logs "${name}" > "${LOG_DIR}/${name}-${INPUT_TOKENS}.log" 2>&1 || true
        docker rm -f "${name}" >/dev/null 2>&1 || true
    done
    rm -f "${TMP_REQUEST}" "${TMP_RESPONSE}"
}
trap cleanup EXIT

run_probe() {
    local side="$1" name port model image gpu_count
    if [[ "${side}" == teacher ]]; then
        name="opd_context_teacher_probe"
        port=8018
        model="${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8}"
        image="${OPD_TEACHER_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
        gpu_count='"device=4,5,6,7"'
    else
        name="opd_context_student_probe"
        port=8019
        model="${STUDENT_MODEL:-/data/xgq/models/Qwen/Qwen3.5-9B}"
        image="${OPD_STUDENT_IMAGE:-verl:vllm024.dev2-uniagent-runable}"
        gpu_count='"device=0,1"'
    fi
    if docker container inspect "${name}" >/dev/null 2>&1; then
        echo "Container ${name} already exists; remove it before probing." >&2
        exit 1
    fi
    echo "${side}: testing ${INPUT_TOKENS} input tokens (max_model_len=${MAX_MODEL_LEN})"
    if [[ "${side}" == teacher ]]; then
        docker run -d --name "${name}" --gpus "${gpu_count}" --ipc host --network host \
            -v /data:/data:ro "${image}" "${model}" \
            --host 127.0.0.1 --port "${port}" \
            --tensor-parallel-size 4 --enable-expert-parallel \
            --gpu-memory-utilization 0.80 --max-model-len "${MAX_MODEL_LEN}" \
            --max-num-batched-tokens 4096 --max-logprobs 64 --enforce-eager \
            --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
            --trust-remote-code >/dev/null
    else
        docker run -d --name "${name}" --gpus "${gpu_count}" --ipc host --network host \
            -v /data:/data:ro --entrypoint vllm "${image}" serve "${model}" \
            --host 127.0.0.1 --port "${port}" \
            --tensor-parallel-size 2 --gpu-memory-utilization 0.4 \
            --max-model-len "${MAX_MODEL_LEN}" --max-num-batched-tokens "${MAX_MODEL_LEN}" \
            --enforce-eager --trust-remote-code >/dev/null
    fi
    STARTED+=("${name}")

    local ready=0
    for _ in $(seq 1 240); do
        if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if [[ "$(docker inspect -f '{{.State.Running}}' "${name}")" != true ]]; then
            break
        fi
        sleep 3
    done
    if (( ready == 0 )); then
        echo "${side}: FAILED to start. Last log lines:" >&2
        docker logs --tail 35 "${name}" >&2
        return 1
    fi

    python3 - "${model}" "${INPUT_TOKENS}" "${PROBE_FULL_PREFILL:-0}" > "${TMP_REQUEST}" <<'PY'
import json
import sys

model, length, full = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1"
prompt_ids = [9419, 1814] if not full else [9419] + [1814] * (length - 1)
json.dump({"model": model, "prompt": prompt_ids, "max_tokens": 1, "temperature": 0}, sys.stdout)
PY
    if ! curl -fsS --max-time 3600 -H 'Content-Type: application/json' \
        -d "@${TMP_REQUEST}" "http://127.0.0.1:${port}/v1/completions" \
        -o "${TMP_RESPONSE}"; then
        echo "${side}: FAILED completion request; response:" >&2
        head -c 2000 "${TMP_RESPONSE}" >&2 || true
        return 1
    fi
    echo "${side}: PASS startup and $([[ "${PROBE_FULL_PREFILL:-0}" == 1 ]] && echo full-length || echo short) request"
}

if [[ "${SIDE}" == teacher || "${SIDE}" == both ]]; then
    run_probe teacher
fi
if [[ "${SIDE}" == student || "${SIDE}" == both ]]; then
    run_probe student
fi
