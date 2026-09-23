#!/usr/bin/env bash
# Start the external FP8 teacher on GPUs 4-7 for train_opd_entry.sh.
set -euo pipefail

CONTAINER_NAME="${OPD_TEACHER_CONTAINER:-opd_teacher_qwen38}"
MODEL="${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8}"
PORT="${OPD_TEACHER_PORT:-8008}"
BIND_HOST="${OPD_TEACHER_BIND_HOST:-127.0.0.1}"
GPU_DEVICES="${OPD_TEACHER_GPUS:-4,5,6,7}"
IMAGE="${OPD_TEACHER_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"
PROMPT_TOKENS="${MAX_PROMPT_LENGTH:-8192}"
RESPONSE_TOKENS="${MAX_RESPONSE_LENGTH:-122880}"
MAX_MODEL_LEN="${OPD_MAX_MODEL_LEN:-$((PROMPT_TOKENS + RESPONSE_TOKENS + 1))}"

if docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    RUNNING_MAX_MODEL_LEN="$(docker inspect --format '{{json .Config.Cmd}}' "${CONTAINER_NAME}" | python3 -c '
import json
import sys
cmd = json.load(sys.stdin)
print(cmd[cmd.index("--max-model-len") + 1] if "--max-model-len" in cmd else "unknown")
')"
    RUNNING_BIND_HOST="$(docker inspect --format '{{json .Config.Cmd}}' "${CONTAINER_NAME}" | python3 -c '
import json
import sys
cmd = json.load(sys.stdin)
print(cmd[cmd.index("--host") + 1] if "--host" in cmd else "unknown")
')"
    if [[ "${RUNNING_MAX_MODEL_LEN}" != "${MAX_MODEL_LEN}" || "${RUNNING_BIND_HOST}" != "${BIND_HOST}" ]]; then
        echo "Teacher ${CONTAINER_NAME}: running max_model_len=${RUNNING_MAX_MODEL_LEN}, host=${RUNNING_BIND_HOST}; requested max_model_len=${MAX_MODEL_LEN}, host=${BIND_HOST}." >&2
        if [[ "${OPD_TEACHER_REPLACE_EXISTING:-0}" != 1 ]]; then
            echo "Stop the existing container, or set OPD_TEACHER_REPLACE_EXISTING=1 to replace it." >&2
            exit 1
        fi
        docker rm -f "${CONTAINER_NAME}" >/dev/null
    else
        echo "Teacher container ${CONTAINER_NAME} is already running with max_model_len=${MAX_MODEL_LEN}."
    fi
fi
if ! docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
    if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi
    docker run -d \
        --name "${CONTAINER_NAME}" \
        --gpus "\"device=${GPU_DEVICES}\"" \
        --ipc host --network host \
        -v /data:/data:ro \
        "${IMAGE}" "${MODEL}" \
        --host "${BIND_HOST}" --port "${PORT}" \
        --tensor-parallel-size 4 --enable-expert-parallel \
        --gpu-memory-utilization 0.80 \
        --max-model-len "${MAX_MODEL_LEN}" --max-num-batched-tokens 4096 \
        --max-logprobs 64 --enforce-eager \
        --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
        --trust-remote-code
fi

echo "Waiting for teacher at http://127.0.0.1:${PORT}/health ..."
for _ in $(seq 1 240); do
    if curl --silent --show-error --fail --max-time 2 \
        "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "Teacher is ready."
        exit 0
    fi
    if ! docker ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        docker logs --tail 80 "${CONTAINER_NAME}" >&2
        echo "Teacher container exited before becoming ready." >&2
        exit 1
    fi
    sleep 3
done
docker logs --tail 80 "${CONTAINER_NAME}" >&2
echo "Teacher did not become ready within 12 minutes." >&2
exit 1
