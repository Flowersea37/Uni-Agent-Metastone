#!/usr/bin/env bash
# Run inside verl_test on the head node. Head: 4 trainer + 4 teacher GPUs;
# worker: 8 rollout GPUs. Both machines need the same /data mount and verl_test.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
HEAD_IP="${HEAD_IP:-174.1.59.5}"
WORKER_IP="${WORKER_IP:-174.1.59.4}"
WORKER_SSH="${WORKER_SSH:-root@${WORKER_IP}}"
CONTAINER_NAME="${TRAIN_CONTAINER:-verl_test}"

# These addresses are interpolated into the remote shell command.
for address in "${HEAD_IP}" "${WORKER_IP}"; do
    if [[ ! "${address}" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "HEAD_IP and WORKER_IP must be IPv4 addresses: ${address}" >&2
        exit 2
    fi
done
if [[ ! "${CONTAINER_NAME}" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "Invalid TRAIN_CONTAINER: ${CONTAINER_NAME}" >&2
    exit 2
fi

export OPD_TRAINER_MODE=separate_async
export OPD_RAY_MANAGED_EXTERNALLY=1
export OPD_TRAINER_RAY_RESOURCE=opd_trainer
export NNODES=1 NGPUS_PER_NODE=4
export ROLLOUT_NNODES=1 ROLLOUT_NGPUS_PER_NODE=8
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPD_TEACHER_GPUS=4,5,6,7
export OPD_TEACHER_BIND_HOST=0.0.0.0
export OPD_TEACHER_REPLACE_EXISTING=1
export OPD_TEACHER_URL="http://${HEAD_IP}:${OPD_TEACHER_PORT:-8008}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond3}"
unset NCCL_IGNORE_NET_MISMATCH

export STUDENT_MODEL="${STUDENT_MODEL:-/data/xgq/models/Qwen/Qwen3.5-9B}"
export TEACHER_MODEL="${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-122880}"
export FSDP_SP_SIZE="${FSDP_SP_SIZE:-4}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + FSDP_SP_SIZE - 1) / FSDP_SP_SIZE))}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-1}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
export ROLLOUT_CHECKPOINT_BACKEND="${ROLLOUT_CHECKPOINT_BACKEND:-nccl}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export CONCURRENCY="${CONCURRENCY:-256}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
export PROJECT_NAME="${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-Qwen3.8-Flash-Next-FP8-OPD-separate-2node}"

if (( TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE )); then
    echo "TRAIN_BATCH_SIZE must equal PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE" >&2
    exit 2
fi

echo "Head ${HEAD_IP}: trainer GPUs 0-3, teacher GPUs 4-7; worker ${WORKER_IP}: rollout GPUs 0-7"
echo "Starting external teacher at ${OPD_TEACHER_URL}"
bash "${SCRIPT_DIR}/start_opd_teacher.sh"
if ! curl --silent --show-error --fail --max-time 5 "${OPD_TEACHER_URL}/health" >/dev/null; then
    echo "Teacher is not reachable via ${OPD_TEACHER_URL}; check its bind address and port" >&2
    exit 1
fi

echo "Starting Ray head with four trainer GPUs"
ray stop --force
ray start --head --node-ip-address="${HEAD_IP}" --port=6379 \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 --num-gpus=4 \
    --resources='{"opd_trainer":1}'

echo "Starting Ray worker with eight rollout GPUs"
ssh -o BatchMode=yes "${WORKER_SSH}" \
    "docker start ${CONTAINER_NAME} >/dev/null && docker exec ${CONTAINER_NAME} bash -lc 'set -e; cd ${REPO_ROOT}; export PYTHONPATH=${REPO_ROOT}/verl:${REPO_ROOT} CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}; unset NCCL_IGNORE_NET_MISMATCH; curl -fsS --max-time 5 ${OPD_TEACHER_URL}/health >/dev/null; ray stop --force; ray start --address=${HEAD_IP}:6379 --node-ip-address=${WORKER_IP} --num-gpus=8'"

echo "Waiting for the 4 + 8 GPU Ray cluster"
for _ in $(seq 1 60); do
    GPU_COUNT=$(python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)
    if (( GPU_COUNT >= 12 )); then
        break
    fi
    sleep 5
done
if (( GPU_COUNT < 12 )); then
    echo "Ray has ${GPU_COUNT}/12 GPUs after five minutes" >&2
    exit 1
fi
ray status

# Reuse the shared OPD configuration without restarting this two-node cluster.
exec bash "${SCRIPT_DIR}/train_opd_base.sh" "$@"
