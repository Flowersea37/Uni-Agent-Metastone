#!/usr/bin/env bash
set -euo pipefail

# mini-swe-agent black-box quickstart.
#
# Fill these env vars or edit the defaults below:
#   DATA_PATH: SWE-Bench parquet, e.g. ~/data/swe_agent/swe_bench_verified.parquet
#   BASE_URL:  OpenAI-compatible /v1 endpoint reachable from the sandbox
#   MODEL:     served model name
#   API_KEY:   bearer token if your endpoint needs one

DATA_PATH="${DATA_PATH:-/data/xgq/data/swe_agent/swe_bench_verified_sample1.parquet}"
TASK_CONFIG="${TASK_CONFIG:-examples/quickstart/inference/task_config_mini_swe_agent_blackbox.yaml}"
API_KEY="${API_KEY:-EMPTY}"
LOG_DIR="${LOG_DIR:-/data/xgq/projects/RLs/uni-agent-metastone/logs}"
LIMIT="${LIMIT:-1}"
CONCURRENCY="${CONCURRENCY:-1}"
BASE_URL="https://api.vectron.meta-stone.com/v1"
MODEL="DeepSeek/DeepSeek-V4-Flash-0731"

: "${BASE_URL:?Set BASE_URL to an OpenAI-compatible /v1 endpoint reachable from inside the sandbox. Do not use localhost with a remote sandbox.}"
: "${MODEL:?Set MODEL to the served model name.}"

# External API mode.
# python3 examples/inference/parallel_infer_api.py \
#     --data-path "${DATA_PATH}" \
#     --task-config "${TASK_CONFIG}" \
#     --base-url "${BASE_URL}" \
#     --model "${MODEL}" \
#     --api-key "${API_KEY}" \
#     --log-dir "${LOG_DIR}" \
#     --concurrency "${CONCURRENCY}" \
#     --limit "${LIMIT}" \
#     --result-path "${LOG_DIR}/results.json"

# verl rollout mode.
# Uncomment after you have Ray + GPUs ready.
#
ray stop -f

ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=4

ray status

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=1

ray job submit \
    --runtime-env examples/quickstart/inference/runtime_env.yaml \
    --working-dir . \
    -- python3 examples/inference/parallel_infer_verl.py \
    --data-path "${DATA_PATH}" \
    --model-path /data/xgq/models/Qwen/Qwen3.5-0.8B \
    --task-config "${TASK_CONFIG}" \
    --tool-parser qwen3_coder \
    --tensor-parallel-size 1 \
    --nnodes 1 \
    --n-gpus-per-node 1 \
    --log-dir "${LOG_DIR}" \
    --concurrency "${CONCURRENCY}" \
    --limit "${LIMIT}"
