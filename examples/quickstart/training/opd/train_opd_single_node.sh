#!/usr/bin/env bash
# Edit these experiment defaults here; train_opd_base.sh owns the launcher details.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"

# Models and task data.
export STUDENT_MODEL="${STUDENT_MODEL:-/data/xgq/models/Qwen/Qwen3.5-9B}"
export TEACHER_MODEL="${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8}"
export TRAIN_FILE="${TRAIN_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}"
export VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
export TASK_CONFIG="${TASK_CONFIG:-${REPO_ROOT}/examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml}"

# One node: four GPUs for the student/rollout and four for the teacher.
export OPD_TEACHER_URL="${OPD_TEACHER_URL:-http://127.0.0.1:8008}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NNODES="${NNODES:-1}"
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export TEACHER_NNODES="${TEACHER_NNODES:-1}"
export TEACHER_GPUS_PER_NODE="${TEACHER_GPUS_PER_NODE:-4}"
export ROLLOUT_TP="${ROLLOUT_TP:-2}"
export TEACHER_TP="${TEACHER_TP:-4}"
export TEACHER_EP="${TEACHER_EP:-4}"
export TEACHER_GPU_MEM_UTIL="${TEACHER_GPU_MEM_UTIL:-0.85}"

# Batch, context, and run length.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
if [[ -n "${OPD_TEACHER_URL}" ]]; then
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-122880}"
else
    export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-131072}"
fi
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-10}"

# Distillation and rollout capacity.
export DISTILLATION_LOSS_MODE="${DISTILLATION_LOSS_MODE:-k1}"
export DISTILLATION_TOPK="${DISTILLATION_TOPK:-64}"
export USE_POLICY_GRADIENT="${USE_POLICY_GRADIENT:-True}"
export CONCURRENCY="${CONCURRENCY:-256}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-8}"

export PROJECT_NAME="${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-Qwen3.8-Flash-Next-FP8-OPD}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-$(date +%Y%m%d%H%M)_exp}"
export RUNTIME_DIR="${RUNTIME_DIR:-${REPO_ROOT}/train_logs}"

# Check the teacher before train_opd_base.sh stops the existing Ray cluster.
# A model's tokenizer can load even when its architecture cannot be served.
if [[ -n "${OPD_TEACHER_URL}" ]]; then
    python3 - "${OPD_TEACHER_URL}" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/") + "/health"
try:
    with urllib.request.urlopen(url, timeout=5) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}")
except Exception as exc:
    raise SystemExit(
        f"OPD teacher service is not ready at {url}: {exc}. "
        "Start it with examples/quickstart/training/opd/start_opd_teacher.sh first."
    )
PY
else
python3 - "${TEACHER_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1])
config_path = model_path / "config.json"
if not config_path.is_file():
    raise SystemExit(f"OPD teacher config not found: {config_path}")
architecture = json.loads(config_path.read_text())["architectures"][0]

transformers_version = "unknown"
problems = []
try:
    import transformers
    from transformers import AutoConfig

    transformers_version = transformers.__version__
    AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
except Exception as exc:
    problems.append(
        f"OPD teacher {architecture} is unsupported by Transformers "
        f"{transformers_version}: {str(exc).splitlines()[0]}"
    )

supported = None
try:
    import vllm
    from vllm.model_executor.models.registry import ModelRegistry

    supported = architecture in ModelRegistry.get_supported_archs()
except Exception as exc:
    problems.append(f"Could not check vLLM teacher support: {exc}")
if supported is False:
    problems.append(
        f"OPD teacher {architecture} is unsupported by vLLM {vllm.__version__}. "
        "Use a runtime with this architecture before launching training."
    )
if problems:
    raise SystemExit("\n".join(problems))
PY
fi

exec bash "${SCRIPT_DIR}/train_opd_base.sh" "$@"
