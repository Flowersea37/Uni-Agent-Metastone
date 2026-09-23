#!/usr/bin/env bash
# SWE-bench agent rollout + on-policy distillation | Qwen3.5 | vLLM | FSDP2
#
# Run from any directory. The defaults use 4 GPUs for the student/rollout and
# 4 dedicated GPUs for the teacher on one 8-GPU node.

set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Models and SWE-agent task.
STUDENT_MODEL=${STUDENT_MODEL:-/data/xgq/models/Qwen/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.5-35B-A3B}
TRAIN_FILE=${TRAIN_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}
VAL_FILE=${VAL_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}
TASK_CONFIG=${TASK_CONFIG:-/data/xgq/projects/RLs/uni-agent/examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml}

# Ray resources. Teacher GPUs are a dedicated resource pool, in addition to
# trainer.nnodes * trainer.n_gpus_per_node.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TEACHER_NNODES=${TEACHER_NNODES:-1}
TEACHER_GPUS_PER_NODE=${TEACHER_GPUS_PER_NODE:-4}

# OPD.
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
USE_POLICY_GRADIENT=${USE_POLICY_GRADIENT:-True}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-64}
TEACHER_TP=${TEACHER_TP:-4}
TEACHER_EP=${TEACHER_EP:-4}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.4}

# Agent-framework rollout (same task path as train_qwen3p5_dense_multi_node.sh).
TOOL_PARSER=${TOOL_PARSER:-qwen3_coder}
GATEWAY_COUNT=${GATEWAY_COUNT:-8}
CONCURRENCY=${CONCURRENCY:-256}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${STUDENT_MODEL}")"}
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-False}
ROLLOUT_NAME=${ROLLOUT_NAME:-vllm}
ROLLOUT_MODE=${ROLLOUT_MODE:-async}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-$((NNODES * NGPUS_PER_NODE))}

# Batch and sequence lengths follow the SWE-agent recipe. Lower these first if
# the student or teacher KV cache cannot accommodate the full 136K context.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((1024 * 128))}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-swe-agent-opd-fsdp}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"$(date +%Y%m%d%H%M)_exp"}
RUNTIME_DIR=${RUNTIME_DIR:-/data/xgq/projects/RLs/uni-agent/train_logs}
CKPTS_DIR=${CKPTS_DIR:-${RUNTIME_DIR}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-${RUNTIME_DIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# The teacher scores one token beyond the complete student sequence.
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

ray stop --force

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_LOCAL_GPUS=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)
export RAY_DEBUG=${RAY_DEBUG:-legacy}

ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus="${NUM_LOCAL_GPUS}" \
    --ray-debugger-external

# For multiple nodes, start/join child nodes before launching and ensure the
# cluster has trainer GPUs plus the dedicated teacher GPUs.
REQUIRED_GPUS=$((NNODES * NGPUS_PER_NODE + TEACHER_NNODES * TEACHER_GPUS_PER_NODE))
AVAILABLE_GPUS=$(python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)
if (( AVAILABLE_GPUS < REQUIRED_GPUS )); then
    echo "Ray has ${AVAILABLE_GPUS} GPUs, but OPD requires ${REQUIRED_GPUS} " \
         "(${NNODES}x${NGPUS_PER_NODE} student + ${TEACHER_NNODES}x${TEACHER_GPUS_PER_NODE} teacher)." >&2
    exit 1
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.prompt_key=prompt
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.return_raw_chat=True
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=False
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=False
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME}
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE}
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.prompt_length=${MAX_PROMPT_LENGTH}
    actor_rollout_ref.rollout.response_length=${MAX_RESPONSE_LENGTH}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER}
    actor_rollout_ref.rollout.agent.num_workers=${ROLLOUT_WORKERS}
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.opd_entry.OPDAgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME}
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
)

REWARD=(
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.resume_mode=auto
    trainer.default_local_dir=${CKPTS_DIR}
)

OPD=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_GPUS_PER_NODE}
    distillation.nnodes=${TEACHER_NNODES}
    distillation.teacher_models.teacher_model.model_path=${TEACHER_MODEL}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${TEACHER_TP}
    distillation.teacher_models.teacher_model.inference.expert_parallel_size=${TEACHER_EP}
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEM_UTIL}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}
    distillation.distillation_loss.loss_mode=${DISTILLATION_LOSS_MODE}
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=${USE_POLICY_GRADIENT}
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

mkdir -p "${AGENT_LOG_DIR}" "${CKPTS_DIR}" logs
START_TIME=$(date +%Y%m%d_%H%M%S)
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${OPD[@]}" \
    "$@" 2>&1 | tee "logs/qwen3.5-9b-swe-agent-opd-fsdp-${START_TIME}.log"
