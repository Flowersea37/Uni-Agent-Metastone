#!/usr/bin/env bash
# SWE-bench agent rollout + on-policy distillation | Qwen3.5 | vLLM
#
# Run from any directory. The defaults use 4 GPUs for the student/rollout and
# 4 dedicated GPUs for the teacher on one 8-GPU node.

set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-/data/xgq/projects/RLs/uni-agent-metastone}
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/verl:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Models and SWE-agent task.
STUDENT_MODEL=${STUDENT_MODEL:-/data/xgq/models/Qwen/Qwen3.5-4B}
TEACHER_MODEL=${TEACHER_MODEL:-/data/xgq/models/Qwen/Qwen3.5-35B-A3B}
TRAIN_FILE=${TRAIN_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}
VAL_FILE=${VAL_FILE:-/data/xgq/data/swe_agent/swe_bench_verified.parquet}
TASK_CONFIG=${TASK_CONFIG:-/data/xgq/projects/RLs/uni-agent-metastone/examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml}

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
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
OPD_BACKEND=${OPD_BACKEND:-fsdp2}
if [[ "${OPD_BACKEND}" == fsdp2 ]]; then
    FSDP_SP_SIZE=${FSDP_SP_SIZE:-1}
    if ! [[ "${FSDP_SP_SIZE}" =~ ^[1-9][0-9]*$ ]] || (( NGPUS_PER_NODE % FSDP_SP_SIZE != 0 )); then
        echo "FSDP_SP_SIZE must be a positive divisor of NGPUS_PER_NODE=${NGPUS_PER_NODE}" >&2
        exit 2
    fi
elif [[ "${OPD_BACKEND}" == megatron ]]; then
    MEGATRON_TP=${MEGATRON_TP:-2}
    MEGATRON_CP=${MEGATRON_CP:-2}
    MEGATRON_PP=${MEGATRON_PP:-1}
    for size in "${MEGATRON_TP}" "${MEGATRON_CP}" "${MEGATRON_PP}"; do
        if ! [[ "${size}" =~ ^[1-9][0-9]*$ ]]; then
            echo "Megatron TP, CP and PP must be positive integers" >&2
            exit 2
        fi
    done
    if (( (NNODES * NGPUS_PER_NODE) % (MEGATRON_TP * MEGATRON_CP * MEGATRON_PP) != 0 )); then
        echo "Megatron TP*CP*PP must divide the trainer GPU count" >&2
        exit 2
    fi
else
    echo "Unsupported OPD_BACKEND=${OPD_BACKEND}; expected fsdp2 or megatron" >&2
    exit 2
fi
ACTOR_LR=${ACTOR_LR:-1e-6}
USE_FUSED_KERNELS=${USE_FUSED_KERNELS:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-9B-swe-agent-opd-fsdp}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"$(date +%Y%m%d%H%M)_exp"}
RUNTIME_DIR=${RUNTIME_DIR:-/data/xgq/projects/RLs/uni-agent-metastone/train_logs}
CKPTS_DIR=${CKPTS_DIR:-${RUNTIME_DIR}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-${RUNTIME_DIR}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# The teacher scores one token beyond the complete student sequence.
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NUM_LOCAL_GPUS=$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)
export RAY_DEBUG=${RAY_DEBUG:-legacy}

# Multi-node launcher owns the Ray lifecycle. Keep the single-node entry's
# original start behavior when this flag is unset.
if [[ "${OPD_RAY_MANAGED_EXTERNALLY:-0}" != 1 ]]; then
    ray stop --force
    ray start --head \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --dashboard-port=8265 \
        --num-gpus="${NUM_LOCAL_GPUS}" \
        --ray-debugger-external
fi

# For multiple nodes, start/join child nodes before launching and ensure the
# cluster has trainer GPUs plus the dedicated teacher GPUs.
REQUIRED_GPUS=$((NNODES * NGPUS_PER_NODE))
if [[ "${OPD_TRAINER_MODE:-sync}" == separate_async ]]; then
    : "${ROLLOUT_NNODES:?ROLLOUT_NNODES is required for separate_async}"
    : "${ROLLOUT_NGPUS_PER_NODE:?ROLLOUT_NGPUS_PER_NODE is required for separate_async}"
    REQUIRED_GPUS=$((REQUIRED_GPUS + ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
fi
if [[ -z "${OPD_TEACHER_URL:-}" ]]; then
    REQUIRED_GPUS=$((REQUIRED_GPUS + TEACHER_NNODES * TEACHER_GPUS_PER_NODE))
fi
AVAILABLE_GPUS=$(python3 - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)
if (( AVAILABLE_GPUS < REQUIRED_GPUS )); then
    echo "Ray has ${AVAILABLE_GPUS} GPUs, but OPD requires ${REQUIRED_GPUS}." >&2
    exit 1
fi

V1_MODE=(trainer.use_v1=True trainer.v1.trainer_mode="${OPD_TRAINER_MODE:-sync}")
if [[ "${OPD_TRAINER_MODE:-sync}" == separate_async ]]; then
    : "${ROLLOUT_NNODES:?ROLLOUT_NNODES is required for separate_async}"
    : "${ROLLOUT_NGPUS_PER_NODE:?ROLLOUT_NGPUS_PER_NODE is required for separate_async}"
    : "${ROLLOUT_CHECKPOINT_BACKEND:=nccl}"
    : "${ROLLOUT_CHECKPOINT_MODULE:=verl.checkpoint_engine.nccl_checkpoint_engine}"
    : "${PARAMETER_SYNC_STEP:=1}"
    : "${NUM_WARMUP_BATCHES:=1}"
    if (( TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE )); then
        echo "separate_async needs TRAIN_BATCH_SIZE = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE" >&2
        exit 2
    fi
    V1_MODE+=(
        trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}"
        trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}"
        actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}"
        actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}"
        actor_rollout_ref.rollout.checkpoint_engine.backend="${ROLLOUT_CHECKPOINT_BACKEND}"
        actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module="${ROLLOUT_CHECKPOINT_MODULE}"
        transfer_queue.enable=True
    )
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
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.use_torch_compile=False
)

CONFIG_NAME=ppo_trainer
if [[ "${OPD_BACKEND}" == fsdp2 ]]; then
    ACTOR+=(
        actor_rollout_ref.actor.strategy=fsdp2
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=${FSDP_SP_SIZE}
        actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
        actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
        actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=False
        actor_rollout_ref.actor.fsdp_config.offload_policy=False
        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${FSDP_SP_SIZE}
    )
    REF+=(
        actor_rollout_ref.ref.strategy=fsdp2
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=${FSDP_SP_SIZE}
        actor_rollout_ref.ref.fsdp_config.offload_policy=False
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
        actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=False
        actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${FSDP_SP_SIZE}
    )
else
    CONFIG_NAME=ppo_megatron_trainer
    # OPD keeps one extra token for teacher scoring; vLLM requires the model's
    # configured position limit to cover rollout.max_model_len as well.
    MODEL+=(
        +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=${MAX_MODEL_LEN}
    )
    ACTOR+=(
        actor_rollout_ref.actor.strategy=megatron
        actor_rollout_ref.actor.clip_ratio_low=0.2
        actor_rollout_ref.actor.clip_ratio_high=0.28
        actor_rollout_ref.actor.clip_ratio_c=10.0
        actor_rollout_ref.actor.optim.lr_decay_style=constant
        actor_rollout_ref.actor.optim.lr_decay_steps=2000
        actor_rollout_ref.actor.optim.weight_decay=0.1
        actor_rollout_ref.actor.megatron.use_mbridge=True
        actor_rollout_ref.actor.megatron.vanilla_mbridge=True
        actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
        actor_rollout_ref.actor.megatron.param_offload=True
        actor_rollout_ref.actor.megatron.grad_offload=True
        actor_rollout_ref.actor.megatron.optimizer_offload=True
        actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${MEGATRON_TP}
        actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${MEGATRON_PP}
        actor_rollout_ref.actor.megatron.context_parallel_size=${MEGATRON_CP}
        actor_rollout_ref.actor.megatron.sequence_parallel=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
        +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
        +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False
        +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
        +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    )
    REF+=(
        actor_rollout_ref.ref.strategy=megatron
        actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
        actor_rollout_ref.ref.megatron.param_offload=True
        actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${MEGATRON_TP}
        actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${MEGATRON_PP}
        actor_rollout_ref.ref.megatron.context_parallel_size=${MEGATRON_CP}
    )
fi

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
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}
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
if [[ -n "${OPD_TEACHER_URL:-}" ]]; then
    OPD+=("++ray_kwargs.ray_init.runtime_env.env_vars.OPD_TEACHER_URL=${OPD_TEACHER_URL}")
fi
if [[ -n "${OPD_TRAINER_RAY_RESOURCE:-}" ]]; then
    OPD+=("++ray_kwargs.ray_init.runtime_env.env_vars.OPD_TRAINER_RAY_RESOURCE=${OPD_TRAINER_RAY_RESOURCE}")
fi

mkdir -p "${AGENT_LOG_DIR}" "${CKPTS_DIR}" logs
START_TIME=$(date +%Y%m%d_%H%M%S)
python3 -m verl.trainer.main_ppo --config-name="${CONFIG_NAME}" \
    "${V1_MODE[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${OPD[@]}" \
    "$@" 2>&1 | tee "logs/qwen3.5-9b-swe-agent-opd-${OPD_BACKEND}-${START_TIME}.log"
