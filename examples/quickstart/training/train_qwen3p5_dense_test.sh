#!/usr/bin/env bash
set -xeuo pipefail
# export NCCL_SOCKET_IFNAME=bond3
# export NCCL_DEBUG=INFO
MODEL_NAME=${MODEL_NAME:-"Qwen3.5-9B"}
# MODEL_NAME=${MODEL_NAME:-"Qwen3-8B"}
RUNTIME_DIR="${RUNTIME_DIR:-/data/xgq/projects/RLs/uni-agent/train_logs}"

project_name=${PROJECT_NAME:-"Uni-Agent-${MODEL_NAME}-megatron"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M)_exp"}

MODEL_PATH=${MODEL_PATH:-"/data/xgq/models/Qwen/${MODEL_NAME}"}
TRAIN_FILE=${TRAIN_FILE:-"/data/xgq/data/swe_agent/swe_bench_verified.parquet"}
TEST_FILE=${TEST_FILE:-"/data/xgq/data/swe_agent/swe_bench_verified.parquet"}

RUNTIME_ENV=${RUNTIME_ENV:-"/data/xgq/projects/RLs/uni-agent/examples/quickstart/training/runtime_env_local.yaml"}
CKPTS_DIR=${CKPTS_DIR:-"${RUNTIME_DIR}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RUNTIME_DIR}/logs/${project_name}/${exp_name}"}
GPU_MEMORY_CSV=${GPU_MEMORY_CSV:-"${RUNTIME_DIR}/gpu_memory/${project_name}/${exp_name}/gpu_memory_by_step.csv"}
ROLLOUT_GPU_MEMORY_CSV=${ROLLOUT_GPU_MEMORY_CSV:-"$(dirname "${GPU_MEMORY_CSV}")/rollout_gpu_samples.csv"}
ROLLOUT_ENGINE_MEMORY_CSV=${ROLLOUT_ENGINE_MEMORY_CSV:-"$(dirname "${GPU_MEMORY_CSV}")/rollout_engine_memory.csv"}
ROLLOUT_REQUEST_METRICS_CSV=${ROLLOUT_REQUEST_METRICS_CSV:-"$(dirname "${GPU_MEMORY_CSV}")/vllm_request_metrics.csv"}
# Must be launched from the repository root so Ray packages both `verl/` and `uni_agent/`.
# --- Agent-framework rollout (replaces the swe_agent agent-loop) --------------
# Run-wide task base (agent + sandbox + sampling), loaded from this YAML by
# uni_agent.framework.task_runner.run_task and deep-merged onto each row's task.
# Same file-path idea as the old agent_loop_config_path; new (task-config) schema.
TASK_CONFIG=${TASK_CONFIG:-"/data/xgq/projects/RLs/uni-agent/examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml"}
TOOL_PARSER=${TOOL_PARSER:-"qwen3_coder"}    # gateway tool-call parser; MUST match the model chat template
# GATEWAY_COUNT=${GATEWAY_COUNT:-8}            # gateway actors fronting the engine
# CONCURRENCY=${CONCURRENCY:-256}              # max in-flight rollout sessions (runner cap)
GATEWAY_COUNT=${GATEWAY_COUNT:-4}            # gateway actors fronting the engine
CONCURRENCY=${CONCURRENCY:-32}              # max in-flight rollout sessions (runner cap)
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}
MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-False}  # opt-in: zero the loss mask for unfinished episodes

# Optional actor-training quantization. Supported invocation styles:
#   quanti_pa=FP8 bash train_qwen3p5_dense_test.sh
#   quanti_pa=FP4 bash train_qwen3p5_dense_test.sh
#   bash train_qwen3p5_dense_test.sh FP8
# Empty means the original Megatron defaults (no explicit FP8/FP4 override).
quanti_pa=${quanti_pa:-${QUANTI_PA:-${1:-}}}
actor_quantization=${ACTOR_QUANTIZATION:-${quanti_pa}}
rollout_quantization=${ROLLOUT_QUANTIZATION:-${quanti_pa}}
quantization_args=()
ray_job_env_args=()
case "${actor_quantization^^}" in
    "")
        echo "Actor quantization: default"
        ;;
    FP8)
        quantization_args+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=hybrid
            +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=delayed
            +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_wgrad=False
            +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe=delayed
        )
        echo "Actor quantization: FP8 (hybrid, delayed)"
        ;;
    FP4)
        quantization_args+=(
            +actor_rollout_ref.actor.megatron.override_transformer_config.fp4=e2m1
            +actor_rollout_ref.actor.megatron.override_transformer_config.fp4_recipe=nvfp4
        )
        echo "Quantization: FP4 (e2m1, nvfp4)"
        ;;
    *)
        echo "Unsupported quanti_pa='${quanti_pa}'. Expected FP8, FP4, or empty." >&2
        exit 2
        ;;
esac

case "${rollout_quantization^^}" in
    "") echo "Rollout quantization: default" ;;
    FP8)
        # Qwen3.5 block-FP8 on Blackwell uses the compatible CUTLASS path.
        export VLLM_USE_DEEP_GEMM=0
        ray_job_env_args+=(VLLM_USE_DEEP_GEMM=0)
        quantization_args+=(+actor_rollout_ref.rollout.quantization=fp8)
        echo "Rollout quantization: FP8"
        ;;
    *) echo "Unsupported ROLLOUT_QUANTIZATION='${rollout_quantization}' (expected FP8 or empty)." >&2; exit 2 ;;
esac

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"} # sglang or vllm

# Algorithm parameters
adv_estimator=${ADV_ESTIMATOR:-grpo}

use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}

clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}
clip_ratio_c=${CLIP_RATIO_C:-10.0}

# Response length parameters
max_prompt_length=${MAX_PROMPT_LENGTH:-$((1024 * 8))}
max_response_length=${MAX_RESPONSE_LENGTH:-$((1024 * 128))}
enable_overlong_buffer=${ENABLE_OVERLONG_BUFFER:-False}
overlong_buffer_len=${OVERLONG_BUFFER_LEN:-$((1024 * 4))}  # unused
overlong_penalty_factor=${OVERLONG_PENALTY_FACTOR:-1.0}

loss_agg_mode=${LOSS_AGG_MODE:-"token-mean"}
loss_mode=${LOSS_MODE:-vanilla}

# Algorithm
temperature=${TEMPERATURE:-1.0}
do_sample=${DO_SAMPLE:-True}
top_p=${TOP_P:-1.0}
top_k=${TOP_K:--1}
full_determinism=${FULL_DETERMINISM:-False}
data_shuffle=${DATA_SHUFFLE:-True}
data_seed=${DATA_SEED:-null}
rollout_seed=${ROLLOUT_SEED:-42}
val_temperature=${VAL_TEMPERATURE:-1.0}
val_do_sample=${VAL_DO_SAMPLE:-True}
val_top_p=${VAL_TOP_P:-0.95}
val_top_k=${VAL_TOP_K:--1}

# Performance Related Parameter
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-True}
# gen_tp=${GEN_TP:-2}
# train_tp=${TP:-4}
# train_pp=${PP:-1}
gen_tp=${GEN_TP:-1}
train_cp=${CP:-1}
train_tp=${TP:-8}
train_pp=${PP:-1}
if [[ -n ${SEQUENCE_PARALLEL:-} ]]; then
    sequence_parallel=${SEQUENCE_PARALLEL}
elif [[ ${train_tp} -gt 1 ]]; then
    sequence_parallel=True
else
    sequence_parallel=False
fi
if [[ -n ${USE_REMOVE_PADDING:-} ]]; then
    use_remove_padding=${USE_REMOVE_PADDING}
else
    use_remove_padding=True
fi
# Megatron's fused forward requires the remove-padding representation.  Merely
# falling back inside one forward call is not sufficient because model startup
# has already monkey-patched GPTModel.forward.  Keep both switches consistent,
# while allowing an explicit override for standalone experiments.
use_fused_kernels=${USE_FUSED_KERNELS:-${use_remove_padding}}
# train_cp=${CP:-1}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=${OFFLOAD_FRACTION:-1.0}

# install mbridge
# pip3 install git+https://github.com/ISEEKYAN/mbridge
USE_MBRIDGE=${USE_MBRIDGE:-True}
USE_DIST_CKPT=${USE_DIST_CKPT:-False}

# ============================================================================
# V1 trainer topology
#   colocate_async: trainer and rollout share the trainer GPU pool.
#   separate_async: trainer and standalone rollout use different GPU pools.
# ============================================================================
trainer_mode=${TRAINER_MODE:-"separate_async"}

# Trainer resource pool.
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}
TRAINER_GPUS=$((NNODES * NGPUS_PER_NODE))

# Standalone rollout resource pool used by separate_async.
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-4}
ROLLOUT_GPUS=$((ROLLOUT_NNODES * ROLLOUT_NGPUS_PER_NODE))
ROLLOUT_CHECKPOINT_BACKEND=${ROLLOUT_CHECKPOINT_BACKEND:-"nccl"}
ROLLOUT_CHECKPOINT_MODULE=${ROLLOUT_CHECKPOINT_MODULE:-}
if [[ -z ${ROLLOUT_CHECKPOINT_MODULE} && ${ROLLOUT_CHECKPOINT_BACKEND} == "nccl" ]]; then
    # Import explicitly on every Ray worker. This registers the backend and,
    # unlike verl.checkpoint_engine's optional package import, preserves useful
    # dependency errors (for example, a missing CuPy installation).
    ROLLOUT_CHECKPOINT_MODULE="verl.checkpoint_engine.nccl_checkpoint_engine"
fi

# Uni-Agent rollout workers. Keep this independent from trainer GPU count.
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-${ROLLOUT_GPUS}}

train_prompt_bsz=${TRAIN_PROMPT_BSZ:-1}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-8}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-1}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
parameter_sync_step=${PARAMETER_SYNC_STEP:-1}
lr_decay_steps=${LR_DECAY_STEPS:-2000}
test_freq=${TEST_FREQ:-10}
save_freq=${SAVE_FREQ:-10}
total_training_steps=${TOTAL_TRAINING_STEPS:-null}

trainer_mode_args=()
case "${trainer_mode}" in
    colocate_async)
        REQUIRED_GPUS=${TRAINER_GPUS}
        trainer_mode_args+=(
            trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches}
        )
        ;;
    separate_async)
        REQUIRED_GPUS=$((TRAINER_GPUS + ROLLOUT_GPUS))

        # Official V1 separate_async invariant:
        # data.train_batch_size = parameter_sync_step * actor.ppo_mini_batch_size
        expected_train_bsz=$((parameter_sync_step * train_prompt_mini_bsz))
        if [[ ${train_prompt_bsz} -ne ${expected_train_bsz} ]]; then
            echo "Invalid separate_async batch configuration:" >&2
            echo "  TRAIN_PROMPT_BSZ=${train_prompt_bsz}" >&2
            echo "  PARAMETER_SYNC_STEP=${parameter_sync_step}" >&2
            echo "  PPO_MINI_BATCH_SIZE=${train_prompt_mini_bsz}" >&2
            echo "Expected TRAIN_PROMPT_BSZ = PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE = ${expected_train_bsz}" >&2
            exit 2
        fi

        trainer_mode_args+=(
            trainer.v1.separate_async.num_warmup_batches=${num_warmup_batches}
            trainer.v1.separate_async.parameter_sync_step=${parameter_sync_step}
            actor_rollout_ref.rollout.nnodes=${ROLLOUT_NNODES}
            actor_rollout_ref.rollout.n_gpus_per_node=${ROLLOUT_NGPUS_PER_NODE}
            actor_rollout_ref.rollout.checkpoint_engine.backend=${ROLLOUT_CHECKPOINT_BACKEND}
            actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module=${ROLLOUT_CHECKPOINT_MODULE}
        )
        ;;
    *)
        echo "Unsupported TRAINER_MODE='${trainer_mode}'. Expected colocate_async or separate_async." >&2
        exit 2
        ;;
esac

# ============================================================================
# Rollout correction is disabled by default for the standard GRPO + PPO
# baseline. Override these variables to enable behavior-anchor or decoupled
# rollout-correction experiments.
# ============================================================================
bypass_mode=${BYPASS_MODE:-False}                                # True => old_log_prob = rollout_log_prob
bypass_loss_type=${BYPASS_LOSS_TYPE:-ppo_clip}                   # ppo_clip | reinforce
rollout_is=${ROLLOUT_IS:-null}                                   # PPO clip already applies the IS ratio
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}                # single float => TIS upper clamp; "lo_hi" string => IcePop
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-False}  # normalize IS weights to mean=1.0 within a batch
rollout_rs=${ROLLOUT_RS:-null}                                   # no rejection sampling
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-null}

export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
# Keep the uploaded working-dir package alive while Ray creates the per-job
# runtime environment. Large repositories can outlive Ray's 600s temporary
# reference window, especially immediately after a cluster restart.
export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=${RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S:-3600}
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
# export NCCL_SOCKET_IFNAME=bond3
# unset NCCL_IGNORE_NET_MISMATCH

ray stop --force
nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
  | sort -u \
  | xargs -r kill -9
  
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus=$NUM_GPUS

  
while true; do
    GPU_COUNT=$(python - <<'PY'
import ray
ray.init(address="auto", ignore_reinit_error=True)
print(int(ray.cluster_resources().get("GPU", 0)))
ray.shutdown()
PY
)

    echo "Current Ray GPUs: ${GPU_COUNT}/${REQUIRED_GPUS}"

    if [ "$GPU_COUNT" -ge "$REQUIRED_GPUS" ]; then
        echo "All GPUs ready."
        break
    fi

    sleep 5
done




ray status

# Raw NVML sampling covers vLLM rollout processes, whose allocations are not
# visible to the actor worker's PyTorch allocator metrics.
python3 "${PWD}/tools/monitor_rollout_gpu.py" --output "${ROLLOUT_GPU_MEMORY_CSV}" --interval 1 &
ROLLOUT_MONITOR_PID=$!
cleanup_rollout_monitor() {
    kill "${ROLLOUT_MONITOR_PID}" 2>/dev/null || true
    wait "${ROLLOUT_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_rollout_monitor EXIT

# ray job submit --no-wait --runtime-env $RUNTIME_ENV \
ray job submit --runtime-env $RUNTIME_ENV \
    -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 "${ray_job_env_args[@]}" VERL_GPU_MEMORY_LOG="${GPU_MEMORY_CSV}" \
    python3 -m verl.trainer.main_ppo \
    --config-name=ppo_megatron_trainer \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=${trainer_mode} \
    "${trainer_mode_args[@]}" \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.shuffle=${data_shuffle} \
    data.seed=${data_seed} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.seed=${rollout_seed} \
    ++actor_rollout_ref.rollout.engine_memory_log="${ROLLOUT_ENGINE_MEMORY_CSV}" \
    ++actor_rollout_ref.rollout.request_metrics_log="${ROLLOUT_REQUEST_METRICS_CSV}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=${clip_ratio_c} \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=${use_fused_kernels} \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.sequence_parallel=${sequence_parallel} \
    "${quantization_args[@]}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=False \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    algorithm.rollout_correction.bypass_mode=${bypass_mode} \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    algorithm.rollout_correction.loss_type=${bypass_loss_type} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.bypass_mode=${bypass_mode} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is=${rollout_is} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs=${rollout_rs} \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs_threshold="${rollout_rs_threshold}" \
    ++actor_rollout_ref.actor.policy_loss.rollout_correction.loss_type=${bypass_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    +actor_rollout_ref.actor.checkpoint.save_contents=['model','hf_model'] \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=longest \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${SERVED_MODEL_NAME} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
    ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE} \
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.do_sample=${do_sample} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.full_determinism=${full_determinism} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${val_do_sample} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','swanlab'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    ++trainer.gpu_memory_log="${GPU_MEMORY_CSV}" \
    trainer.val_before_train=False \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${total_training_steps} \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.test_freq="${test_freq}"
