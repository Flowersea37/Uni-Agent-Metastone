# 双机 Megatron OPD 训练交接指南

本文对应当前版本的 [`train_opd_multi_node_separate_megatron.sh`](examples/quickstart/training/opd/train_opd_multi_node_separate_megatron.sh)。请在 **head 机器的 `verl_test` 容器内**运行它。脚本依次调用 [`train_opd_multi_node_separate.sh`](examples/quickstart/training/opd/train_opd_multi_node_separate.sh)、[`start_opd_teacher.sh`](examples/quickstart/training/opd/start_opd_teacher.sh) 和 [`train_opd_base.sh`](examples/quickstart/training/opd/train_opd_base.sh)。下文的默认值按实际赋值和调用顺序整理；入口文件中关于“TP=2、CP=2”的注释已过时，实际默认值是 **TP=4、CP=1**。

## 1. 固定部署拓扑

| 机器 | GPU | 用途 | 管理方式 |
| --- | --- | --- | --- |
| head，默认 `174.1.59.5` | 0–3 | Megatron 学生模型训练、reference | Ray head，注册 4 张 GPU 和 `opd_trainer:1` |
| head | 4–7 | Qwen3.8 FP8 教师推理 | 独立的 vLLM Docker 容器，不进入 Ray GPU 资源池 |
| worker，默认 `174.1.59.4` | 0–7 | 学生模型 vLLM rollout | Ray worker，注册 8 张 GPU |

这套脚本固定为 **两台各至少 8 张 GPU、4 张训练 + 4 张教师 + 8 张 rollout**。`NNODES`、`NGPUS_PER_NODE`、`ROLLOUT_NNODES`、`ROLLOUT_NGPUS_PER_NODE` 和 GPU 编号会被中间脚本覆盖，不能仅靠环境变量改拓扑；不同 GPU 布局需先改脚本。Ray 资源总数应为 **12 GPU**，另外 4 张教师 GPU 由 Docker 使用。

## 2. 启动前准备

1. 两台机器均安装可用的 NVIDIA 驱动、Docker、NVIDIA Container Toolkit；每台至少 8 张可见 GPU。两台机器的 `/data` 必须挂载同一份项目、模型和数据，且**容器内路径一致**。代码至少包含本仓库和 `verl/` 子目录；两台使用兼容的代码版本。
2. 两台机器预先创建名为 `verl_test` 的训练容器，默认镜像示例为 `verl:vllm024.dev2-uniagent-runable`。参考 [`docker_start.sh`](docker_start.sh)：需要 `--gpus all --network host --ipc host --privileged --shm-size=64g`、`-v /data:/data` 和 `-v /var/run/docker.sock:/var/run/docker.sock`。head 容器需先启动；脚本会经 SSH 启动 worker 上**已有的** `verl_test` 容器，并不会创建它。Docker socket 同时供教师服务和 SWE-bench 沙箱使用。
3. head 宿主机预先拉取教师镜像 `vllm/vllm-openai:qwen38-flash-next`。教师容器以 `--network host`、`--gpus device=4,5,6,7`、只读 `/data` 挂载启动；默认服务端口为 `8008`。镜像版本应能加载指定的 Qwen3.8 FP8 模型。
4. 默认学生权重在 `/data/xgq/models/Qwen/Qwen3.5-9B`，教师权重在 `/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8`。两者应是完整的本地模型目录，供对应容器读取。教师 vLLM 默认 TP=4、EP 开启；学生 rollout 默认 TP=1，因此 worker 会创建 8 个单卡 rollout 副本。
5. 默认训练和验证数据都是 `/data/xgq/data/swe_agent/swe_bench_verified.parquet`。需要含 `prompt` 列，以及 `extra_info.tools_kwargs.task` 中的 SWE-bench 任务配置、沙箱镜像和元数据。可使用 `python -m uni_agent.tasks.swe_bench.preprocess --local-save-dir /data/xgq/data/swe_agent` 生成；数据处理代码见 [`preprocess.py`](uni_agent/tasks/swe_bench/preprocess.py)。任务 YAML 默认是 [`task_config_mini_swe_agent_blackbox.yaml`](examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml)。SWE-bench 沙箱镜像须可由 Docker 拉取或已缓存，相关网络和磁盘空间也要可用。
6. 两台训练容器须已经具备本仓库运行依赖，包括 PyTorch/CUDA、Ray、verl、vLLM、Megatron/MBridge、`transfer_queue`、CuPy、`mini-swe-agent` 及任务依赖。**启动脚本不会安装依赖**，也不会自动应用 `runtime_env_local.yaml`。若使用 NCCL checkpoint backend，尤其检查 worker 中的 CuPy 与其 CUDA 版本匹配。具体版本以可运行的容器镜像和锁定环境为准；不要只凭包名猜测 CUDA 版本。
7. head 容器能以无交互 SSH 登录 worker 宿主机（默认 `root@174.1.59.4`），远端账户能调用 `docker start` 和 `docker exec`。两台机器的 Ray、NCCL 与教师服务网络需互通；默认 `NCCL_SOCKET_IFNAME=bond3`，应改为实际网卡。Ray head 使用 `6379`，dashboard 使用 `8265`，教师使用 `8008`；Ray worker 和分布式训练还会用到动态端口，不能只开放这三个端口。

可先在**head 容器内**做只读检查（按实际 IP 和路径替换）：

```bash
cd /data/xgq/projects/RLs/uni-agent-metastone
nvidia-smi -L
test -f /data/xgq/models/Qwen/Qwen3.5-9B/config.json
test -f /data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8/config.json
test -f /data/xgq/data/swe_agent/swe_bench_verified.parquet
docker image inspect vllm/vllm-openai:qwen38-flash-next >/dev/null
docker info >/dev/null
python -c 'import ray, torch, vllm, cupy, transfer_queue; print("core imports OK")'
ssh -o BatchMode=yes root@174.1.59.4 'nvidia-smi -L; docker inspect verl_test >/dev/null'
```

还应在 worker 的 `verl_test` 容器中检查依赖，例如 `ssh root@174.1.59.4 'docker exec verl_test python -c "import ray, torch, vllm, cupy, transfer_queue"'`；若容器尚未启动，先由管理员启动再检查。

## 3. 入口参数

参数分两类：**环境变量**在命令前设置，由三个脚本依次读取；**Hydra override** 写在脚本名后，原样传给 `train_opd_base.sh`，位于配置参数列表末尾，同名键以最后的 override 为准。环境变量适合控制启动阶段（如 IP、模型、教师端口），Hydra override 适合调整训练配置；Hydra override 不会改变此前已经启动的教师容器或 Ray 集群。

| 环境变量 | 当前默认值 | 含义与约束 |
| --- | --- | --- |
| `HEAD_IP` / `WORKER_IP` | `174.1.59.5` / `174.1.59.4` | 两机可互通的 IPv4 地址；分别用于 Ray 和教师访问、worker 加入集群。 |
| `WORKER_SSH` / `TRAIN_CONTAINER` | `root@${WORKER_IP}` / `verl_test` | SSH 目标及远端已有训练容器名。 |
| `NCCL_SOCKET_IFNAME` | `bond3` | 两机共同使用的训练网络接口。 |
| `REPO_ROOT` | 脚本所在仓库根目录 | 两台容器内均须存在相同路径。 |
| `STUDENT_MODEL` / `TEACHER_MODEL` | `/data/xgq/models/Qwen/Qwen3.5-9B` / `/data/xgq/models/Qwen/Qwen3.8-Flash-Next-FP8` | 本地模型目录。 |
| `TRAIN_FILE` / `VAL_FILE` | `/data/xgq/data/swe_agent/swe_bench_verified.parquet` | Parquet 数据文件；默认同一份。 |
| `TASK_CONFIG` | `examples/quickstart/training/task_config_mini_swe_agent_blackbox.yaml` 的仓库绝对路径 | SWE-bench 沙箱和 agent 配置。 |
| `MEGATRON_TP` / `MEGATRON_CP` / `MEGATRON_PP` | `4` / `1` / `1` | 训练和 reference 的张量、上下文、流水线并行。三者必须为正整数，乘积整除 4。 |
| `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | `8192` / `122880` | prompt 和 response 的 token 上限；合计 `131072`。教师与 rollout 的 `max_model_len` 默认再加 1，即 `131073`。 |
| `PPO_MAX_TOKEN_LEN_PER_GPU` | `ceil((8192+122880)/MEGATRON_CP)`，默认 `131072` | actor/reference 的每 GPU token 上限；覆盖 CP 时应一起核对。 |
| `ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU` | `131072` | rollout log-prob 的独立 token 上限，默认不随 CP 缩减。 |
| `TRAIN_BATCH_SIZE` / `N_RESP_PER_PROMPT` | `1` / `8` | 每个训练 batch 的 prompt 数、每个 prompt 的 rollout 响应数。默认一个 prompt 产生 8 条响应。 |
| `PPO_MINI_BATCH_SIZE` / `PARAMETER_SYNC_STEP` | `1` / `1` | 异步训练须满足 `TRAIN_BATCH_SIZE = PPO_MINI_BATCH_SIZE × PARAMETER_SYNC_STEP`，否则入口报错。 |
| `NUM_WARMUP_BATCHES` | `1` | 分离式异步训练预热 batch 数。 |
| `ROLLOUT_TP` / `ROLLOUT_WORKERS` | `1` / `8` | rollout vLLM 的张量并行度、agent worker 数。改变 TP 会改变副本数。 |
| `ROLLOUT_GPU_MEM_UTIL` | `0.75` | 学生 vLLM 的 GPU 显存利用率目标；不是教师参数。 |
| `GATEWAY_COUNT` / `CONCURRENCY` | `4` / `32` | agent gateway 数量、最大并发会话数。 |
| `OPD_TEACHER_PORT` / `OPD_TEACHER_IMAGE` | `8008` / `vllm/vllm-openai:qwen38-flash-next` | 教师服务端口和镜像。教师 URL 自动设为 `http://${HEAD_IP}:${OPD_TEACHER_PORT}`。 |
| `OPD_TEACHER_CONTAINER` / `OPD_MAX_MODEL_LEN` | `opd_teacher_qwen38` / prompt+response+1 | 教师容器名、教师 vLLM 的上下文上限。若改后者，应与训练所需长度一致。 |
| `PROJECT_NAME` / `EXPERIMENT_NAME` | `Uni-Agent-Qwen3.5-9B-Qwen3.8-Flash-Next-FP8-OPD-megatron-separate-2node` / 时间戳加 `_exp` | 日志和 checkpoint 目录分组。 |
| `RUNTIME_DIR` / `CKPTS_DIR` / `AGENT_LOG_DIR` | 仓库 `train_logs` / 其下 `ckpts/<项目>/<实验>` / `logs/<项目>/<实验>` | 训练产物路径，可单独覆盖。 |
| `TOTAL_EPOCHS` / `TOTAL_TRAINING_STEPS` | `1` / `null` | 训练轮数、可选总步数上限。 |
| `SAVE_FREQ` / `TEST_FREQ` | `10` / `10` | 保存和验证频率。 |
| `ACTOR_LR` | `1e-6` | actor 学习率。 |

教师 Docker 启动命令中 `--tensor-parallel-size 4`、`--gpu-memory-utilization 0.80`、`--max-logprobs 64`、`--max-num-batched-tokens 4096` 是写死的。`train_opd_base.sh` 中的 `TEACHER_GPU_MEM_UTIL=0.4` 属于内部教师配置；本入口使用外部教师 Docker，它**不控制**该 Docker 的 `0.80`。`FSDP_SP_SIZE` 对此 Megatron 入口无效。默认任务 YAML 的 `agent.model.max_total_tokens=65536` 是 agent 自身预算，也不同于模型服务的 `131073` 上下文限制。

## 4. 启动示例

先在 head 宿主机启动已有的训练容器并进入其中，再执行入口：

```bash
docker start verl_test
docker exec -it verl_test bash
```

```bash
HEAD_IP=174.1.59.5 \
WORKER_IP=174.1.59.4 \
WORKER_SSH=root@174.1.59.4 \
NCCL_SOCKET_IFNAME=bond3 \
bash /data/xgq/projects/RLs/uni-agent-metastone/examples/quickstart/training/opd/train_opd_multi_node_separate_megatron.sh
```

例如要改成 TP=2、CP=2，并限制训练步数：

```bash
MEGATRON_TP=2 MEGATRON_CP=2 TOTAL_TRAINING_STEPS=2 \
bash /data/xgq/projects/RLs/uni-agent-metastone/examples/quickstart/training/opd/train_opd_multi_node_separate_megatron.sh
```

如需直接传 Hydra 参数，可在脚本后追加，例如 `trainer.logger='["console"]'`。请先核实所传键在当前 verl 配置中存在。实际入口还固定启用 `separate_async`、Megatron、`k1` 蒸馏损失、policy gradient、异步 vLLM rollout 和 SWE-bench agent；改 Hydra 参数并不保证与启动阶段的资源设置一致。

## 5. 启动影响、观察与排错

启动过程会：启动或复用 head 上的教师 Docker 容器；若已有教师的 `max_model_len` 或绑定地址不符，**强制删除并重建该容器**；在两台训练容器内执行 `ray stop --force`，重建 4+8 GPU 的 Ray 集群。确认两机没有其他依赖同一 Ray 实例或同名教师容器的任务后再运行。教师健康检查最多等待约 12 分钟，Ray GPU 资源最多等待约 5 分钟。不要同时启动另一份同名训练任务。

训练日志写入仓库 `logs/qwen3.5-9b-swe-agent-opd-megatron-<时间戳>.log`；agent 轨迹和 checkpoint 分别写入上表所述目录。默认 trainer logger 还包含 `swanlab`，若环境没有可用的 SwanLab 配置，可在命令末尾追加 `trainer.logger='["console"]'`。

```bash
# 在 head 训练容器中查看 Ray 和教师状态
ray status
curl -fsS http://174.1.59.5:8008/health
docker logs --tail 100 opd_teacher_qwen38
ls -lt /data/xgq/projects/RLs/uni-agent-metastone/logs/qwen3.5-9b-swe-agent-opd-megatron-*.log | head
```

常见故障定位：

- 教师健康检查失败：先看 `docker logs opd_teacher_qwen38`、GPU 4–7 是否空闲、模型路径、镜像和 `8008` 端口。
- worker SSH 或 Ray 加入失败：检查免密 SSH、远端 `verl_test`、两机代码路径和网卡；在 worker 容器内核对 Ray/CuPy 等依赖。`ray status` 应看到总计 12 GPU。
- 长序列 OOM：区分 head 的训练 GPU、worker 的 rollout GPU 和 head 的教师 GPU；分别调整 TP/CP、batch、并发、rollout 显存利用率或上下文长度。单独通过教师 128K 请求测试不能证明完整训练可运行。
- SWE-bench rollout 失败：检查 Parquet 的任务字段、沙箱镜像可达性、Docker socket、`mini-swe-agent` 和任务 YAML。代码层面的 `max_total_tokens=65536` 会提前限制 agent 轨迹。

本指南基于脚本静态核对；它不代表已在接收者的机器上验证容器依赖、模型兼容性或显存容量。交接时建议同时提供训练镜像的精确 digest/包清单及两台机器的硬件信息。
