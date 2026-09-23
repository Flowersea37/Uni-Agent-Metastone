#!/usr/bin/env python3
"""Generate a Markdown report for colocated vs separated async training."""

import argparse
import csv
import math
import statistics
from pathlib import Path


MODES = ("colocate_async", "separate_async")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return math.nan


def finite(values):
    return [value for value in values if math.isfinite(value)]


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(finite(values))
    if not values:
        return math.nan
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def fmt(value: float, digits: int = 2) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.{digits}f}"


def wall_seconds(run_dir: Path) -> float:
    try:
        return float((run_dir / "wall_end.txt").read_text()) - float((run_dir / "wall_start.txt").read_text())
    except (FileNotFoundError, ValueError):
        return math.nan


def summarize_run(run_dir: Path, warmup_steps: int) -> dict:
    steps = read_csv(run_dir / "gpu_memory_by_step.csv")
    measured = steps[warmup_steps:] if len(steps) > warmup_steps else steps
    step_s = finite([number(row, "step_s") for row in measured])
    gen_s = finite([number(row, "gen_s") for row in measured])
    update_s = finite([number(row, "update_actor_s") for row in measured])
    tokens = finite([number(row, "total_tokens") for row in measured])
    trajectories = len(list((run_dir / "trajectories").glob("step_*/session-*/trajectory.json")))
    exit_code = (run_dir / "exit_code.txt").read_text().strip() if (run_dir / "exit_code.txt").is_file() else "running/unknown"
    return {
        "steps": len(steps),
        "wall": wall_seconds(run_dir),
        "step_sum": sum(step_s),
        "step_mean": statistics.fmean(step_s) if step_s else math.nan,
        "step_median": statistics.median(step_s) if step_s else math.nan,
        "step_p95": percentile(step_s, 0.95),
        "gen_mean": statistics.fmean(gen_s) if gen_s else math.nan,
        "update_mean": statistics.fmean(update_s) if update_s else math.nan,
        "tokens_per_s": sum(tokens) / sum(step_s) if tokens and sum(step_s) else math.nan,
        "trajectories": trajectories,
        "exit_code": exit_code,
    }


def summarize_gpus(run_dir: Path, expected_gpus: int) -> list[dict]:
    rows = read_csv(run_dir / "gpu_samples.csv")
    by_gpu: dict[int, list[dict[str, str]]] = {index: [] for index in range(expected_gpus)}
    for row in rows:
        try:
            by_gpu.setdefault(int(row["gpu_index"]), []).append(row)
        except (KeyError, ValueError):
            pass
    summaries = []
    for index in sorted(by_gpu):
        gpu_rows = by_gpu[index]
        util = finite([number(row, "gpu_util_percent") for row in gpu_rows])
        memory = finite([number(row, "memory_used_mib") for row in gpu_rows])
        summaries.append({
            "index": index,
            "samples": len(util),
            "idle_zero": 100 * sum(value == 0 for value in util) / len(util) if util else math.nan,
            "idle_low": 100 * sum(value <= 5 for value in util) / len(util) if util else math.nan,
            "util_mean": statistics.fmean(util) if util else math.nan,
            "util_p95": percentile(util, 0.95),
            "memory_mean": statistics.fmean(memory) / 1024 if memory else math.nan,
            "memory_peak": max(memory) / 1024 if memory else math.nan,
        })
    return summaries


def speedup(baseline: float, candidate: float) -> float:
    return baseline / candidate if math.isfinite(baseline) and math.isfinite(candidate) and candidate else math.nan


def mean_gpu_metric(gpus: list[dict], key: str, indices: range) -> float:
    values = finite([gpu[key] for gpu in gpus if gpu["index"] in indices])
    return statistics.fmean(values) if values else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--expected-gpus", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    runs = {mode: summarize_run(args.result_root / mode, args.warmup_steps) for mode in MODES}
    gpus = {mode: summarize_gpus(args.result_root / mode, args.expected_gpus) for mode in MODES}
    colocate, separate = runs["colocate_async"], runs["separate_async"]
    wall_speedup = speedup(colocate["wall"], separate["wall"])
    step_speedup = speedup(colocate["step_mean"], separate["step_mean"])
    half = args.expected_gpus // 2
    low_indices = range(half)
    high_indices = range(half, args.expected_gpus)
    sep_low_util = mean_gpu_metric(gpus["separate_async"], "util_mean", low_indices)
    sep_high_util = mean_gpu_metric(gpus["separate_async"], "util_mean", high_indices)
    sep_low_idle = mean_gpu_metric(gpus["separate_async"], "idle_zero", low_indices)
    sep_high_idle = mean_gpu_metric(gpus["separate_async"], "idle_zero", high_indices)
    sep_low_mem = mean_gpu_metric(gpus["separate_async"], "memory_mean", low_indices)
    sep_high_mem = mean_gpu_metric(gpus["separate_async"], "memory_mean", high_indices)
    colocate_utils = finite([gpu["util_mean"] for gpu in gpus["colocate_async"]])
    colocate_util = statistics.fmean(colocate_utils) if colocate_utils else math.nan
    wall_delta = separate["wall"] - colocate["wall"]
    step_delta = separate["step_mean"] - colocate["step_mean"]

    lines = [
        "# Separate async vs colocate async",
        "",
        f"数据来源：colocate `{(args.result_root / 'colocate_async').resolve()}`；"
        f"separate `{(args.result_root / 'separate_async').resolve()}`。",
        "",
        "## 结论",
        "",
        f"- 分离模式端到端加速比：**{fmt(wall_speedup, 3)}x**（colocate wall time / separate wall time）。",
        f"- 分离模式平均 step 加速比：**{fmt(step_speedup)}x**（跳过前 {args.warmup_steps} 个 warmup step）。",
        "- `util=0%` 是严格空闲率，`util≤5%` 是更稳健的低负载率；两者均来自约 1 秒一次的 NVML 采样。",
        "",
        "## 训练时间与吞吐",
        "",
        "| 模式 | 状态码 | 完成 steps | Wall time (s) | Step 总时长 (s) | Step 平均 (s) | Step 中位数 (s) | Step P95 (s) | Gen 平均 (s) | Update 平均 (s) | Tokens/s | 轨迹数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        run = runs[mode]
        lines.append(
            f"| {mode} | {run['exit_code']} | {run['steps']} | {fmt(run['wall'])} | {fmt(run['step_sum'])} | "
            f"{fmt(run['step_mean'])} | {fmt(run['step_median'])} | {fmt(run['step_p95'])} | "
            f"{fmt(run['gen_mean'])} | {fmt(run['update_mean'])} | {fmt(run['tokens_per_s'])} | {run['trajectories']} |"
        )

    lines += [
        "", "## 现象与可能原因", "",
        f"- **整体耗时接近**：分离模式 wall time 相对 colocate 为 {wall_delta:+.2f} 秒，"
        f"平均 step 为 {step_delta:+.2f} 秒。两组来自独立运行，且轨迹数 "
        f"{colocate['trajectories']} vs {separate['trajectories']}，不能把微小差异直接归因于部署模式。",
        f"- **分离模式呈现前后两组 GPU 负载不均**：GPU 0–{half - 1} 平均利用率 "
        f"{fmt(sep_low_util)}%、严格空闲率 {fmt(sep_low_idle)}%、平均显存 {fmt(sep_low_mem)} GiB；"
        f"GPU {half}–{args.expected_gpus - 1} 分别为 {fmt(sep_high_util)}%、"
        f"{fmt(sep_high_idle)}%、{fmt(sep_high_mem)} GiB。colocate 全卡平均利用率为 {fmt(colocate_util)}%。"
        "后四卡只是相对更忙，严格空闲时间仍超过一半。"
        "较低显存组可能对应 rollout，较高显存组可能对应 trainer；仅凭 NVML 采样不能确认进程到物理卡的映射，需以 Ray actor/GPU 绑定记录核实。",
        f"- **生成与更新的负载不同**：分离模式平均 gen={fmt(separate['gen_mean'])} 秒、"
        f"update={fmt(separate['update_mean'])} 秒；colocate 分别为 {fmt(colocate['gen_mean'])} 秒和 "
        f"{fmt(colocate['update_mean'])} 秒。若实际配置为分离模式 trainer {half} 张卡、colocate 共享 "
        f"{args.expected_gpus} 张卡，更新耗时增加可能抵消 rollout 并行的收益。"
        "但异步 gen_s 是消费预取批次的等待/生成统计，不能直接当作纯推理计算时间。",
        "- **为什么出现低利用率**：SWE agent 在多轮推理之间还会执行工具、Docker 命令和奖励计算；"
        "GPU 利用率低不等于 rollout worker 没工作。预取队列供给不足、请求并发不足或尾部长轨迹也可能让模型间歇等待。"
        "这些是待验证解释，不是本报告已经证明的瓶颈。",
        "- **下一步验证**：将 `gpu_samples.csv` 按 step 时间窗拆分，并结合 "
        "`vllm_request_metrics.csv`、Ray actor 的 CUDA_VISIBLE_DEVICES、队列等待日志检查低利用率时段；"
        "若要判断是否稳定加速，至少重复多轮并比较相同 token 工作量。",
    ]

    lines += [
        "", "## 每张 GPU 的空闲率与负载对比", "",
        "单元格：colocate → separate（差值）。差值 = separate − colocate；利用率直接相减，以 % 显示，并非相对变化率。",
        "同一 GPU 编号在两种模式中可能承担不同角色，以下按物理 GPU 编号对齐。", "",
        "| GPU | 空闲率 % | 低负载率 % | 平均利用率 % | P95 利用率 % | 平均显存 GiB | 峰值显存 GiB |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    gpu_metrics = [
        ("idle_zero", "空闲率 util=0", "%", "%", 2),
        ("idle_low", "低负载率 util≤5", "%", "%", 2),
        ("util_mean", "平均利用率", "%", "%", 2),
        ("util_p95", "P95 利用率", "%", "%", 2),
        ("memory_mean", "平均显存", " GiB", " GiB", 2),
        ("memory_peak", "峰值显存", " GiB", " GiB", 2),
    ]
    indexed = {mode: {gpu["index"]: gpu for gpu in gpus[mode]} for mode in MODES}
    for index in sorted(set(indexed[MODES[0]]) | set(indexed[MODES[1]])):
        cells = [str(index)]
        for key, label, unit, delta_unit, digits in gpu_metrics:
            baseline = indexed[MODES[0]].get(index, {}).get(key, math.nan)
            candidate = indexed[MODES[1]].get(index, {}).get(key, math.nan)
            left = fmt(baseline, digits)
            right = fmt(candidate, digits)
            delta = candidate - baseline
            change = f"{delta:+.{digits}f}" if math.isfinite(delta) else "N/A"
            cells.append(f"{left} → {right} ({change}{delta_unit})")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    for mode in MODES:
        counts = [gpu["samples"] for gpu in gpus[mode]]
        sample_note = str(counts[0]) if len(set(counts)) == 1 else ", ".join(
            f"GPU {gpu['index']}={gpu['samples']}" for gpu in gpus[mode]
        )
        means = finite([gpu["util_mean"] for gpu in gpus[mode]])
        imbalance = statistics.pstdev(means) if len(means) > 1 else math.nan
        lines += [f"{mode}：逐卡采样数 {sample_note}；逐卡平均利用率标准差 {fmt(imbalance)}%。", ""]

    lines += [
        "## 口径与原始数据",
        "",
        f"- 本报告按 {args.expected_gpus} 张 GPU 展示；模型、batch、同步周期及实际 GPU 分配应以两组 train.log 为准，分析器不自动验证配置一致性。",
        "- colocate 模式让 trainer/rollout 共享 GPU；separate 模式使用独立资源池，具体分配以运行日志为准。",
        "- Wall time 包含 Ray 重启、runtime-env、模型初始化、rollout 和训练；step 指标不包含作业启动开销。",
        "- 异步模式中 `gen_s` 可能接近零，表示当前 step 消费的是预取轨迹，不代表生成没有成本；因此应优先比较 wall time 和 step time。",
        "- 每组原始文件：`train.log`、`gpu_memory_by_step.csv`、`gpu_samples.csv`、`rollout_engine_memory.csv`、`vllm_request_metrics.csv`、`trajectories/`。",
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
