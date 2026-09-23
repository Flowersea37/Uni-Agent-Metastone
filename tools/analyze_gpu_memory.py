#!/usr/bin/env python3
"""Summarize and plot one or more veRL per-step GPU-memory CSV files."""

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="gpu_memory_by_step.csv files")
    parser.add_argument("--output-dir", type=Path, default=Path("gpu_memory_analysis"))
    parser.add_argument("--skip-first", type=int, default=1, help="warm-up steps excluded from summary (default: 1)")
    return parser.parse_args()


def load_rows(paths):
    rows = []
    for path in paths:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                row["source"] = str(path)
                row["step"] = int(row["step"])
                for key in ("peak_allocated_gb", "peak_reserved_gb", "update_actor_s", "step_s", "total_tokens"):
                    row[key] = float(row[key]) if row.get(key) else None
                rows.append(row)
    if not rows:
        raise SystemExit("No data rows found.")
    return rows


def group_name(row):
    fp8 = row.get("fp8") or "off"
    fp8_recipe = row.get("fp8_recipe") or "none"
    fp4 = row.get("fp4") or "off"
    fp4_recipe = row.get("fp4_recipe") or "none"
    return (
        f"{row['experiment_name']} ({row['precision']}, "
        f"fp8={fp8}/{fp8_recipe}, fp4={fp4}/{fp4_recipe})"
    )


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def load_phase_rows(paths):
    """Load the richer rollout_gpu_by_step.csv next to each actor CSV."""
    runs = {}
    for actor_path in paths:
        phase_path = actor_path.with_name("rollout_gpu_by_step.csv")
        if not phase_path.exists():
            continue
        with phase_path.open(newline="") as source:
            phase_rows = list(csv.DictReader(source))
        if not phase_rows:
            continue
        label = actor_path.parent.name
        for row in phase_rows:
            row["step"] = int(row["step"])
            for key, value in list(row.items()):
                if key not in ("step", "precision", "rollout_quantization"):
                    try:
                        row[key] = float(value) if value != "" else None
                    except (TypeError, ValueError):
                        pass
        runs[label] = sorted(phase_rows, key=lambda row: row["step"])
    return runs


def analyzed_rows(runs, skip_first):
    return {name: (rows[skip_first:] or rows) for name, rows in runs.items()}


def metric_mean(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else None


def metric_median(rows, key):
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def weighted_rate(rows, numerator, denominator, fallback=None):
    pairs = [(row.get(numerator), row.get(denominator)) for row in rows]
    pairs = [(n, d) for n, d in pairs if n is not None and d not in (None, 0)]
    if pairs:
        return sum(n for n, _ in pairs) / sum(d for _, d in pairs)
    return metric_mean(rows, fallback) if fallback else None


def find_train_logs(path):
    for parent in path.resolve().parents:
        if parent.name == "train_logs":
            return parent
    return None


def load_trajectory_stats(paths, skip_first):
    """Load exact prompt identities and per-trajectory timing from session dumps."""
    result = {}
    instance_pattern = re.compile(r"instance_id=([^,)]+)")
    for actor_path in paths:
        with actor_path.open(newline="") as source:
            actor_rows = list(csv.DictReader(source))
        if not actor_rows:
            continue
        label = actor_path.parent.name
        train_logs = find_train_logs(actor_path)
        if train_logs is None:
            continue
        run_dir = train_logs / "logs" / actor_rows[0]["project_name"] / actor_rows[0]["experiment_name"]
        selected_steps = [int(row["step"]) for row in actor_rows][skip_first:]
        if not selected_steps:
            selected_steps = [int(row["step"]) for row in actor_rows]
        step_stats = {}
        all_timings, rewards = [], []
        for step in selected_steps:
            ids, timings = [], []
            for session_dir in (run_dir / f"step_{step}").glob("session-*"):
                task_log = session_dir / "task.log"
                if task_log.exists():
                    match = instance_pattern.search(task_log.read_text(encoding="utf-8", errors="ignore"))
                    if match:
                        ids.append(match.group(1))
                meta_path = session_dir / "trajectory.json"
                if not meta_path.exists():
                    continue
                try:
                    trajectories = json.loads(meta_path.read_text(encoding="utf-8")).get("trajectories", [])
                except (OSError, json.JSONDecodeError):
                    continue
                for trajectory in trajectories:
                    timing = (trajectory.get("reward_extra_info") or {}).get("timing")
                    if isinstance(timing, dict):
                        timings.append(timing)
                        all_timings.append(timing)
                    reward = trajectory.get("reward_score")
                    if isinstance(reward, (int, float)):
                        rewards.append(float(reward))
            step_stats[step] = {"prompt_signature": tuple(sorted(Counter(ids).items())), "timings": timings}
        def timing_mean(key):
            values = [float(item[key]) for item in all_timings if item.get(key) is not None]
            return statistics.fmean(values) if values else None
        gen_total = sum(float(item["gen_s"]) for item in all_timings if item.get("gen_s") is not None)
        model_total = sum(float(item["model_rollout_s"]) for item in all_timings if item.get("model_rollout_s") is not None)
        result[label] = {
            "steps": step_stats,
            "timing_count": len(all_timings),
            "model_rollout_mean_s": timing_mean("model_rollout_s"),
            "sandbox_mean_s": timing_mean("sandbox_interaction_s"),
            "reward_mean_s": timing_mean("reward_s"),
            "model_share_percent": model_total / gen_total * 100 if gen_total else None,
            "reward_mean": statistics.fmean(rewards) if rewards else None,
        }
    return result


def write_phase_report(runs, trajectory_stats, output_dir, skip_first):
    """Write a workload-validated report with engine, actor and end-to-end sections."""
    selected = analyzed_rows(runs, skip_first)
    if "default" not in selected:
        return None
    path = output_dir / "result_analysis.md"
    run_order = [name for name in ("default", "rollout_fp8", "actor_fp8", "both_fp8", "fp8") if name in selected]
    lines = [
        "# 量化实验对比结果（分层口径）",
        "",
        f"> 汇总跳过每组前 {skip_first} 个预热 step；若剩余为空，则使用该组已有全部 step。",
        "",
    ]
    def value_text(value, unit="", integer=False):
        if value is None:
            return "N/A"
        rendered = f"{value:,.0f}" if integer else f"{value:.2f}"
        return rendered + (f" {unit}" if unit else "")
    def comparison_row(label, values, unit="", integer=False):
        base = values.get("default")
        cells = []
        for name in run_order:
            value = values.get(name)
            text = value_text(value, unit, integer)
            if name != "default" and value is not None and base not in (None, 0):
                text += f" ({(value - base) / base * 100:+.1f}%)"
            cells.append(text)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["## 1. 实验有效性", "", "| 指标 | " + " | ".join(run_order) + " |", "|---|" + "---:|" * len(run_order)]
    default_signatures = {
        step: data["prompt_signature"] for step, data in trajectory_stats.get("default", {}).get("steps", {}).items()
    }
    for name in run_order:
        signatures = trajectory_stats.get(name, {}).get("steps", {})
        matched = bool(signatures) and all(default_signatures.get(step) == data["prompt_signature"] for step, data in signatures.items())
        trajectory_stats.setdefault(name, {})["prompt_match"] = matched
    lines.append("| Prompt 序列与 default 一致 | " + " | ".join("是" if trajectory_stats.get(n, {}).get("prompt_match") else "否/缺失" for n in run_order) + " |")
    comparison_row("有效 step 数", {n: len(selected[n]) for n in run_order}, integer=True)
    comparison_row("Update 输入 token 总数", {n: sum(r.get("update_tokens") or 0 for r in selected[n]) for n in run_order}, integer=True)
    comparison_row("vLLM 输出 token 总数", {n: sum(r.get("vllm_output_tokens") or 0 for r in selected[n]) for n in run_order}, integer=True)
    comparison_row("具有精确计时的轨迹数", {n: trajectory_stats.get(n, {}).get("timing_count") for n in run_order}, integer=True)

    lines += ["", "## 2. Rollout 引擎性能与显存", "", "| 指标 | " + " | ".join(run_order) + " |", "|---|" + "---:|" * len(run_order)]
    comparison_row("vLLM decode 吞吐（加权）", {n: weighted_rate(selected[n], "vllm_decode_tokens", "vllm_decode_s", "vllm_decode_token_s") for n in run_order}, "token/s")
    comparison_row("vLLM 每 GPU 忙碌吞吐（加权）", {n: weighted_rate(selected[n], "vllm_output_tokens", "vllm_gpu_busy_s", "vllm_token_per_gpu_busy_s") for n in run_order}, "token/GPU-s")
    comparison_row("请求 TTFT 中位数", {n: metric_median(selected[n], "vllm_median_ttft_s") for n in run_order}, "s")
    comparison_row("vLLM 权重显存/GPU", {n: metric_mean(selected[n], "vllm_model_weights_gb_per_gpu") for n in run_order}, "GiB")
    comparison_row("KV Cache 容量/GPU", {n: metric_mean(selected[n], "vllm_kv_cache_capacity_gb_per_gpu") for n in run_order}, "GiB")
    comparison_row("Rollout 最小空闲显存", {n: min(r["rollout_min_gpu_free_gb"] for r in selected[n] if r.get("rollout_min_gpu_free_gb") is not None) for n in run_order}, "GiB")

    lines += ["", "## 3. Actor Update 性能", "", "| 指标 | " + " | ".join(run_order) + " |", "|---|" + "---:|" * len(run_order)]
    comparison_row("Update 吞吐（加权）", {n: weighted_rate(selected[n], "update_tokens", "update_actor_s", "update_tokens_per_s") for n in run_order}, "token/s")
    comparison_row("每 step Update 时间中位数", {n: metric_median(selected[n], "update_actor_s") for n in run_order}, "s")
    comparison_row("Update 整卡峰值", {n: max(r["update_peak_gpu_used_gb"] for r in selected[n] if r.get("update_peak_gpu_used_gb") is not None) for n in run_order}, "GiB")

    lines += ["", "## 4. Agent 端到端时间分解（每轨迹平均）", "", "> 这部分受工具路径和沙盒速度影响，不代表纯模型吞吐。", "", "| 指标 | " + " | ".join(run_order) + " |", "|---|" + "---:|" * len(run_order)]
    comparison_row("模型 rollout 时间", {n: trajectory_stats.get(n, {}).get("model_rollout_mean_s") for n in run_order}, "s")
    comparison_row("沙盒交互时间", {n: trajectory_stats.get(n, {}).get("sandbox_mean_s") for n in run_order}, "s")
    comparison_row("模型时间占比", {n: trajectory_stats.get(n, {}).get("model_share_percent") for n in run_order}, "%")
    comparison_row("Reward/eval 时间", {n: trajectory_stats.get(n, {}).get("reward_mean_s") for n in run_order}, "s")
    comparison_row("平均 reward", {n: trajectory_stats.get(n, {}).get("reward_mean") for n in run_order})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def plot_phase_comparison(plt, runs, output_dir, skip_first):
    """Plot exactly four per-step training metrics as line charts."""
    if not runs:
        return None
    colors = {
        "default": "#4C78A8",
        "rollout_fp8": "#F58518",
        "actor_fp8": "#54A24B",
        "both_fp8": "#E45756",
        "fp8": "#E45756",
    }
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    panels = [
        (axes[0, 0], "update_tokens_per_s", "Update throughput", "Tokens/s"),
        (axes[0, 1], "vllm_decode_token_s", "vLLM decode throughput (tools excluded)", "Tokens/s"),
        (axes[1, 0], "gen_s", "Agent rollout wall time (includes tools)", "Seconds"),
        (axes[1, 1], "update_peak_gpu_used_gb", "Update whole-GPU peak", "GiB / GPU"),
    ]
    for ax, metric, title, ylabel in panels:
        for run, rows in sorted(runs.items()):
            ax.plot([r["step"] for r in rows], [r.get(metric) for r in rows],
                    marker="o", linewidth=2, color=colors.get(run), label=run)
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.suptitle("Default vs FP8: training metrics by step", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    path = output_dir / "training_metrics_by_step.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    capacity_panels = [
        (axes[0, 0], "vllm_model_weights_gb_per_gpu", "vLLM model weights", "GiB / GPU"),
        (axes[0, 1], "vllm_kv_cache_capacity_gb_per_gpu", "vLLM KV-cache capacity", "GiB / GPU"),
        (axes[1, 0], "rollout_min_gpu_free_gb", "Minimum free memory during rollout", "GiB / GPU"),
        (axes[1, 1], "update_peak_gpu_used_gb", "Update whole-GPU peak", "GiB / GPU"),
    ]
    for ax, metric, title, ylabel in capacity_panels:
        names, values, bar_colors = [], [], []
        for run, rows in sorted(runs.items()):
            value = next((r.get(metric) for r in rows if r.get(metric) is not None), None)
            if value is not None:
                names.append(run)
                values.append(value)
                bar_colors.append(colors.get(run))
        ax.bar(names, values, color=bar_colors)
        ax.set_title(title)
        ax.set_xlabel("Run (initialization-time constant)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.ticklabel_format(style="plain", axis="y", useOffset=False)
    fig.suptitle("Default vs FP8: static rollout-engine memory and capacity", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .96))
    capacity_path = output_dir / "rollout_engine_metrics_by_step.png"
    fig.savefig(capacity_path, dpi=180)
    plt.close(fig)
    return path, capacity_path


def main():
    args = parse_args()
    rows = load_rows(args.inputs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in rows:
        grouped[group_name(row)].append(row)

    summary_path = args.output_dir / "gpu_memory_summary.csv"
    summary_fields = [
        "run",
        "steps_total",
        "steps_analyzed",
        "peak_allocated_max_gb",
        "peak_allocated_mean_gb",
        "peak_allocated_median_gb",
        "peak_reserved_max_gb",
        "update_actor_mean_s",
        "update_actor_median_s",
        "tokens_per_update_second",
        "vs_baseline_saved_gb",
        "vs_baseline_saved_percent",
        "vs_baseline_update_speedup",
    ]
    summaries = []
    for name, group in sorted(grouped.items()):
        group.sort(key=lambda row: row["step"])
        analyzed = group[args.skip_first :]
        if not analyzed:
            analyzed = group
        allocated = [row["peak_allocated_gb"] for row in analyzed if row["peak_allocated_gb"] is not None]
        reserved = [row["peak_reserved_gb"] for row in analyzed if row["peak_reserved_gb"] is not None]
        update_s = mean(row["update_actor_s"] for row in analyzed)
        update_values = [row["update_actor_s"] for row in analyzed if row["update_actor_s"] is not None]
        total_tokens = sum(row["total_tokens"] for row in analyzed if row["total_tokens"] is not None)
        total_update_s = sum(row["update_actor_s"] for row in analyzed if row["update_actor_s"] is not None)
        summaries.append(
            {
                "run": name,
                "_precision": analyzed[0]["precision"],
                "_allocated_mean": mean(allocated),
                "_update_mean": update_s,
                "_update_throughput": total_tokens / total_update_s if total_update_s else None,
                "steps_total": len(group),
                "steps_analyzed": len(analyzed),
                "peak_allocated_max_gb": fmt(max(allocated) if allocated else None),
                "peak_allocated_mean_gb": fmt(mean(allocated)),
                "peak_allocated_median_gb": fmt(statistics.median(allocated) if allocated else None),
                "peak_reserved_max_gb": fmt(max(reserved) if reserved else None),
                "update_actor_mean_s": fmt(update_s),
                "update_actor_median_s": fmt(statistics.median(update_values) if update_values else None),
                "tokens_per_update_second": fmt(total_tokens / total_update_s if total_update_s else None),
            }
        )

    # Rollout-only FP8 keeps the actor precision at BF16, so actor precision
    # cannot distinguish it from the real default run.  Prefer the explicit
    # experiment label/path and retain the old precision fallback for legacy
    # two-run inputs whose filenames do not contain ``default``.
    default_baselines = [
        item
        for item in summaries
        if "default" in item["run"].split(" (", 1)[0].lower()
    ]
    if len(default_baselines) == 1:
        baseline = default_baselines[0]
    else:
        baselines = [item for item in summaries if item["_precision"] not in ("fp8", "fp4")]
        baseline = baselines[0] if len(baselines) == 1 else None
    # Keep baseline metrics separately: the output cleanup below removes the
    # private calculation fields from each summary dictionary.
    baseline_memory = baseline["_allocated_mean"] if baseline is not None else None
    baseline_update_throughput = baseline["_update_throughput"] if baseline is not None else None
    for item in summaries:
        saved = saved_percent = speedup = None
        if baseline is not None and item is not baseline:
            current_memory = item["_allocated_mean"]
            if baseline_memory is not None and current_memory is not None:
                saved = baseline_memory - current_memory
                saved_percent = saved / baseline_memory * 100 if baseline_memory else None
            if baseline_update_throughput is not None and item["_update_throughput"]:
                speedup = item["_update_throughput"] / baseline_update_throughput
        item["vs_baseline_saved_gb"] = fmt(saved)
        item["vs_baseline_saved_percent"] = fmt(saved_percent)
        item["vs_baseline_update_speedup"] = fmt(speedup)
        item.pop("_precision")
        item.pop("_allocated_mean")
        item.pop("_update_mean")
        item.pop("_update_throughput")

    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)

    phase_runs = load_phase_rows(args.inputs)
    trajectory_stats = load_trajectory_stats(args.inputs, args.skip_first)
    phase_report = write_phase_report(phase_runs, trajectory_stats, args.output_dir, args.skip_first)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"Summary: {summary_path}")
        if phase_report:
            print(f"Result analysis: {phase_report}")
        print("matplotlib is not installed; skipped plot generation.")
        return

    phase_plots = plot_phase_comparison(plt, phase_runs, args.output_dir, args.skip_first)
    print(f"Summary: {summary_path}")
    if phase_plots:
        print(f"Training metrics plot: {phase_plots[0]}")
        print(f"Rollout engine plot: {phase_plots[1]}")
    if phase_report:
        print(f"Result analysis: {phase_report}")


if __name__ == "__main__":
    main()
