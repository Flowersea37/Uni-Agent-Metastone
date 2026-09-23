"""Create a compact report for two fixed-batch Actor replay runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def number(row, name):
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        return 0.0


def audit_hashes(path: Path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line)["sha256"] for line in stream if line.strip()]


def summarize(csv_path: Path, skip: int):
    data = rows(csv_path)
    applied_skip = skip if len(data) > skip else 0
    selected = data[applied_skip:]
    tokens = sum(number(row, "update_tokens") for row in selected)
    seconds = sum(number(row, "update_actor_s") for row in selected)
    all_tokens = sum(number(row, "update_tokens") for row in data)
    all_seconds = sum(number(row, "update_actor_s") for row in data)
    # The per-step trainer CSV records the actor CUDA allocator high-water mark
    # under peak_allocated_gb.  It is reset around each update, so unlike a
    # whole-process nvidia-smi sample it isolates the update phase.
    peaks = [number(row, "peak_allocated_gb") for row in selected]
    return {
        "steps": len(selected),
        "skipped_steps": applied_skip,
        "all_tokens": all_tokens,
        "all_seconds": all_seconds,
        "all_throughput": all_tokens / all_seconds if all_seconds else 0.0,
        "tokens": tokens,
        "seconds": seconds,
        "throughput": tokens / seconds if seconds else 0.0,
        "peak": max(peaks, default=0.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--default-csv", type=Path, required=True)
    parser.add_argument("--fp8-csv", type=Path, required=True)
    parser.add_argument("--dump-audit", type=Path, required=True)
    parser.add_argument("--default-audit", type=Path, required=True)
    parser.add_argument("--fp8-audit", type=Path, required=True)
    parser.add_argument("--skip-first", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    default = summarize(args.default_csv, args.skip_first)
    fp8 = summarize(args.fp8_csv, args.skip_first)
    hashes_dump = audit_hashes(args.dump_audit)
    hashes_default = audit_hashes(args.default_audit)
    hashes_fp8 = audit_hashes(args.fp8_audit)
    identical = bool(hashes_dump and hashes_dump == hashes_default == hashes_fp8)
    distinct_batches = len(set(hashes_default))
    speedup = fp8["throughput"] / default["throughput"] if default["throughput"] else 0.0
    cold_speedup = fp8["all_throughput"] / default["all_throughput"] if default["all_throughput"] else 0.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.output_dir / "fixed_batch_summary.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["run", *default.keys(), "input_sha256"])
        writer.writeheader()
        writer.writerow({"run": "default", **default, "input_sha256": hashes_default[0]})
        writer.writerow({"run": "actor_fp8", **fp8, "input_sha256": hashes_fp8[0]})

    report = args.output_dir / "fixed_batch_report.md"
    change = (speedup - 1.0) * 100
    cold_change = (cold_speedup - 1.0) * 100
    report.write_text(
        "# Fixed-batch Actor FP8 对比\n\n"
        f"- dump / default / actor_fp8 的 SHA256 序列一致：{'是' if identical else '否'}\n"
        f"- 固定 batch 数 / 不同 SHA256 数：{len(hashes_default)} / {distinct_batches}\n"
        f"- default / actor_fp8 实际跳过预热 update：{default['skipped_steps']} / {fp8['skipped_steps']}。\n\n"
        "| 指标 | default | actor_fp8 | 变化 |\n|---|---:|---:|---:|\n"
        f"| 有效 update 数 | {default['steps']} | {fp8['steps']} | - |\n"
        f"| 全部 step 吞吐（含编译） | {default['all_throughput']:.2f} token/s | {fp8['all_throughput']:.2f} token/s | {cold_change:+.1f}% |\n"
        f"| Update token 总数 | {default['tokens']:.0f} | {fp8['tokens']:.0f} | - |\n"
        f"| Update 总时间 | {default['seconds']:.2f} s | {fp8['seconds']:.2f} s | - |\n"
        f"| Update 加权吞吐 | {default['throughput']:.2f} token/s | {fp8['throughput']:.2f} token/s | {change:+.1f}% |\n"
        f"| Update CUDA allocated 峰值/GPU | {default['peak']:.2f} GiB | {fp8['peak']:.2f} GiB | {fp8['peak']-default['peak']:+.2f} GiB |\n",
        encoding="utf-8",
    )
    if not identical:
        raise SystemExit("Fixed-batch audit failed: dump/default/FP8 batch sequences differ")
    print(report)


if __name__ == "__main__":
    main()
