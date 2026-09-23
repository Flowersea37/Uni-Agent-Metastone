#!/usr/bin/env python3
"""Build per-step rollout/update speed and NVML memory summaries."""

import argparse
import csv
import statistics
from pathlib import Path


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in (None, "") else None


def summarize_memory(samples, start, end):
    window = [sample for sample in samples if start <= sample["timestamp"] <= end]
    if not window:
        return None
    per_gpu_peak = {}
    for sample in window:
        gpu = int(sample["gpu_index"])
        per_gpu_peak[gpu] = max(per_gpu_peak.get(gpu, 0.0), sample["memory_used_mib"])
    return {
        "samples": len(window),
        "peak": max(per_gpu_peak.values()) / 1024,
        "mean_peak": sum(per_gpu_peak.values()) / len(per_gpu_peak) / 1024,
        "min_free": min(x["memory_free_mib"] for x in window) / 1024,
        "peak_util": max(x["gpu_util_percent"] for x in window),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("actor_steps", type=Path)
    parser.add_argument("gpu_samples", type=Path)
    parser.add_argument("--engine-memory", type=Path)
    parser.add_argument("--request-metrics", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.actor_steps.open(newline="") as source:
        steps = list(csv.DictReader(source))
    with args.gpu_samples.open(newline="") as source:
        samples = list(csv.DictReader(source))
    engine_rows = []
    if args.engine_memory and args.engine_memory.exists():
        with args.engine_memory.open(newline="") as source:
            engine_rows = list(csv.DictReader(source))
    request_rows = []
    if args.request_metrics and args.request_metrics.exists():
        with args.request_metrics.open(newline="") as source:
            request_rows = list(csv.DictReader(source))

    def request_summary(step, rollout_start, rollout_end):
        # Async warmup/in-flight requests can retain the preceding weight
        # version. Wall-clock overlap with this rollout is the reliable owner.
        rows = [
            r for r in request_rows
            if r.get("request_end") not in (None, "")
            and rollout_start <= float(r["request_end"]) <= rollout_end
        ]
        if not rows:
            rows = [r for r in request_rows if str(r.get("global_steps", "")) == str(step)]
        if not rows:
            return {}
        def vals(key):
            return [float(r[key]) for r in rows if r.get(key) not in (None, "")]
        output_tokens = sum(int(r.get("output_tokens") or 0) for r in rows)
        decode_tokens = sum(max(int(r.get("output_tokens") or 0) - 1, 0) for r in rows)
        e2e = sum(vals("server_e2e_s"))
        decode = sum(vals("decode_s"))
        ttft = vals("ttft_s")
        per_request = vals("decode_tokens_per_s")
        def union_seconds(intervals):
            merged = []
            for start, end in sorted(intervals):
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            return sum(end - start for start, end in merged)
        intervals = [(float(r["request_start"]), float(r["request_end"])) for r in rows
                     if r.get("request_start") and r.get("request_end")]
        per_replica = {}
        for r in rows:
            if r.get("request_start") and r.get("request_end"):
                per_replica.setdefault(r.get("replica_rank", ""), []).append(
                    (float(r["request_start"]), float(r["request_end"])))
        cluster_busy_s = union_seconds(intervals)
        gpu_busy_s = sum(union_seconds(items) for items in per_replica.values())
        return {
            "vllm_requests": len(rows),
            "vllm_output_tokens": output_tokens,
            "vllm_decode_tokens": decode_tokens,
            "vllm_server_e2e_s": e2e,
            "vllm_decode_s": decode,
            "vllm_cluster_busy_s": cluster_busy_s,
            "vllm_gpu_busy_s": gpu_busy_s,
            "vllm_server_e2e_token_s": output_tokens / e2e if e2e else None,
            "vllm_decode_token_s": decode_tokens / decode if decode else None,
            "vllm_median_request_decode_token_s": statistics.median(per_request) if per_request else None,
            "vllm_median_ttft_s": statistics.median(ttft) if ttft else None,
            "vllm_busy_cluster_token_s": output_tokens / cluster_busy_s if cluster_busy_s else None,
            "vllm_token_per_gpu_busy_s": output_tokens / gpu_busy_s if gpu_busy_s else None,
        }
    for sample in samples:
        for key in ("timestamp", "memory_used_mib", "memory_free_mib", "gpu_util_percent"):
            sample[key] = float(sample[key])

    output = []
    def engine_mean(key):
        values = [float(item[key]) for item in engine_rows if item.get(key) not in (None, "")]
        return sum(values) / len(values) if values else None

    bytes_per_gib = 1024**3
    weights_gb = engine_mean("model_weights_bytes")
    kv_gb = engine_mean("kv_cache_available_bytes")
    non_torch_gb = engine_mean("non_torch_memory_bytes")
    activation_gb = engine_mean("peak_activation_memory_bytes")
    cudagraph_gb = engine_mean("cudagraph_memory_estimate_bytes")
    for row in steps:
        end = number(row, "timestamp")
        step_s = number(row, "step_s")
        gen_s = number(row, "gen_s")
        update_s = number(row, "update_actor_s")
        if end is None or step_s is None or gen_s is None or update_s is None:
            continue
        start = end - step_s
        rollout_end = start + gen_s
        req = request_summary(row["step"], start, rollout_end)
        # update_actor is the final timed compute phase in _step_once(). The
        # CSV timestamp follows metric computation, so this is a close upper
        # bound; a 1-second NVML interval is the dominant timing uncertainty.
        update_end = end
        update_start = update_end - update_s
        rollout_memory = summarize_memory(samples, start, rollout_end)
        update_memory = summarize_memory(samples, update_start, update_end)
        if rollout_memory is None or update_memory is None:
            continue
        rollout_speed = number(row, "rollout_tokens_per_s")
        update_speed = number(row, "update_tokens_per_s")
        response_tokens = number(row, "response_tokens")
        update_tokens = number(row, "update_tokens")
        # Backward compatibility for CSVs produced before explicit token
        # counts were added. total_tokens is suitable for update only.
        if update_speed is None and number(row, "total_tokens") is not None and update_s > 0:
            update_tokens = number(row, "total_tokens")
            update_speed = update_tokens / update_s
        output.append(
            {
                "step": row["step"],
                "precision": row.get("precision", ""),
                "rollout_quantization": row.get("rollout_quantization", "none"),
                "update_tokens_per_s": "" if update_speed is None else f"{update_speed:.6f}",
                "rollout_tokens_per_s": "" if rollout_speed is None else f"{rollout_speed:.6f}",
                "vllm_requests": req.get("vllm_requests", ""),
                "vllm_output_tokens": req.get("vllm_output_tokens", ""),
                "vllm_decode_tokens": req.get("vllm_decode_tokens", ""),
                "vllm_server_e2e_s": "" if req.get("vllm_server_e2e_s") is None else f"{req['vllm_server_e2e_s']:.6f}",
                "vllm_decode_s": "" if req.get("vllm_decode_s") is None else f"{req['vllm_decode_s']:.6f}",
                "vllm_cluster_busy_s": "" if req.get("vllm_cluster_busy_s") is None else f"{req['vllm_cluster_busy_s']:.6f}",
                "vllm_gpu_busy_s": "" if req.get("vllm_gpu_busy_s") is None else f"{req['vllm_gpu_busy_s']:.6f}",
                "vllm_server_e2e_token_s": "" if req.get("vllm_server_e2e_token_s") is None else f"{req['vllm_server_e2e_token_s']:.6f}",
                "vllm_decode_token_s": "" if req.get("vllm_decode_token_s") is None else f"{req['vllm_decode_token_s']:.6f}",
                "vllm_median_request_decode_token_s": "" if req.get("vllm_median_request_decode_token_s") is None else f"{req['vllm_median_request_decode_token_s']:.6f}",
                "vllm_median_ttft_s": "" if req.get("vllm_median_ttft_s") is None else f"{req['vllm_median_ttft_s']:.9f}",
                "vllm_busy_cluster_token_s": "" if req.get("vllm_busy_cluster_token_s") is None else f"{req['vllm_busy_cluster_token_s']:.6f}",
                "vllm_token_per_gpu_busy_s": "" if req.get("vllm_token_per_gpu_busy_s") is None else f"{req['vllm_token_per_gpu_busy_s']:.6f}",
                "rollout_peak_gpu_used_gb": f"{rollout_memory['peak']:.6f}",
                "update_peak_gpu_used_gb": f"{update_memory['peak']:.6f}",
                "vllm_model_weights_gb_per_gpu": "" if weights_gb is None else f"{weights_gb / bytes_per_gib:.6f}",
                "vllm_kv_cache_capacity_gb_per_gpu": "" if kv_gb is None else f"{kv_gb / bytes_per_gib:.6f}",
                "vllm_kv_cache_blocks": "" if engine_mean("kv_cache_blocks") is None else int(engine_mean("kv_cache_blocks")),
                "vllm_kv_cache_token_capacity": "" if engine_mean("kv_cache_size_tokens") is None else int(engine_mean("kv_cache_size_tokens")),
                "vllm_kv_cache_max_concurrency": "" if engine_mean("kv_cache_max_concurrency") is None else f"{engine_mean('kv_cache_max_concurrency'):.6f}",
                "vllm_non_torch_memory_gb_per_gpu": "" if non_torch_gb is None else f"{non_torch_gb / bytes_per_gib:.6f}",
                "vllm_peak_activation_gb_per_gpu": "" if activation_gb is None else f"{activation_gb / bytes_per_gib:.6f}",
                "vllm_cudagraph_estimate_gb_per_gpu": "" if cudagraph_gb is None else f"{cudagraph_gb / bytes_per_gib:.6f}",
                "rollout_start": f"{start:.6f}",
                "rollout_end": f"{rollout_end:.6f}",
                "gen_s": f"{gen_s:.6f}",
                "response_tokens": "" if response_tokens is None else int(response_tokens),
                "rollout_samples": rollout_memory["samples"],
                "rollout_mean_gpu_peak_used_gb": f"{rollout_memory['mean_peak']:.6f}",
                "rollout_min_gpu_free_gb": f"{rollout_memory['min_free']:.6f}",
                "rollout_peak_gpu_util_percent": f"{rollout_memory['peak_util']:.2f}",
                "update_start": f"{update_start:.6f}",
                "update_end": f"{update_end:.6f}",
                "update_actor_s": f"{update_s:.6f}",
                "update_tokens": "" if update_tokens is None else int(update_tokens),
                "update_samples": update_memory["samples"],
                "update_mean_gpu_peak_used_gb": f"{update_memory['mean_peak']:.6f}",
                "update_min_gpu_free_gb": f"{update_memory['min_free']:.6f}",
                "update_peak_gpu_util_percent": f"{update_memory['peak_util']:.2f}",
            }
        )
    if not output:
        raise SystemExit("No GPU samples overlapped completed rollout windows.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    print(f"Rollout summary: {args.output}")


if __name__ == "__main__":
    main()
