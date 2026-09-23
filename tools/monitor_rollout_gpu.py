#!/usr/bin/env python3
"""Continuously record per-GPU NVML-visible memory and utilization to CSV."""

import argparse
import csv
import subprocess
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "gpu_index", "memory_used_mib", "memory_free_mib", "gpu_util_percent"]
    needs_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", newline="", buffering=1) as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        while True:
            now = time.time()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            for line in result.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) == 4:
                    writer.writerow(dict(zip(fields, [f"{now:.6f}", *values], strict=True)))
            time.sleep(max(args.interval, 0.1))


if __name__ == "__main__":
    main()
