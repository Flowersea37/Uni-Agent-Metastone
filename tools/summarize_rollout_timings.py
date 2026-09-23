#!/usr/bin/env python3
"""Summarize rollout and sandbox timing from readable trajectory JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    ("rollout_s", "Rollout total", "s"),
    ("sandbox_s", "Sandbox interaction", "s"),
    ("sandbox_ratio", "Sandbox / rollout", "%"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    json_output = args.json_output or root / "rollout_timing_summary.json"
    csv_output = args.csv_output or root / "rollout_timing_summary.csv"
    markdown_output = args.markdown_output or root / "rollout_timing_summary.md"

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(root.rglob("trajectory_readable.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            timing = document.get("metrics", {}).get("timing") or {}
            rollout_s = timing.get("gen_s")
            sandbox_s = timing.get("sandbox_interaction_s")
            if not isinstance(rollout_s, (int, float)) or not isinstance(sandbox_s, (int, float)):
                skipped.append({"path": str(path.relative_to(root)), "reason": "missing aggregate timing"})
                continue
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "session_id": document.get("session_id"),
                    "rollout_s": float(rollout_s),
                    "sandbox_s": float(sandbox_s),
                    "sandbox_ratio": float(sandbox_s) / float(rollout_s) * 100 if rollout_s > 0 else None,
                }
            )
        except Exception as exc:  # keep summaries available when one dump is malformed
            skipped.append({"path": str(path.relative_to(root)), "reason": f"{type(exc).__name__}: {exc}"})

    summary = {
        key: {**stats([row[key] for row in rows if row[key] is not None]), "unit": unit}
        for key, _, unit in METRICS
    }
    payload = {
        "root": str(root),
        "trajectory_count": len(rows),
        "skipped_count": len(skipped),
        "summary": summary,
        "trajectories": rows,
        "skipped": skipped,
    }
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "session_id", "rollout_s", "sandbox_s", "sandbox_ratio"))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Rollout timing summary",
        "",
        f"Parsed trajectories: {len(rows)}; skipped: {len(skipped)}.",
        "",
        "| Metric | Unit | Count | Min | Max | Mean | Median |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label, unit in METRICS:
        item = summary[key]
        lines.append(
            f"| {label} | {unit} | {item['count']} | {fmt(item['min'])} | {fmt(item['max'])} | "
            f"{fmt(item['mean'])} | {fmt(item['median'])} |"
        )
    if skipped:
        lines.extend(["", "## Skipped trajectories", "", "| Path | Reason |", "|---|---|"])
        lines.extend(f"| `{item['path']}` | {item['reason']} |" for item in skipped)
    markdown_output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(markdown_output)
    print(csv_output)
    print(json_output)


if __name__ == "__main__":
    main()
