"""Refresh per-step network totals from monitor_net.py and phase CSV files."""

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path


def _integrate(points, start_ts, end_ts):
    tx_gbit = rx_gbit = 0.0
    for previous, current in zip(points, points[1:]):
        overlap = max(0.0, min(current[0], end_ts) - max(previous[0], start_ts))
        if overlap:
            tx_gbit += current[1] * overlap
            rx_gbit += current[2] * overlap
    return tx_gbit / 8, rx_gbit / 8


def refresh_network_summary(phase_csv: str, network_pattern: str, output_csv: str) -> None:
    """Atomically rewrite a summary containing every completed step."""
    with open(phase_csv, newline="") as source:
        phases = list(csv.DictReader(source))
    completed_steps = {int(row["step"]) for row in phases if row["phase"] == "step" and row["status"] == "ok"}
    phases = [row for row in phases if row["step"] and int(row["step"]) in completed_steps]
    if not phases:
        return
    for row in phases:
        row["step"] = int(row["step"])
        row["start_ts"] = float(row["start_ts"])
        row["end_ts"] = float(row["end_ts"])
        row["duration_s"] = float(row["duration_s"])

    samples = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    min_ts, max_ts = min(p["start_ts"] for p in phases), max(p["end_ts"] for p in phases)
    for path in glob.glob(network_pattern):
        with open(path, newline="") as source:
            for row in csv.DictReader(source):
                timestamp = float(row["timestamp"])
                if min_ts - 2 <= timestamp <= max_ts + 2:
                    key = (row["host"], row["type"])
                    samples[key][timestamp][0] += float(row["tx_gbps"])
                    samples[key][timestamp][1] += float(row["rx_gbps"])
    points_by_key = {key: sorted((ts, *rates) for ts, rates in values.items()) for key, values in samples.items()}

    totals = defaultdict(lambda: [0.0, 0.0])
    durations = defaultdict(float)
    occurrences = defaultdict(int)
    for phase in phases:
        phase_key = (phase["step"], phase["phase"])
        durations[phase_key] += phase["duration_s"]
        occurrences[phase_key] += 1
        for (_, net_type), points in points_by_key.items():
            tx_gb, rx_gb = _integrate(points, phase["start_ts"], phase["end_ts"])
            totals[(phase["step"], phase["phase"], net_type)][0] += tx_gb
            totals[(phase["step"], phase["phase"], net_type)][1] += rx_gb

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    fields = ["step", "phase", "network_type", "occurrences", "duration_s", "tx_GB", "rx_GB", "avg_tx_Gbps", "avg_rx_Gbps"]
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for (step, phase, net_type), (tx_gb, rx_gb) in sorted(totals.items()):
            duration = durations[(step, phase)]
            writer.writerow({
                "step": step, "phase": phase, "network_type": net_type,
                "occurrences": occurrences[(step, phase)], "duration_s": f"{duration:.3f}",
                "tx_GB": f"{tx_gb:.3f}", "rx_GB": f"{rx_gb:.3f}",
                "avg_tx_Gbps": f"{tx_gb * 8 / duration:.3f}" if duration else "0",
                "avg_rx_Gbps": f"{rx_gb * 8 / duration:.3f}" if duration else "0",
            })
    os.replace(temporary, output)
