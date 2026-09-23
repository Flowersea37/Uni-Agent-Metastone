#!/usr/bin/env python3
"""Correlate monitor_net.py CSV samples with veRL phase intervals."""

import argparse
import csv
import glob
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def expand_inputs(values):
    paths = []
    for value in values:
        for item in value.split(","):
            matches = glob.glob(item)
            paths.extend(matches or [item])
    return list(dict.fromkeys(paths))


def load_phases(path):
    with open(path, newline="") as source:
        rows = list(csv.DictReader(source))
    for row in rows:
        row["start_ts"] = float(row["start_ts"])
        row["end_ts"] = float(row["end_ts"])
        row["duration_s"] = float(row["duration_s"])
        row["step"] = int(row["step"]) if row["step"] else -1
    return rows


def load_network(paths, start_ts, end_ts):
    # One timestamp contains one row per device. Sum devices of the same host/type.
    samples = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    for path in paths:
        with open(path, newline="") as source:
            for row in csv.DictReader(source):
                ts = float(row["timestamp"])
                if start_ts - 2 <= ts <= end_ts + 2:
                    key = (row["host"], row["type"])
                    samples[key][ts][0] += float(row["tx_gbps"])
                    samples[key][ts][1] += float(row["rx_gbps"])
    return {key: sorted((ts, *rates) for ts, rates in points.items()) for key, points in samples.items()}


def integrate(points, start_ts, end_ts):
    """Integrate sampled Gb/s over an interval; monitor samples describe the preceding interval."""
    tx_gbit = rx_gbit = 0.0
    for previous, current in zip(points, points[1:]):
        left, right = previous[0], current[0]
        overlap = max(0.0, min(right, end_ts) - max(left, start_ts))
        if overlap:
            tx_gbit += current[1] * overlap
            rx_gbit += current[2] * overlap
    return tx_gbit / 8, rx_gbit / 8  # GB (decimal)


def write_summary(path, phases, samples):
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for phase in phases:
        for (host, net_type), points in samples.items():
            tx_gb, rx_gb = integrate(points, phase["start_ts"], phase["end_ts"])
            key = (phase["step"], phase["phase"], net_type)
            totals[key][0] += phase["duration_s"]
            totals[key][1] += tx_gb
            totals[key][2] += rx_gb
            totals[key][3] += 1

    # Duration was added once per host. Use phase intervals themselves for the denominator.
    phase_duration = defaultdict(float)
    phase_count = defaultdict(int)
    for phase in phases:
        key = (phase["step"], phase["phase"])
        phase_duration[key] += phase["duration_s"]
        phase_count[key] += 1

    with open(path, "w", newline="") as output:
        fields = ["step", "phase", "network_type", "occurrences", "duration_s", "tx_GB", "rx_GB", "avg_tx_Gbps", "avg_rx_Gbps"]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for (step, phase, net_type), (_, tx_gb, rx_gb, _) in sorted(totals.items()):
            duration = phase_duration[(step, phase)]
            writer.writerow({
                "step": step,
                "phase": phase,
                "network_type": net_type,
                "occurrences": phase_count[(step, phase)],
                "duration_s": f"{duration:.3f}",
                "tx_GB": f"{tx_gb:.3f}",
                "rx_GB": f"{rx_gb:.3f}",
                "avg_tx_Gbps": f"{tx_gb * 8 / duration:.3f}" if duration else "0",
                "avg_rx_Gbps": f"{rx_gb * 8 / duration:.3f}" if duration else "0",
            })
    return totals


def plot_report(path, label, phases, samples, totals):
    if plt is None:
        plot_svg(path.with_suffix(".svg"), label, phases, totals)
        return path.with_suffix(".svg")
    start = min(row["start_ts"] for row in phases)
    fig, axes = plt.subplots(3, 1, figsize=(18, 13), constrained_layout=True)
    colors = {"rdma": "#1f77b4", "bond": "#ff7f0e"}

    for net_type, axis in zip(("rdma", "bond"), axes[:2]):
        found = False
        for (host, sample_type), points in sorted(samples.items()):
            if sample_type != net_type:
                continue
            found = True
            x = [point[0] - start for point in points]
            axis.plot(x, [point[1] for point in points], linewidth=0.8, label=f"{host} TX")
            axis.plot(x, [point[2] for point in points], linewidth=0.8, linestyle="--", label=f"{host} RX")
        axis.set_title(f"{net_type.upper()} throughput")
        axis.set_ylabel("Gb/s")
        axis.grid(alpha=0.25)
        if found:
            axis.legend(ncol=4, fontsize=8)

    # Exclude the enclosing 'step' interval from bars to avoid presenting it as an additive phase.
    phase_names = sorted({row["phase"] for row in phases if row["phase"] != "step"})
    x_positions = list(range(len(phase_names)))
    width = 0.2
    series = [("rdma", 1, "RDMA TX"), ("rdma", 2, "RDMA RX"), ("bond", 1, "Bond TX"), ("bond", 2, "Bond RX")]
    for index, (net_type, value_index, name) in enumerate(series):
        values = [sum(value[value_index] for (step, name, kind), value in totals.items() if name == phase and kind == net_type) for phase in phase_names]
        axes[2].bar([x + (index - 1.5) * width for x in x_positions], values, width, label=name)
    axes[2].set_xticks(x_positions, phase_names, rotation=30, ha="right")
    axes[2].set_ylabel("Communication volume (GB, decimal)")
    axes[2].set_title("Communication volume by veRL phase")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(ncol=4)
    axes[2].set_xlabel("Phase (gen means trainer waiting/sampling rollout data in colocate_async)")
    fig.suptitle(f"Network communication report: {label}", fontsize=16)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_svg(path, label, phases, totals):
    """Dependency-free fallback used when matplotlib is unavailable."""
    names = sorted({row["phase"] for row in phases if row["phase"] != "step"})
    series = [("rdma", 1, "RDMA TX", "#1f77b4"), ("rdma", 2, "RDMA RX", "#6baed6"),
              ("bond", 1, "Bond TX", "#ff7f0e"), ("bond", 2, "Bond RX", "#fdae6b")]
    values = {(phase, net_type, index): sum(value[index] for (step, name, kind), value in totals.items() if name == phase and kind == net_type)
              for phase in names for net_type, index, _, _ in series}
    maximum = max(values.values(), default=1.0) or 1.0
    width, row_height = 1200, 52
    height = 130 + len(names) * row_height
    chart_left, chart_width = 220, 900
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<style>text{font-family:sans-serif} .small{font-size:13px}</style>',
             f'<text x="30" y="35" font-size="22">Network communication by phase: {label}</text>',
             f'<text x="30" y="60" class="small">Scale: 0 to {maximum:.3f} GB (decimal); step excluded because it contains all sub-phases.</text>']
    for row, phase in enumerate(names):
        y = 90 + row * row_height
        parts.append(f'<text x="10" y="{y + 18}" class="small">{phase}</text>')
        for offset, (net_type, index, title, color) in enumerate(series):
            value = values[(phase, net_type, index)]
            bar_width = chart_width * value / maximum
            bar_y = y + offset * 10
            parts.append(f'<rect x="{chart_left}" y="{bar_y}" width="{bar_width:.2f}" height="8" fill="{color}"/>')
            if value:
                parts.append(f'<text x="{chart_left + bar_width + 5:.2f}" y="{bar_y + 8}" font-size="10">{value:.2f}</text>')
    legend_x = [500, 650, 800, 950]
    for x, (_, _, title, color) in zip(legend_x, series):
        parts.extend([f'<rect x="{x}" y="25" width="12" height="12" fill="{color}"/>',
                      f'<text x="{x + 17}" y="36" class="small">{title}</text>'])
    parts.append('</svg>')
    path.write_text("\n".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network-csv", nargs="+", required=True, help="CSV paths or glob patterns")
    parser.add_argument("--phase-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="training")
    parser.add_argument("--step", default="all", help="all, first, last, or a numeric global step")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phases = load_phases(args.phase_csv)
    if not phases:
        raise SystemExit("phase CSV is empty")
    available_steps = sorted({phase["step"] for phase in phases if phase["step"] >= 0})
    if args.step != "all":
        if not available_steps:
            raise SystemExit("phase CSV does not contain a valid global step")
        selected_step = available_steps[0] if args.step == "first" else available_steps[-1] if args.step == "last" else int(args.step)
        phases = [phase for phase in phases if phase["step"] == selected_step]
        if not phases:
            raise SystemExit(f"step {selected_step} is not present in phase CSV")
        args.label = f"{args.label}, step {selected_step}"
    network_paths = expand_inputs(args.network_csv)
    missing = [path for path in network_paths if not Path(path).exists()]
    if missing:
        raise SystemExit(f"network CSV not found: {missing}")
    samples = load_network(network_paths, min(p["start_ts"] for p in phases), max(p["end_ts"] for p in phases))
    if not samples:
        raise SystemExit("no network samples overlap the recorded training phases")
    summary_name = "phase_summary.csv" if args.step == "all" else f"phase_summary_{args.step}.csv"
    summary_path = output_dir / summary_name
    totals = write_summary(summary_path, phases, samples)
    report_path = plot_report(output_dir / "communication_report.png", args.label, phases, samples, totals)
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
