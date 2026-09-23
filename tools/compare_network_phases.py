#!/usr/bin/env python3
"""Create the final IB versus no-IB phase comparison from two run summaries."""

import argparse
import csv
from pathlib import Path


def load(path):
    with open(path, newline="") as source:
        return {(row["phase"], row["network_type"]): row for row in csv.DictReader(source)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ib-summary", required=True)
    parser.add_argument("--no-ib-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    ib, no_ib = load(args.ib_summary), load(args.no_ib_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted((set(ib) | set(no_ib)) - {("step", "rdma"), ("step", "bond")})

    rows = []
    for phase, net_type in keys:
        left, right = ib.get((phase, net_type), {}), no_ib.get((phase, net_type), {})
        row = {"phase": phase, "network_type": net_type}
        for field in ("tx_GB", "rx_GB", "avg_tx_Gbps", "avg_rx_Gbps"):
            ib_value, no_ib_value = float(left.get(field, 0)), float(right.get(field, 0))
            row[f"ib_{field}"] = f"{ib_value:.3f}"
            row[f"no_ib_{field}"] = f"{no_ib_value:.3f}"
            row[f"delta_{field}"] = f"{ib_value - no_ib_value:.3f}"
        rows.append(row)

    csv_path = output_dir / "ib_vs_no_ib.csv"
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys() if rows else ["phase", "network_type"])
        writer.writeheader()
        writer.writerows(rows)

    # The final chart compares total TX+RX volume for each phase and network type.
    labels = [f"{row['phase']} ({row['network_type']})" for row in rows]
    ib_values = [float(row["ib_tx_GB"]) + float(row["ib_rx_GB"]) for row in rows]
    no_ib_values = [float(row["no_ib_tx_GB"]) + float(row["no_ib_rx_GB"]) for row in rows]
    maximum = max(ib_values + no_ib_values, default=1.0) or 1.0
    width, row_height = 1300, 42
    height = 100 + len(rows) * row_height
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
             '<style>text{font-family:sans-serif}.small{font-size:12px}</style>',
             '<text x="20" y="30" font-size="22">IB vs no-IB communication volume by veRL phase</text>',
             '<rect x="850" y="17" width="14" height="14" fill="#1f77b4"/><text x="870" y="29" class="small">IB</text>',
             '<rect x="920" y="17" width="14" height="14" fill="#ff7f0e"/><text x="940" y="29" class="small">no-IB</text>']
    for index, (label, ib_value, no_ib_value) in enumerate(zip(labels, ib_values, no_ib_values)):
        y = 60 + index * row_height
        parts.append(f'<text x="10" y="{y + 15}" class="small">{label}</text>')
        parts.append(f'<rect x="260" y="{y}" width="{850 * ib_value / maximum:.2f}" height="14" fill="#1f77b4"/>')
        parts.append(f'<rect x="260" y="{y + 17}" width="{850 * no_ib_value / maximum:.2f}" height="14" fill="#ff7f0e"/>')
        parts.append(f'<text x="1120" y="{y + 12}" class="small">IB {ib_value:.2f} / no-IB {no_ib_value:.2f} GB</text>')
    parts.append("</svg>")
    svg_path = output_dir / "ib_vs_no_ib.svg"
    svg_path.write_text("\n".join(parts))
    print(f"Comparison chart: {svg_path}")
    print(f"Comparison data: {csv_path}")


if __name__ == "__main__":
    main()
