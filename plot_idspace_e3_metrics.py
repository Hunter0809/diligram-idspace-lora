#!/usr/bin/env python3
"""Plot IDSpace E3 training metrics with optimizer steps on the x-axis."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path, help="metrics.jsonl produced by idspace_e3_train.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.metrics.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in args.metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"no metrics found: {args.metrics}")

    normalized = []
    for row in rows:
        normalized.append({
            "step": int(row.get("optimizer_step", row.get("micro_step", 0))),
            "micro_step": int(row.get("micro_step", 0)),
            "loss": float(row.get("loss", 0.0)),
            "learning_rate": float(row.get("learning_rate", 0.0)),
            "samples_per_second": float(row.get("samples_per_second", 0.0)),
            "tokens_per_second": float(row.get("tokens_per_second", 0.0)),
            "peak_memory_allocated_mb": float(row.get("peak_memory_allocated_mb", 0.0)),
            "peak_memory_reserved_mb": float(row.get("peak_memory_reserved_mb", 0.0)),
        })

    csv_path = output_dir / "training_metrics_by_step.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0]))
        writer.writeheader()
        writer.writerows(normalized)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"CSV saved to {csv_path}; install matplotlib to render PNG: {exc}") from exc

    steps = [row["step"] for row in normalized]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plots = [
        (axes[0, 0], "loss", "Loss", "Loss"),
        (axes[0, 1], "learning_rate", "Learning rate", "Learning rate"),
        (axes[1, 0], "samples_per_second", "Throughput", "Samples / second"),
        (axes[1, 1], "peak_memory_allocated_mb", "Peak GPU memory", "Allocated MiB"),
    ]
    for axis, key, title, ylabel in plots:
        axis.plot(steps, [row[key] for row in normalized], linewidth=1.4)
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    fig.suptitle("IDSpace E3 training metrics")
    png_path = output_dir / "training_metrics_by_step.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(json.dumps({"csv": str(csv_path), "png": str(png_path), "points": len(normalized)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
