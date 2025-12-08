#!/usr/bin/env python3
"""
summarize_results.py

Aggregates benchmark CSV results and generates a summary report in Markdown format.

Usage:
    python benchmarks/summarize_results.py benchmarks/results/*.csv
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def load_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load CSV file and return list of rows as dictionaries."""
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def is_api_provider(label: str) -> bool:
    """Check if label is for an API provider."""
    api_keywords = ['replicate', 'fal', 'together', 'api']
    return any(keyword in label.lower() for keyword in api_keywords)


def compute_stats(latencies: List[float], api_costs: List[float] = None) -> Dict[str, float]:
    """Compute statistics from latency list and optionally API costs."""
    if not latencies:
        return {
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "avg": 0.0,
            "min": 0.0,
            "max": 0.0,
            "std": 0.0,
            "throughput": 0.0,
            "api_cost_per_image": 0.0,
        }

    arr = np.array(latencies)
    avg = float(np.mean(arr))
    throughput = 1.0 / avg if avg > 0 else 0.0

    # Calculate API cost if provided
    api_cost_per_image = 0.0
    if api_costs and len(api_costs) > 0:
        api_cost_per_image = float(np.mean(api_costs))

    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "avg": avg,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "throughput": throughput,
        "api_cost_per_image": api_cost_per_image,
    }


def compute_cost_per_1000_images(
    throughput_images_per_sec: float,
    gpu_cost_per_hour: float
) -> float:
    """
    Compute cost per 1,000 images.

    Args:
        throughput_images_per_sec: Images per second
        gpu_cost_per_hour: GPU cost in dollars per hour

    Returns:
        Cost in dollars per 1,000 images
    """
    if throughput_images_per_sec <= 0:
        return 0.0

    # Time to generate 1000 images (in hours)
    hours_per_1000_images = 1000.0 / (throughput_images_per_sec * 3600.0)

    # Cost = hours * cost_per_hour
    return hours_per_1000_images * gpu_cost_per_hour


def main():
    parser = argparse.ArgumentParser(description="Summarize benchmark results from CSV files")
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Paths to CSV result files"
    )
    parser.add_argument(
        "--gpu_cost_per_hour",
        type=float,
        default=1.10,
        help="GPU cost per hour in dollars (for cost calculation). Default: $1.10 (example H100 spot price)"
    )

    args = parser.parse_args()

    # Aggregate data by label and batch_size
    data_by_config = defaultdict(lambda: {"latencies": [], "costs": []})

    for csv_file in args.csv_files:
        csv_path = Path(csv_file)
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_file}", file=sys.stderr)
            continue

        rows = load_csv(csv_path)
        for row in rows:
            label = row.get('label', 'unknown')
            batch_size = int(row.get('batch_size', 1))
            latency = float(row.get('latency_seconds', 0.0))
            success = row.get('success', 'True').lower() == 'true'
            cost = float(row.get('cost_usd', 0.0)) if 'cost_usd' in row else 0.0

            if success and latency > 0:
                key = (label, batch_size)
                data_by_config[key]["latencies"].append(latency)
                if cost > 0:
                    data_by_config[key]["costs"].append(cost)

    if not data_by_config:
        print("Error: No valid data found in CSV files", file=sys.stderr)
        sys.exit(1)

    # Compute statistics for each configuration
    results = []
    for (label, batch_size), data in sorted(data_by_config.items()):
        latencies = data["latencies"]
        costs = data["costs"]

        stats = compute_stats(latencies, costs)

        # For API providers, use actual costs; for local, calculate GPU cost
        if stats["api_cost_per_image"] > 0:
            cost_per_1k = stats["api_cost_per_image"] * 1000
        else:
            cost_per_1k = compute_cost_per_1000_images(
                stats["throughput"],
                args.gpu_cost_per_hour
            )

        results.append({
            "label": label,
            "batch_size": batch_size,
            "count": len(latencies),
            **stats,
            "cost_per_1k_images": cost_per_1k,
        })

    # Print Markdown table
    print("\n" + "="*80)
    print("FLUX-1.DEV BENCHMARK SUMMARY")
    print("="*80)
    print()

    # Summary table
    print("| Configuration | Batch | Count | p50 (s) | p90 (s) | p95 (s) | Avg (s) | Throughput (img/s) | Cost/1K imgs |")
    print("|---------------|------:|------:|--------:|--------:|--------:|--------:|-------------------:|-------------:|")

    for result in results:
        print(
            f"| {result['label']:<13} "
            f"| {result['batch_size']:>5} "
            f"| {result['count']:>5} "
            f"| {result['p50']:>7.3f} "
            f"| {result['p90']:>7.3f} "
            f"| {result['p95']:>7.3f} "
            f"| {result['avg']:>7.3f} "
            f"| {result['throughput']:>18.3f} "
            f"| ${result['cost_per_1k_images']:>11.2f} |"
        )

    print()
    print("="*80)
    print()

    # Detailed statistics
    print("## Detailed Statistics")
    print()
    for result in results:
        print(f"### {result['label']} (Batch Size: {result['batch_size']})")
        print(f"- **Samples**: {result['count']}")
        print(f"- **p50 Latency**: {result['p50']:.3f} seconds")
        print(f"- **p90 Latency**: {result['p90']:.3f} seconds")
        print(f"- **p95 Latency**: {result['p95']:.3f} seconds")
        print(f"- **Average Latency**: {result['avg']:.3f} seconds")
        print(f"- **Min Latency**: {result['min']:.3f} seconds")
        print(f"- **Max Latency**: {result['max']:.3f} seconds")
        print(f"- **Std Dev**: {result['std']:.3f} seconds")
        print(f"- **Throughput**: {result['throughput']:.3f} images/second")
        print(f"- **Cost per 1,000 images**: ${result['cost_per_1k_images']:.2f} (at ${args.gpu_cost_per_hour:.2f}/hour)")
        print()

    # Cost calculation explanation
    print("## Cost Calculation")
    print()
    print(f"**GPU Cost**: ${args.gpu_cost_per_hour:.2f} per hour")
    print()
    print("**Formula**:")
    print("```")
    print("cost_per_1000_images = (1000 / throughput_images_per_sec) * (gpu_cost_per_hour / 3600)")
    print("                     = (1000 / (images/sec)) * ($/hour / 3600 sec/hour)")
    print("```")
    print()
    print("**Example**:")
    if results:
        example = results[0]
        print(f"- Configuration: {example['label']}")
        print(f"- Throughput: {example['throughput']:.3f} images/second")
        print(f"- Time per 1000 images: {1000 / example['throughput']:.1f} seconds = {1000 / example['throughput'] / 60:.1f} minutes")
        print(f"- Cost: (1000 / {example['throughput']:.3f}) * (${args.gpu_cost_per_hour:.2f} / 3600) = ${example['cost_per_1k_images']:.2f}")
        print()

    # Speedup comparison (if we have both baseline and optimized)
    baseline_results = [r for r in results if 'baseline' in r['label'].lower()]
    optimized_results = [r for r in results if 'trt' in r['label'].lower()]

    if baseline_results and optimized_results:
        print("## Speedup Analysis")
        print()
        for base in baseline_results:
            for opt in optimized_results:
                if base['batch_size'] == opt['batch_size']:
                    speedup = base['avg'] / opt['avg'] if opt['avg'] > 0 else 0.0
                    cost_savings = ((base['cost_per_1k_images'] - opt['cost_per_1k_images']) / base['cost_per_1k_images'] * 100) if base['cost_per_1k_images'] > 0 else 0.0

                    print(f"### Batch Size {base['batch_size']}")
                    print(f"- **Baseline**: {base['label']} - {base['avg']:.3f}s avg, {base['throughput']:.3f} img/s")
                    print(f"- **Optimized**: {opt['label']} - {opt['avg']:.3f}s avg, {opt['throughput']:.3f} img/s")
                    print(f"- **Speedup**: {speedup:.2f}x faster")
                    print(f"- **Cost Savings**: {cost_savings:.1f}% reduction (${base['cost_per_1k_images']:.2f} → ${opt['cost_per_1k_images']:.2f} per 1K images)")
                    print()

    print("="*80)


if __name__ == "__main__":
    main()
