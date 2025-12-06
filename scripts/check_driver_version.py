#!/usr/bin/env python3
"""
Check NVIDIA runtime info inside a docker-compose service.

Runs `docker compose exec <service> nvidia-smi ...` to gather driver/CUDA
versions, compares the driver to an expected value, and prints a short
inventory of GPUs.
"""

import argparse
import csv
import re
import subprocess
import sys
from typing import List


EXPECTED_DRIVER = "570.172.08"


def run_cmd(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def fetch_nvidia_info(service: str):
    query_cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        service,
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    ]
    result = run_cmd(query_cmd)
    if result.returncode != 0:
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        details = stderr or stdout or "(no output)"
        sys.stderr.write(f"Failed to run command ({' '.join(query_cmd)}): {details}\n")
        return None, result.returncode

    rows = []
    reader = csv.reader(line for line in result.stdout.splitlines() if line.strip())
    for row in reader:
        if len(row) >= 2:
            name, driver = (part.strip() for part in row[:2])
            rows.append({"name": name, "driver": driver})
    if not rows:
        sys.stderr.write("No driver info returned from nvidia-smi.\n")
        return None, 1

    # CUDA version is only printed in the banner; query fields don't include it on
    # some driver versions. Parse the banner separately.
    cuda_cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        service,
        "nvidia-smi",
    ]
    cuda_version = None
    cuda_result = run_cmd(cuda_cmd)
    if cuda_result.returncode == 0:
        match = re.search(r"CUDA Version:\s*([\d.]+)", cuda_result.stdout)
        if match:
            cuda_version = match.group(1)

    for row in rows:
        row["cuda"] = cuda_version or "unknown"

    return rows, 0


def print_inventory(rows):
    print("GPU inventory (name | driver | CUDA reported):")
    for idx, entry in enumerate(rows):
        print(f"  [{idx}] {entry['name']} | {entry['driver']} | {entry['cuda']}")


def check_driver(service: str, expected: str) -> int:
    rows, code = fetch_nvidia_info(service)
    if rows is None:
        return code

    driver_versions = {row["driver"] for row in rows}
    cuda_versions = {row["cuda"] for row in rows}

    print_inventory(rows)
    print(f"Driver versions: {', '.join(sorted(driver_versions))}")
    print(f"CUDA versions (reported by driver): {', '.join(sorted(cuda_versions))}")

    if not driver_versions:
        sys.stderr.write("No driver version returned from nvidia-smi.\n")
        return 1

    if driver_versions == {expected}:
        print(f"Driver version matches expected {expected}.")
        return 0

    print(f"Driver version mismatch. Expected {expected}, got {', '.join(sorted(driver_versions))}.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check NVIDIA driver + CUDA info inside a docker-compose service."
    )
    parser.add_argument(
        "service",
        nargs="?",
        default="miner",
        help="docker-compose service name to exec into (default: miner)",
    )
    parser.add_argument(
        "--expected",
        default=EXPECTED_DRIVER,
        help=f"expected driver version string (default: {EXPECTED_DRIVER})",
    )
    args = parser.parse_args()
    return check_driver(args.service, args.expected)


if __name__ == "__main__":
    sys.exit(main())
