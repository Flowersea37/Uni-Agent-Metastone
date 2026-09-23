#!/usr/bin/env bash
set -euo pipefail

# Quickstart wrapper for preparing the SWE-Bench Verified parquet.

exec examples/inference/prepare_swe_bench_verified_data.sh "$@"
