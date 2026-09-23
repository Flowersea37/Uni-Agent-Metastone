#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="https://hf-mirror.com"
# Prepare the Uni-Agent-formatted SWE-Bench Verified parquet used by
# examples/inference/parallel_infer_api.py and parallel_infer_verl.py.
#
# Default output:
#   ~/data/swe_agent/swe_bench_verified.parquet
#
# For a quick smoke test, this script keeps only 1 instance by default.  Set
# MAX_INSTANCES=0 or unset the flag below yourself if you want the full dataset.

SAVE_DIR="${SAVE_DIR:-/data/xgq/data/swe_agent}"
MAX_INSTANCES="${MAX_INSTANCES:-1}"

python3 - <<'PY'
import importlib.util
import sys

missing = [
    package
    for package in ("datasets", "pyarrow", "swebench")
    if importlib.util.find_spec(package) is None
]
if missing:
    print(
        "Missing Python packages: " + ", ".join(missing) + "\n"
        "Install them first, for example:\n"
        "  python3 -m pip install datasets pyarrow swebench",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

mkdir -p "${SAVE_DIR}"

python3 -m uni_agent.tasks.swe_bench.preprocess \
    --local-save-dir "${SAVE_DIR}" \
    # --max-instances "${MAX_INSTANCES}"

echo
echo "Wrote: ${SAVE_DIR}/swe_bench_verified.parquet"
echo
echo "Use it with:"
echo "  DATA_PATH=\"${SAVE_DIR}/swe_bench_verified.parquet\" \\"
echo "  LIMIT=1 \\"
echo "  CONCURRENCY=1 \\"
echo "  ./examples/quickstart/inference/run_infer_mini_swe_agent_blackbox.sh"
