#!/usr/bin/env bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${1:-/data/xgq/projects/RLs/uni-agent/train_logs/logs/Uni-Agent-Qwen3.5-2B-megatron/separate_async_Qwen3.5-2B_20260915_064236}
failures=0
processed=0

while IFS= read -r -d "" file; do
    session_dir=$(dirname "$file")
    if python3 "$SCRIPT_DIR/tools/inspect_trajectory.py" "$session_dir" \
        --output "$session_dir/trajectory_readable.json"; then
        ((processed += 1))
    else
        ((failures += 1))
        echo "Failed to parse: $session_dir" >&2
    fi
done < <(find "$ROOT" -type f -name trajectory.npz -print0)

python3 "$SCRIPT_DIR/tools/summarize_rollout_timings.py" "$ROOT"
echo "Parsed $processed trajectory file(s); failures: $failures"
exit "$failures"
