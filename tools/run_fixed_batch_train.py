"""Run the existing training shell with the dedicated Python entrypoint."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    repo = Path(os.environ.get("REPO_ROOT", "/data/xgq/projects/RLs/uni-agent"))
    source = repo / "examples/quickstart/training/train_qwen3p5_dense_test.sh"
    script = source.read_text(encoding="utf-8")
    old = "python3 -m verl.trainer.main_ppo"
    new = "python3 -m tools.fixed_batch_main_ppo"
    if script.count(old) != 1:
        raise RuntimeError(f"Expected exactly one normal VERL entrypoint in {source}")
    script = script.replace(old, new)
    script = script.replace(
        "trainer.v1.trainer_mode=colocate_async",
        "trainer.v1.trainer_mode=fixed_batch_colocate_async",
    )
    subprocess.run(["bash", "-s"], input=script, text=True, cwd=repo, env=os.environ, check=True)


if __name__ == "__main__":
    main()
