"""Small CSV recorder for correlating PPO phases with external monitoring data."""

import csv
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path


_LOCK = threading.Lock()
_HEADER = ["step", "trigger_step", "phase", "start_ts", "end_ts", "duration_s", "status", "host"]


@contextmanager
def record_phase(phase: str, step: int | None = None, trigger_step: int | None = None):
    """Append one phase interval when VERL_PHASE_LOG is configured."""
    path_value = os.getenv("VERL_PHASE_LOG")
    start_wall = time.time()
    start_mono = time.perf_counter()
    status = "ok"
    try:
        yield
    except BaseException:
        status = "error"
        raise
    finally:
        if path_value:
            path = Path(path_value)
            wrote_phase = False
            try:
                end_wall = time.time()
                row = [
                    "" if step is None else step,
                    "" if trigger_step is None else trigger_step,
                    phase,
                    f"{start_wall:.6f}",
                    f"{end_wall:.6f}",
                    f"{time.perf_counter() - start_mono:.6f}",
                    status,
                    socket.gethostname(),
                ]
                with _LOCK:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    needs_header = not path.exists() or path.stat().st_size == 0
                    with path.open("a", newline="") as output:
                        writer = csv.writer(output)
                        if needs_header:
                            writer.writerow(_HEADER)
                        writer.writerow(row)
                        output.flush()
                wrote_phase = True
            except Exception as error:
                print(f"[phase-monitor] warning: could not record {phase}: {error}", flush=True)
            if phase == "step" and wrote_phase:
                print(f"[phase-monitor] step {step} recorded in {path}", flush=True)
                network_pattern = os.getenv("VERL_NETWORK_CSV_PATTERN")
                summary_path = os.getenv("VERL_PHASE_SUMMARY")
                if network_pattern and summary_path:
                    try:
                        from verl.utils.network_phase_summary import refresh_network_summary

                        refresh_network_summary(str(path), network_pattern, summary_path)
                        print(f"[phase-monitor] network summary refreshed: {summary_path}", flush=True)
                    except Exception as error:
                        print(f"[phase-monitor] warning: could not refresh network summary: {error}", flush=True)
