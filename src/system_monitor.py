"""Lightweight host/container telemetry for the web UI."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass


@dataclass
class _CpuSnapshot:
    ts: float
    total: int
    idle: int


_LAST_CPU: _CpuSnapshot | None = None


def _read_cpu_snapshot() -> _CpuSnapshot | None:
    try:
        fields = PathLikeRead("/proc/stat").splitlines()[0].split()[1:]
        vals = [int(v) for v in fields]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return _CpuSnapshot(time.monotonic(), sum(vals), idle)
    except Exception:
        return None


def _cpu_percent() -> float | None:
    global _LAST_CPU
    snap = _read_cpu_snapshot()
    if snap is None:
        return None
    prev = _LAST_CPU
    _LAST_CPU = snap
    if prev is None:
        return None
    total_delta = snap.total - prev.total
    idle_delta = snap.idle - prev.idle
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def _memory() -> dict[str, float] | None:
    try:
        info: dict[str, int] = {}
        for line in PathLikeRead("/proc/meminfo").splitlines():
            key, rest = line.split(":", 1)
            info[key] = int(rest.strip().split()[0]) * 1024
        total = info.get("MemTotal")
        available = info.get("MemAvailable")
        if not total or available is None:
            return None
        used = total - available
        return {
            "ram_used_gb": used / (1024**3),
            "ram_total_gb": total / (1024**3),
            "ram_percent": 100.0 * used / total,
        }
    except Exception:
        return None


def _gpu() -> dict[str, float | str] | None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2.0,
        ).strip()
    except Exception:
        return None
    if not out:
        return None
    rows = []

    def _num(value: str) -> float | None:
        try:
            return float(value)
        except ValueError:
            return None

    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        util = _num(parts[1])
        temp = _num(parts[4])
        if util is None:
            continue
        rows.append(
            {
                "name": parts[0],
                "util": util,
                "mem_used_mb": _num(parts[2]),
                "mem_total_mb": _num(parts[3]),
                "temp_c": temp,
            }
        )
    if not rows:
        return None
    util = max(r["util"] for r in rows)
    temps = [r["temp_c"] for r in rows if r["temp_c"] is not None]
    result: dict[str, float | str] = {
        "gpu_name": rows[0]["name"] if len(rows) == 1 else f"{len(rows)} GPUs",
        "gpu_percent": util,
    }
    if temps:
        result["gpu_temp_c"] = max(temps)
    mem_rows = [
        r
        for r in rows
        if r["mem_used_mb"] is not None and r["mem_total_mb"] not in (None, 0)
    ]
    if mem_rows:
        total_mem = sum(float(r["mem_total_mb"]) for r in mem_rows)
        used_mem = sum(float(r["mem_used_mb"]) for r in mem_rows)
        result.update(
            {
                "gpu_mem_used_gb": used_mem / 1024.0,
                "gpu_mem_total_gb": total_mem / 1024.0,
                "gpu_mem_percent": 100.0 * used_mem / total_mem if total_mem else 0.0,
            }
        )
    else:
        result["gpu_mem_note"] = "unified memory"
    return result


def PathLikeRead(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sample_system() -> dict:
    """Return best-effort system utilization metrics."""
    data: dict[str, float | str] = {"pid": os.getpid()}
    cpu = _cpu_percent()
    if cpu is not None:
        data["cpu_percent"] = cpu
    mem = _memory()
    if mem:
        data.update(mem)
    gpu = _gpu()
    if gpu:
        data.update(gpu)
    return data
