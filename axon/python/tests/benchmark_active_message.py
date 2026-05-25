"""Benchmark for Axon Python eager path performance.

Measures single-tensor RPC round-trip latency and throughput across three
payload sizes: a tiny packet that always stays in the Eager path, a medium
packet near the RNDV threshold, and a large packet that always triggers RNDV.

Run directly (requires axon_python_runtime.so on PYTHONPATH):
    python benchmark_active_message.py

Run via Bazel:
    bazel run //axon/python:benchmark_active_message
"""

import asyncio
import argparse
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import test_utils  # noqa: F401 – sets up sys.path for axon module

import axon

DISABLE_FAST_PATH_ENV = "AXON_DISABLE_FAST_PATH"
BENCHMARK_CHILD_ENV = "AXON_BENCHMARK_ACTIVE_MESSAGE_CHILD"

ABLATION_MODES = [
    ("fast-path (axon-future)", "0"),
    ("slow-path (eventfd)", "1"),
]

# ---------------------------------------------------------------------------
# RPC handler
# ---------------------------------------------------------------------------


async def echo(tensor: np.ndarray) -> np.ndarray:
    return tensor.copy()


# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------


class BenchmarkContext:
    def __init__(self, server_name: str = "bench_server"):
        self.server_name = server_name
        self.server: axon.AxonRuntime | None = None
        self.client: axon.AxonRuntime | None = None

    async def __aenter__(self) -> axon.AxonRuntime:
        self.server = axon.AxonRuntime(self.server_name, timeout=5000)
        self.server.start()
        self.server.register_function(echo, 0, from_dlpack_fn=np.from_dlpack)

        self.client = axon.AxonRuntime("bench_client", timeout=5000)
        self.client.start_client()
        await asyncio.sleep(0.5)
        await self.client.connect_endpoint_async(
            self.server.get_local_address(), self.server_name
        )
        return self.client

    async def __aexit__(self, *_):
        if self.client:
            self.client.stop()
        if self.server:
            self.server.stop()


async def measure_latency(
    client: axon.AxonRuntime,
    tensor: np.ndarray,
    warmup: int,
    iterations: int,
) -> list[float]:
    """Returns per-call round-trip latency samples in milliseconds."""
    invoke_kwargs = dict(
        worker_name="bench_server",
        session_id=0,
        function=0,
        from_dlpack_fn=np.from_dlpack,
    )

    # Warmup – not measured
    for _ in range(warmup):
        await client.invoke(tensor, **invoke_kwargs)

    samples: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        await client.invoke(tensor, **invoke_kwargs)
        samples.append((time.perf_counter() - t0) * 1e3)  # → ms

    return samples


async def measure_throughput(
    client: axon.AxonRuntime,
    tensor: np.ndarray,
    warmup: int,
    iterations: int,
    concurrency: int = 200,
) -> float:
    """Returns total QPS for highly concurrent requests to expose allocator contention."""
    invoke_kwargs = dict(
        worker_name="bench_server",
        session_id=0,
        function=0,
        from_dlpack_fn=np.from_dlpack,
    )

    # Warmup
    for _ in range(warmup):
        await client.invoke(tensor, **invoke_kwargs)

    t0 = time.perf_counter()

    # Fire off many concurrent requests
    tasks = []
    for _ in range(iterations):
        tasks.append(client.invoke(tensor, **invoke_kwargs))
        if len(tasks) >= concurrency:
            await asyncio.gather(*tasks)
            tasks.clear()

    if tasks:
        await asyncio.gather(*tasks)

    total_time = time.perf_counter() - t0
    return iterations / total_time


def report(label: str, size_bytes: int, samples: list[float]) -> None:
    n = len(samples)
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    p99 = sorted(samples)[int(n * 0.99)]
    qps = 1000.0 / mean

    if size_bytes >= 1024:
        size_str = f"{size_bytes // 1024:>4d} KiB"
    else:
        size_str = f"{size_bytes:>4d}   B"

    print(
        f"  {label:<26s}  "
        f"size={size_str}  "
        f"mean={mean:6.3f} ms  "
        f"p50={median:6.3f} ms  "
        f"p99={p99:6.3f} ms  "
        f"Seq QPS={qps:7.1f}"
    )


def report_throughput(label: str, size_bytes: int, qps: float) -> None:
    if size_bytes >= 1024:
        size_str = f"{size_bytes // 1024:>4d} KiB"
    else:
        size_str = f"{size_bytes:>4d}   B"

    print(
        f"  {label:<26s}  size={size_str}  Concurrent QPS={qps:7.1f} (High Contention)"
    )


def parse_child_metrics(output: str) -> dict[str, dict[str, float]]:
    latency_re = re.compile(
        r"^\s+(?P<label>.+?)\s+size=.*?mean=\s*(?P<mean>[0-9.]+) ms\s+"
        r"p50=\s*(?P<p50>[0-9.]+) ms\s+p99=\s*(?P<p99>[0-9.]+) ms\s+"
        r"Seq QPS=\s*(?P<seq_qps>[0-9.]+)",
        re.MULTILINE,
    )
    throughput_re = re.compile(
        r"^\s+(?P<label>.+?)\s+size=.*?Concurrent QPS=\s*(?P<concurrent_qps>[0-9.]+)",
        re.MULTILINE,
    )

    metrics = {
        m.group("label").strip(): {
            "mean": float(m.group("mean")),
            "p50": float(m.group("p50")),
            "p99": float(m.group("p99")),
            "seq_qps": float(m.group("seq_qps")),
        }
        for m in latency_re.finditer(output)
    }
    for m in throughput_re.finditer(output):
        label = m.group("label").strip()
        if label not in metrics:
            raise RuntimeError(f"throughput label missing latency row: {label}")
        metrics[label]["concurrent_qps"] = float(m.group("concurrent_qps"))

    if len(metrics) != len(PAYLOADS):
        raise RuntimeError(f"unexpected metric count: {len(metrics)}")
    return metrics


def run_child_mode(mode: str, disable_fast_path: str) -> str:
    env = os.environ.copy()
    env[DISABLE_FAST_PATH_ENV] = disable_fast_path
    env[BENCHMARK_CHILD_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(sys.path)

    result = subprocess.run(
        [sys.executable, sys.argv[0], "--child", "--mode", mode],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.stderr:
        print(f"[{mode} STDERR]\n{result.stderr}", file=sys.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"benchmark mode {mode} failed with exit code {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def report_ablation(results: dict[str, dict[str, dict[str, float]]]) -> None:
    baseline = results["slow-path (eventfd)"]
    print("=" * 80)
    print("Ablation Summary (Baseline = Slow Path)")
    print("=" * 80)
    for payload_label, _ in PAYLOADS:
        key = payload_label
        base_mean = baseline[key]["mean"]
        base_qps = baseline[key]["concurrent_qps"]
        print(f"\n--- {payload_label} ---")
        for mode, metrics_by_payload in results.items():
            metrics = metrics_by_payload[key]
            latency_delta = (base_mean - metrics["mean"]) / base_mean * 100.0
            qps_delta = (metrics["concurrent_qps"] - base_qps) / base_qps * 100.0
            print(
                f"  {mode:<24s} "
                f"mean={metrics['mean']:6.3f} ms "
                f"latency_delta={latency_delta:7.2f}% "
                f"concurrent_qps={metrics['concurrent_qps']:7.1f} "
                f"qps_delta={qps_delta:7.2f}%"
            )


def run_ablation() -> None:
    results: dict[str, dict[str, dict[str, float]]] = {}
    for mode, disable_fast_path in ABLATION_MODES:
        print("=" * 80)
        print(f"Running mode={mode} " f"{DISABLE_FAST_PATH_ENV}={disable_fast_path} ")
        print("=" * 80)
        output = run_child_mode(mode, disable_fast_path)
        print(output, end="" if output.endswith("\n") else "\n")
        results[mode] = parse_child_metrics(output)

    report_ablation(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

WARMUP = 20
ITERATIONS = 100

# Payload sizes chosen to cover different UCX protocol paths:
#   64 B   → always Eager (well below any eager_thresh)
#   4 KiB  → still Eager on most UCX configs (thresh ~8 KiB)
#   64 KiB → crosses into RNDV territory on typical TCP transports
PAYLOADS = [
    ("eager-tiny  (64 B)", 64),
    ("eager-small (4 KiB)", 4 * 1024),
    ("rndv-large  (64 KiB)", 64 * 1024),
]


async def run_benchmark(mode: str) -> None:
    print("=" * 80)
    print("Axon Python Eager-Path Benchmark")
    print(f"  mode={mode}  warmup={WARMUP}  iterations={ITERATIONS}")
    print(f"  {DISABLE_FAST_PATH_ENV}={os.environ.get(DISABLE_FAST_PATH_ENV, '0')}  ")
    print("=" * 80)

    async with BenchmarkContext() as client:
        for label, size in PAYLOADS:
            print(f"\n--- Testing {label} ---")
            tensor = np.zeros(size, dtype=np.uint8)

            # Sequential latency test
            samples = await measure_latency(client, tensor, WARMUP, ITERATIONS)
            report(label, size, samples)

            # Concurrent throughput test
            # Increase iterations for throughput to get a stable measure
            throughput_iters = 1000
            qps = await measure_throughput(client, tensor, WARMUP, throughput_iters)
            report_throughput(label, size, qps)

    print()


def main() -> None:
    import faulthandler

    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--mode", default="manual")
    args = parser.parse_args()

    faulthandler.enable()
    if args.child or os.environ.get(BENCHMARK_CHILD_ENV) == "1":
        asyncio.run(run_benchmark(args.mode))
        return

    run_ablation()


if __name__ == "__main__":
    main()
