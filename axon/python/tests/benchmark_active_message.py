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
import statistics
import time

import numpy as np
import test_utils  # noqa: F401 – sets up sys.path for axon module

import axon

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
        f"  {label:<26s}  "
        f"size={size_str}  "
        f"Concurrent QPS={qps:7.1f} (High Contention)"
    )


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


async def run_benchmark() -> None:
    print("=" * 80)
    print("Axon Python Eager-Path Benchmark")
    print(f"  warmup={WARMUP}  iterations={ITERATIONS}")
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

    faulthandler.enable()
    asyncio.run(run_benchmark())


if __name__ == "__main__":
    main()
