import os

# Keep BLAS/OpenMP single-threaded so only OUR workers provide parallelism.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103071
N = 20_000_000
P = 0.95          # parallelizable fraction for Amdahl's Law
REPEATS = 7       # timing repetitions per worker count (median is reported)


def amdahl(p_workers: int, parallel_fraction: float = P) -> float:
    """Theoretical max speedup: 1 / ((1 - P) + P / p)."""
    return 1.0 / ((1.0 - parallel_fraction) + parallel_fraction / p_workers)


def chunk_sum(chunk: np.ndarray) -> float:
    return float(np.sum(chunk))


def parallel_sum(pool: ThreadPoolExecutor, data: np.ndarray, workers: int) -> float:
    chunks = np.array_split(data, workers)          # views, no copy
    return sum(pool.map(chunk_sum, chunks))


def benchmark(data: np.ndarray, workers: int) -> tuple[float, float]:
    """Return (median wall time in ms, computed sum) for a worker count."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel_sum(pool, data, workers)           # warm-up (spawns threads)
        times, result = [], 0.0
        for _ in range(REPEATS):
            start = time.perf_counter()
            result = parallel_sum(pool, data, workers)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), result


def main() -> None:
    max_cores = os.cpu_count() or 1
    worker_counts = sorted({1, 2, 4, max_cores, max_cores * 2})

    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Logical cores    : {max_cores}")
    print(f"Array size       : {N:,} float64 (seed = {STUDENT_ID})")
    print(f"Worker counts    : {worker_counts}")
    print(f"Amdahl P         : {P}   |   median of {REPEATS} runs per row\n")

    rng = np.random.default_rng(STUDENT_ID)
    data = rng.random(N, dtype=np.float64)

    results = []
    for w in worker_counts:
        t_ms, total = benchmark(data, w)
        results.append((w, t_ms, total))

    t1 = results[0][1]                              # T(1)
    header = f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}"
    print(header)
    print("-" * len(header))
    for w, t_ms, _ in results:
        speedup = t1 / t_ms
        print(f"{w:>6} | {t_ms:>14.3f} | {speedup:>7.2f}x | {amdahl(w):>17.2f}x")

    print(f"\nAmdahl limit as p -> infinity: {1.0 / (1.0 - P):.2f}x")
    print(f"Sum check (p=1 vs p={results[-1][0]}): "
          f"{results[0][2]:.6f} vs {results[-1][2]:.6f}")


if __name__ == "__main__":
    main()