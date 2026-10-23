import os
import platform
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103071
ITERS = 50_000_000
REPEATS = 3
PAD = 16            # 16 * 8 bytes = 128 bytes between counters (>= 64-byte line)


def physical_cores() -> int:
    if len(sys.argv) > 1:                       # manual override: python task3.py 16
        return int(sys.argv[1])
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)   # fallback guess


@njit(nogil=True)
def hammer(counts, off, slot, iters):
    # `off` is re-read every iteration, so the compiler cannot keep the
    # counter in a register: every increment is a real load+store to memory.
    for _ in range(iters):
        counts[slot + off[0]] += 1


def run(n_threads: int, stride: int) -> tuple[float, np.ndarray]:
    counts = np.zeros(n_threads * stride + stride, dtype=np.int64)
    off = np.zeros(1, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        start = time.perf_counter()
        futures = [pool.submit(hammer, counts, off, i * stride, ITERS)
                   for i in range(n_threads)]
        for f in futures:
            f.result()
        elapsed = (time.perf_counter() - start) * 1000.0
    return elapsed, counts[::stride][:n_threads]


def main() -> None:
    n = physical_cores()
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Logical cores    : {os.cpu_count()}")
    print(f"Threads used (N) : {n}  (physical cores)")
    print(f"Increments/thread: {ITERS:,}   |   median of {REPEATS} runs\n")

    hammer(np.zeros(4, np.int64), np.zeros(1, np.int64), 0, 10)   # JIT warm-up

    results = {}
    for label, stride in (("Unpadded Shared", 1), ("Padded (128 B)", PAD)):
        times = []
        for _ in range(REPEATS):
            t, final = run(n, stride)
            assert (final == ITERS).all(), "counter mismatch!"
            times.append(t)
        results[label] = statistics.median(times)
        print(f"{label:<16}: {results[label]:>10.1f} ms   (runs: "
              f"{', '.join(f'{x:.0f}' for x in times)})")

    ratio = results["Unpadded Shared"] / results["Padded (128 B)"]
    print(f"\nExecution ratio (Unpadded / Padded): {ratio:.2f}x slower when unpadded")


if __name__ == "__main__":
    main()