import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103071
TASKS = 10_000
ROUNDS = 500          
RUNS = 3
TOLERANCE = 1.03     
cores = os.cpu_count() or 1
sizes = sorted({1, max(1, cores // 2), cores, cores * 2, cores * 4, cores * 16, 256})

rng = np.random.default_rng(STUDENT_ID)
block = rng.integers(0, 256, size=1024, dtype=np.uint8)    
@njit(nogil=True)
def hash_task(data, rounds, salt):
    h = np.uint64(14695981039346656037) ^ np.uint64(salt)
    prime = np.uint64(1099511628211)
    for _ in range(rounds):
        for i in range(len(data)):
            h = (h ^ np.uint64(data[i])) * prime
    return h


def run_batch(pool_size):
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=pool_size) as pool:
        results = list(pool.map(lambda t: hash_task(block, ROUNDS, t), range(TASKS)))
    ms = (time.perf_counter() - start) * 1000
    return ms, results


if __name__ == "__main__":
    print("Student ID    :", STUDENT_ID)
    print("Logical cores :", cores)
    print("Tasks         :", f"{TASKS:,}", "x", ROUNDS, "hash rounds over 1KB")
    print("Pool sizes    :", sizes)
    print("Timing        : median of", RUNS, "runs\n")

    hash_task(block, 1, 0)       # compile first, otherwise the first row is slow

    times = []
    reference = None
    for s in sizes:
        runs = []
        for _ in range(RUNS):
            ms, res = run_batch(s)
            runs.append(ms)
            if reference is None:
                reference = res
            assert res == reference, "results changed, something is wrong"
        runs.sort()
        times.append(runs[len(runs) // 2])

    best_i = times.index(min(times))
    best = times[best_i]

    worse_i = None
    for i in range(best_i + 1, len(sizes)):
        if times[i] > best * TOLERANCE:
            worse_i = i
            break

    print(f"{'Pool Size':>9} | {'Execution Time (ms)':>19} | {'Speedup vs 1 thread':>19} | Status")
    print("-" * 75)
    for i, s in enumerate(sizes):
        if i == best_i:
            status = "BEST"
        elif i < best_i:
            status = "still improving"
        elif worse_i is not None and i == worse_i:
            status = "<-- starts WORSENING"
        elif worse_i is not None and i > worse_i:
            status = "degraded"
        else:
            status = "about the same as best"
        print(f"{s:>9} | {times[i]:>19.1f} | {times[0] / times[i]:>18.2f}x | {status}")

    print()
    print(f"Fastest pool size: {sizes[best_i]} ({best:.1f} ms)")
    if worse_i is None:
        print("Execution time never got clearly worse in the sizes I tested.")
    else:
        print(f"Execution time starts getting worse at pool size {sizes[worse_i]} "
              f"({times[worse_i]:.1f} ms, {times[worse_i] / best:.2f}x the best).")