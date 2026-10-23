import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103071
N = 8000
REPEATS = 5


def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n:
            return n
    except Exception:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


@njit(nogil=True)
def sum_rows(m, r0, r1):
    total = 0.0
    for i in range(r0, r1):         
        for j in range(m.shape[1]):  
            total += m[i, j]
    return total


@njit(nogil=True)
def sum_cols(m, c0, c1):
    total = 0.0
    for j in range(c0, c1):          
        for i in range(m.shape[0]): 
            total += m[i, j]
    return total


def run(kernel, m, workers) -> tuple[float, float]:
    bounds = np.linspace(0, N, workers + 1, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        times, result = [], 0.0
        for _ in range(REPEATS + 1):                      
            start = time.perf_counter()
            futs = [pool.submit(kernel, m, int(bounds[w]), int(bounds[w + 1]))
                    for w in range(workers)]
            result = sum(f.result() for f in futs)
            times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times[1:]), result


def main() -> None:
    workers = physical_cores()
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Logical cores    : {os.cpu_count()}")
    print(f"Worker threads   : {workers} (physical cores)")
    print(f"Matrix           : {N}x{N} float64 ({N * N * 8 / 1e6:.0f} MB, seed = {STUDENT_ID})")
    print(f"Timing           : median of {REPEATS} runs\n")

    tiny = np.ones((64, 64))
    sum_rows(tiny, 0, 64)
    sum_cols(tiny, 0, 64)                                 

    rng = np.random.default_rng(STUDENT_ID)
    m = rng.random((N, N), dtype=np.float64)

    t_a, s_a = run(sum_rows, m, workers)
    t_b, s_b = run(sum_cols, m, workers)
    assert np.isclose(s_a, s_b, rtol=1e-9), "modes disagree!"

    print(f"Mode A (row-major, linear)     : {t_a:>10.1f} ms   sum = {s_a:.4f}")
    print(f"Mode B (column-major, strided) : {t_b:>10.1f} ms   sum = {s_b:.4f}")
    print(f"\nCache penalty factor (Mode B / Mode A): {t_b / t_a:.2f}x")


if __name__ == "__main__":
    main()
