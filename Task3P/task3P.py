import sys
import time
import threading

import numpy as np
from numba import njit

STUDENT_ID = 230103071
N = 2_000_000
THREADS = 4
RUNS = 5


@njit(nogil=True)
def serial_scan(a, out):
    out[0] = a[0]
    for i in range(1, len(a)):
        out[i] = out[i - 1] + a[i]


@njit(nogil=True)
def scan_chunk(a, out, lo, hi):
    out[lo] = a[lo]
    for i in range(lo + 1, hi):
        out[i] = out[i - 1] + a[i]
    return out[hi - 1]


@njit(nogil=True)
def add_offset(out, lo, hi, offset):
    for i in range(lo, hi):
        out[i] += offset


def parallel_scan(a, out):
    size = len(a) // THREADS
    bounds = []
    for t in range(THREADS):
        lo = t * size
        hi = (t + 1) * size if t < THREADS - 1 else len(a)   
        bounds.append((lo, hi))

    totals = [0] * THREADS

    def pass1(t):
        lo, hi = bounds[t]
        totals[t] = scan_chunk(a, out, lo, hi)

    def pass2(t, off):
        lo, hi = bounds[t]
        add_offset(out, lo, hi, off)

    ths = [threading.Thread(target=pass1, args=(t,)) for t in range(THREADS)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()

    offsets = [0] * THREADS
    for t in range(1, THREADS):
        offsets[t] = offsets[t - 1] + totals[t - 1]

    ths = [threading.Thread(target=pass2, args=(t, offsets[t])) for t in range(1, THREADS)]
    for th in ths:
        th.start()
    for th in ths:
        th.join()

# so we re making leftover gone to the last chunk here 
def best_time(fn):
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    times.sort()
    return times[len(times) // 2]      # median


if __name__ == "__main__":
    rng = np.random.default_rng(STUDENT_ID)
    a = rng.integers(1, 11, size=N).astype(np.int64)   

    serial_out = np.zeros(N, dtype=np.int64)
    par_out = np.zeros(N, dtype=np.int64)

    serial_scan(a, serial_out)
    parallel_scan(a, par_out)

    t_serial = best_time(lambda: serial_scan(a, serial_out))
    t_par = best_time(lambda: parallel_scan(a, par_out))

    print("Student ID :", STUDENT_ID)
    print("Array size :", f"{N:,}", "| threads:", THREADS)
    print("Last value : serial =", serial_out[-1], "| parallel =", par_out[-1])
    print()

    ok = par_out[-1] == serial_out[-1] and np.array_equal(par_out, serial_out)
    status = "PASS" if ok else "FAIL"

    print(f"Serial Time (ms) | Parallel Time (ms) | Speedup | Verification")
    print(f"{t_serial:>16.2f} | {t_par:>18.2f} | {t_serial / t_par:>6.2f}x | {status}")

    if not ok:
        sys.exit(1)