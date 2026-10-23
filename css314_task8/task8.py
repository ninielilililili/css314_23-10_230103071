
import os
import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

STUDENT_ID = 230103071
N = 1024
REPEATS = 3
MAX_THREADS = os.cpu_count() or 1


@njit(nogil=True)
def mm_ijk(A, B, C, r0, r1, n, T):
    for i in range(r0, r1):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit(nogil=True)
def mm_ikj(A, B, C, r0, r1, n, T):
    for i in range(r0, r1):
        for j in range(n):
            C[i, j] = 0.0
        for k in range(n):
            a = A[i, k]
            for j in range(n):
                C[i, j] += a * B[k, j]


@njit(nogil=True)
def mm_tiled(A, B, C, r0, r1, n, T):
    # Requires n % T == 0.  Row slices (Bk, Ci) give the compiler contiguous
    # 1-D views so the innermost loop vectorizes, same as the non-tiled ikj kernel.
    for i in range(r0, r1):
        for j in range(n):
            C[i, j] = 0.0
    for ii in range(r0, r1, T):
        i_end = min(ii + T, r1)
        for kk in range(0, n, T):
            for jj in range(0, n, T):
                for i in range(ii, i_end):
                    Ci = C[i, jj:jj + T]
                    for k in range(kk, kk + T):
                        a = A[i, k]
                        Bk = B[k, jj:jj + T]
                        for j in range(T):
                            Ci[j] += a * Bk[j]


VARIANTS = [
    ("Non-tiled ijk", mm_ijk, 0, "n/a (B column stride)"),
    ("Non-tiled ikj", mm_ikj, 0, "n/a (B row = 8 KB)"),
    ("Tiled 16x16", mm_tiled, 16, f"{3 * 16 * 16 * 8 / 1024:.0f} KB (L1)"),
    ("Tiled 64x64", mm_tiled, 64, f"{3 * 64 * 64 * 8 / 1024:.0f} KB (L2)"),
]


def run(kernel, T, A, B, C, threads) -> float:
    bounds = np.linspace(0, N, threads + 1, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=threads) as pool:
        def once():
            futs = [pool.submit(kernel, A, B, C, int(bounds[w]), int(bounds[w + 1]), N, T)
                    for w in range(threads)]
            for f in futs:
                f.result()
        times = []
        for _ in range(REPEATS):
            s = time.perf_counter()
            once()
            times.append(time.perf_counter() - s)
    return statistics.median(times) * 1000.0


def main() -> None:
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Logical cores    : {MAX_THREADS}")
    print(f"Matrix           : {N}x{N} float64 ({N * N * 8 / 1e6:.1f} MB each) | median of {REPEATS} runs")
    print("Mode             : CPU tiled multiply (no dedicated GPU path)\n")

    # JIT warm-up on a tiny problem so compile time is not measured.
    # Size 64 so both tile sizes divide it (Numba does no bounds checking).
    w = np.ones((64, 64))
    wc = np.empty((64, 64))
    for _, k, T, _ in VARIANTS:
        k(w, w, wc, 0, 64, 64, T or 16)

    rng = np.random.default_rng(STUDENT_ID)
    A, B = rng.random((N, N)), rng.random((N, N))
    C = np.empty((N, N))
    reference = A @ B
    fmas = N ** 3

    rows = []
    for name, kernel, T, ws in VARIANTS:
        assert T == 0 or N % T == 0, "N must be a multiple of the tile size"
        t1 = run(kernel, T, A, B, C, 1)
        assert np.allclose(C, reference, rtol=1e-9), f"{name}: wrong result"
        tp = run(kernel, T, A, B, C, MAX_THREADS)
        assert np.allclose(C, reference, rtol=1e-9), f"{name}: wrong result (threaded)"
        rows.append((name, ws, t1, tp))

    best1 = min(r[2] for r in rows)
    header = (f"{'Variant':<14} | {'Tile working set':<22} | {'1 thread (ms)':>13} | "
              f"{'ns/FMA':>7} | {'Est. stall ns/FMA':>17} | {f'{MAX_THREADS} threads (ms)':>15} | {'GFLOP/s':>8}")
    print(header)
    print("-" * len(header))
    for name, ws, t1, tp in rows:
        ns = t1 * 1e6 / fmas
        stall = (t1 - best1) * 1e6 / fmas
        gflops = 2 * fmas / (tp / 1000) / 1e9
        print(f"{name:<14} | {ws:<22} | {t1:>13.1f} | {ns:>7.2f} | {stall:>17.2f} | {tp:>15.1f} | {gflops:>8.1f}")

    naive, best = rows[0], min(rows, key=lambda r: r[2])
    print("\n" + "=" * 70)
    print(f"Fastest single-thread variant: {best[0]}  ({best[2]:.1f} ms)")
    print(f"Non-tiled ijk vs {best[0]}: {naive[2] / best[2]:.1f}x slower single-thread, "
          f"{naive[3] / min(r[3] for r in rows):.1f}x slower with {MAX_THREADS} threads")
    print("Stall column = extra time per multiply-add vs the fastest variant (estimate,")
    print("not a hardware cache-miss counter).")
    print("=" * 70)


if __name__ == "__main__":
    main()