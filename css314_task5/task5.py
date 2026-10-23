import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103071
A, B = 2.5, 1.5
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
WORKERS = os.cpu_count() or 1
REFINE_POINTS = 24


def axpb(x: np.ndarray, y: np.ndarray) -> None:
    np.multiply(x, A, out=y)
    np.add(y, B, out=y)


def serial(x, y):
    axpb(x, y)


def parallel(pool, xs, ys):
    list(pool.map(axpb, xs, ys))


def timeit(fn, repeats: int) -> float:
    fn()                                           
    t = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return statistics.median(t) * 1e6              


def measure(pool, n: int) -> tuple[float, float]:
    rng = np.random.default_rng(STUDENT_ID)
    x = rng.random(n)
    y = np.empty(n)
    xs, ys = np.array_split(x, WORKERS), np.array_split(y, WORKERS)
    repeats = max(7, min(3000, int(3e6 / n)))
    t_ser = timeit(lambda: serial(x, y), repeats)
    t_par = timeit(lambda: parallel(pool, xs, ys), repeats)
    assert np.allclose(y, A * x + B)                
    return t_ser, t_par


def row(n, t_ser, t_par):
    sp = t_ser / t_par
    winner = "PARALLEL" if sp > 1 else "serial"
    return f"{n:>12,} | {t_ser:>12.1f} | {t_par:>12.1f} | {sp:>7.2f}x | {winner}"


def main() -> None:
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Workers (cores)  : {WORKERS}   |   kernel: y = {A}*x + {B}\n")
    header = f"{'N':>12} | {'Serial (us)':>12} | {'Parallel (us)':>13} | {'Speedup':>8} | Winner"

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        print("Step 1 - required sizes")
        print(header)
        print("-" * len(header))
        results = {}
        for n in SIZES:
            results[n] = measure(pool, n)
            print(row(n, *results[n]))

        def wins(n):
            return results[n][0] / results[n][1] > 1

        first_win = next((n for i, n in enumerate(SIZES)
                          if all(wins(m) for m in SIZES[i:])), None)

        print("\nStep 2 - refinement")
        if first_win is None:
            print("Parallel never beats serial at the tested sizes on this machine.")
            return
        if first_win == SIZES[0]:
            print(f"Parallel already wins at N = {first_win:,}; no crossover inside the tested range.")
            return

        lo = SIZES[SIZES.index(first_win) - 1]
        hi = first_win
        grid = sorted({int(v) for v in np.geomspace(lo, hi, REFINE_POINTS)})
        print(f"Bracket from Step 1: serial wins at N = {lo:,}, parallel wins at N = {hi:,}")
        print(header)
        print("-" * len(header))
        for n in grid:
            results[n] = measure(pool, n)
            print(row(n, *results[n]))

    all_n = sorted(results)
    stable = next(n for i, n in enumerate(all_n) if all(wins(m) for m in all_n[i:]))
    last_slower = max(n for n in all_n if n < stable)
    print("\n" + "=" * 70)
    print(f"RESULT: multi-threading is SLOWER than serial up to N = {last_slower:,}")
    print(f"        and FASTER than serial from N = {stable:,} upward.")
    print(f"        Crossover range: {last_slower:,} < N <= {stable:,}")
    print("=" * 70)


if __name__ == "__main__":
    main()
