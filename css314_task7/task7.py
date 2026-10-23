import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"

import platform
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

STUDENT_ID = 230103071
N = 30_000_000
REPEATS = 5
BLOCK = 1 << 18                     # 262,144 elements: keeps temporaries in cache
MAX_CORES = os.cpu_count() or 1
CORES = sorted({1, 2, MAX_CORES})


def kernel_a(a, b, lo, hi):
    np.copyto(b[lo:hi], a[lo:hi])


def kernel_b(a, b, lo, hi):
    for s in range(lo, hi, BLOCK):
        e = min(s + BLOCK, hi)
        x = a[s:e]
        out = b[s:e]
        np.multiply(np.sin(x), np.cos(x), out=out)
        out += np.tan(x)


def bench(kernel, a, b, p) -> float:
    bounds = np.linspace(0, N, p + 1, dtype=np.int64)
    with ThreadPoolExecutor(max_workers=p) as pool:
        def once():
            list(pool.map(lambda w: kernel(a, b, int(bounds[w]), int(bounds[w + 1])), range(p)))
        once()                                      # warm-up (pages, threads)
        times = []
        for _ in range(REPEATS):
            s = time.perf_counter()
            once()
            times.append(time.perf_counter() - s)
    return statistics.median(times) * 1000.0


def main() -> None:
    print(f"Student ID       : {STUDENT_ID}")
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Max logical cores: {MAX_CORES}")
    print(f"Array            : {N:,} float64 ({N * 8 / 1e6:.0f} MB per array) | median of {REPEATS} runs\n")

    rng = np.random.default_rng(STUDENT_ID)
    a = rng.random(N)
    b = np.empty(N)
    np.copyto(b, a)                                 # commit pages before timing

    summary = {}
    for name, kernel in (("Kernel A (copy, memory-bound)", kernel_a),
                         ("Kernel B (sin*cos+tan, compute-bound)", kernel_b)):
        print(name)
        header = f"{'Cores':>6} | {'Time (ms)':>10} | {'Speedup':>8} | {'Efficiency':>10}" + \
                 (" | BW (GB/s)" if kernel is kernel_a else "")
        print(header)
        print("-" * len(header))
        t1 = None
        for p in CORES:
            t = bench(kernel, a, b, p)
            t1 = t1 or t
            sp = t1 / t
            eff = sp / p
            line = f"{p:>6} | {t:>10.1f} | {sp:>7.2f}x | {eff:>10.2f}"
            if kernel is kernel_a:
                line += f" | {2 * N * 8 / (t / 1000) / 1e9:>9.1f}"   # read + write traffic
            print(line)
            summary[name] = (sp, eff)
        print()

    (ea, eb) = (summary[k][1] for k in summary)
    print("=" * 70)
    if ea < eb and ea < 0.5:
        print(f"VERDICT: Kernel A (copy) stalled in scaling: efficiency {ea:.2f} at {MAX_CORES} cores")
        print(f"         vs {eb:.2f} for Kernel B. Adding cores did not add memory bandwidth:")
        print("         Kernel A is limited by memory-channel saturation, not by the CPU.")
    elif eb < ea and eb < 0.5:
        print(f"VERDICT: Kernel B (sin*cos+tan) scaled WORSE (efficiency {eb:.2f} vs {ea:.2f}).")
        print("         Unexpected: check for thermal/frequency limits or hybrid E-cores.")
    else:
        print(f"VERDICT: No clear memory saturation (efficiency A = {ea:.2f}, B = {eb:.2f}).")
    print("=" * 70)


if __name__ == "__main__":
    main()