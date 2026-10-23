
import platform
import queue
import statistics
import threading
import time

N_ITEMS = 1000
WORKERS = 4
UNIT = 0.0001          # seconds per cost level
REPEATS = 5

WORKLOADS = {
    "spec      cost = 0.0001*(i % 20)": lambda i: UNIT * (i % 20),
    "clustered cost = 0.0001*(i // 50)": lambda i: UNIT * (i // 50),
}


def run(strategy: str, chunk: int, cost) -> tuple[float, float]:
    """Return (total wall time ms, finish-time delta ms between first and last worker)."""
    finish = [0.0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)

    if strategy == "static":
        size = N_ITEMS // WORKERS
        blocks = [range(w * size, (w + 1) * size) for w in range(WORKERS)]
    else:
        q = queue.Queue()
        for s in range(0, N_ITEMS, chunk):
            q.put(range(s, min(s + chunk, N_ITEMS)))

    def worker(w: int) -> None:
        barrier.wait()
        t0 = time.perf_counter()
        if strategy == "static":
            for i in blocks[w]:
                time.sleep(cost(i))
        else:
            while True:
                try:
                    items = q.get_nowait()
                except queue.Empty:
                    break
                for i in items:
                    time.sleep(cost(i))
        finish[w] = time.perf_counter() - t0

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    barrier.wait()
    for t in threads:
        t.join()
    return max(finish) * 1000.0, (max(finish) - min(finish)) * 1000.0


def main() -> None:
    print(f"OS               : {platform.system()} {platform.release()}")
    print(f"Items / workers  : {N_ITEMS} / {WORKERS}")
    print(f"Per-run reports  : median of {REPEATS} runs\n")

    configs = [("A: Static (4 blocks)", "static", 0),
               ("B: Dynamic chunk=1", "dynamic", 1),
               ("B: Dynamic chunk=5", "dynamic", 5)]

    for name, cost in WORKLOADS.items():
        ideal = sum(cost(i) for i in range(N_ITEMS)) / WORKERS * 1000.0
        print(f"=== Workload: {name}   (ideal = total cost / {WORKERS} = {ideal:.1f} ms)")
        header = f"{'Strategy':<22} | {'Total (ms)':>10} | {'Finish delta (ms)':>17}"
        print(header)
        print("-" * len(header))
        for label, strat, chunk in configs:
            runs = [run(strat, chunk, cost) for _ in range(REPEATS)]
            total = statistics.median(r[0] for r in runs)
            delta = statistics.median(r[1] for r in runs)
            print(f"{label:<22} | {total:>10.1f} | {delta:>17.1f}")
        print()


if __name__ == "__main__":
    main()

