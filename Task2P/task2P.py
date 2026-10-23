import math
import threading
import time

import numpy as np

STUDENT_ID = 230103071
POINTS = 40_000_000
THREADS = 4
BATCH = 4096         


def count_inside(pts):
    return int(np.count_nonzero(pts[:, 0] ** 2 + pts[:, 1] ** 2 <= 1.0))


def worker(idx, quota, get_points, hits):
    done = 0
    inside = 0
    while done < quota:
        n = min(BATCH, quota - done)
        pts = get_points(idx, n)
        inside += count_inside(pts)
        done += n
    hits[idx] = inside


def run(get_points):
    hits = [0] * THREADS
    quota = POINTS // THREADS
    threads = [threading.Thread(target=worker, args=(i, quota, get_points, hits))
               for i in range(THREADS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ms = (time.perf_counter() - start) * 1000
    pi = 4 * sum(hits) / POINTS
    return pi, abs(pi - math.pi), ms


shared_rng = np.random.default_rng(STUDENT_ID * 10)
lock = threading.Lock()


def shared_points(idx, n):
    with lock:
        return shared_rng.random((n, 2))

own_rngs = [np.random.default_rng(STUDENT_ID * 10 + i) for i in range(THREADS)]


def own_points(idx, n):
    return own_rngs[idx].random((n, 2))


if __name__ == "__main__":
    print("Student ID :", STUDENT_ID)
    print("Points     :", f"{POINTS:,}", "| threads:", THREADS)
    print("Seeds      :", [STUDENT_ID * 10 + i for i in range(THREADS)])
    print()

    run(own_points)         
    pi1, err1, t1 = run(shared_points)
    pi2, err2, t2 = run(own_points)

    print(f"Shared PRNG (locked)  : pi = {pi1:.8f}  error = {err1:.8f}  time = {t1:.1f} ms")
    print(f"Independent PRNGs     : pi = {pi2:.8f}  error = {err2:.8f}  time = {t2:.1f} ms")
    print()
    print(f"Speedup from removing the lock: {t1 / t2:.2f}x")
    