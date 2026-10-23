import platform
import sys
import threading
import time

THREADS = 4
PER_THREAD = 2_000_000
EXPECTED = THREADS * PER_THREAD

counter = 0
lock = threading.Lock()


def _switch_point() -> None:
    pass


def naive(n):
    global counter
    for _ in range(n):
        counter += 1


def racy(n):
    global counter
    for _ in range(n):
        tmp = counter
        _switch_point()
        counter = tmp + 1


def locked(n):
    global counter
    for _ in range(n):
        with lock:
            tmp = counter
            _switch_point()
            counter = tmp + 1


def run(fn) -> tuple[int, float]:
    global counter
    counter = 0
    threads = [threading.Thread(target=fn, args=(PER_THREAD,)) for _ in range(THREADS)]
    start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter, (time.perf_counter() - start) * 1000.0


def report(name, value, ms):
    lost = EXPECTED - value
    print(f"{name:<34}: actual = {value:>9,} | lost = {lost:>9,} ({lost / EXPECTED * 100:6.2f}%) | {ms:>9.1f} ms")


def main() -> None:
    print(f"OS / Python      : {platform.system()} {platform.release()} / {platform.python_version()}")
    gil = getattr(sys, "_is_gil_enabled", lambda: True)()
    print(f"GIL enabled      : {gil}")
    print(f"Threads x incr.  : {THREADS} x {PER_THREAD:,}")
    print(f"Expected final   : {EXPECTED:,}\n")

    v1, t1 = run(naive)
    report("1. naive counter += 1, no lock", v1, t1)
    v2, t2 = run(racy)
    report("2. racy read-call-write, no lock", v2, t2)
    v3, t3 = run(locked)
    report("3. same logic WITH threading.Lock", v3, t3)

    print()
    assert v3 == EXPECTED, "locked version must be exact"
    print(f"Lock penalty vs racy (same logic)   : +{t3 - t2:,.1f} ms  ({t3 / t2:.2f}x slower)")
    print(f"Lock penalty vs naive counter += 1  : +{t3 - t1:,.1f} ms  ({t3 / t1:.2f}x slower)")
    print(f"Locked result correct: {v3:,} == {EXPECTED:,}")


if __name__ == "__main__":
    main()