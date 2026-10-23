import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

STUDENT_ID = 230103071
SIZES = [100_000, 1_000_000, 10_000_000]
RUNS = 5


def child(conn):
    while True:
        msg = conn.recv()
        if msg[0] == "stop":
            break
        if msg[0] == "pipe":
            arr = msg[1]                 
            conn.send(arr[-1])           
        elif msg[0] == "shm":
            name, n = msg[1], msg[2]
            try:
                shm = shared_memory.SharedMemory(name=name, track=False)
            except TypeError:           
                shm = shared_memory.SharedMemory(name=name)
            arr = np.ndarray((n,), dtype=np.float64, buffer=shm.buf)
            last = float(arr[-1])        # touch the data, no copy happened
            del arr
            shm.close()
            conn.send(last)


def median(values):
    values = sorted(values)
    return values[len(values) // 2]


def path_a(conn, data):
    start = time.perf_counter()
    conn.send(("pipe", data))
    conn.recv()
    return (time.perf_counter() - start) * 1000


def path_b(conn, data):
    start = time.perf_counter()
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    view = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    view[:] = data                                  
    filled = time.perf_counter()

    conn.send(("shm", shm.name, len(data)))
    conn.recv()
    end = time.perf_counter()

    del view
    shm.close()
    shm.unlink()
    handoff = (end - filled) * 1000
    total = (end - start) * 1000
    return handoff, total


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parent_conn, child_conn = mp.Pipe()
    p = mp.Process(target=child, args=(child_conn,))
    p.start()

    rng = np.random.default_rng(STUDENT_ID)

    print("Student ID:", STUDENT_ID)
    print("Median of", RUNS, "runs, float64 arrays, one extra process\n")
    print(f"{'N':>12} | {'MB':>6} | {'A: pipe (ms)':>12} | {'B: shm handoff (ms)':>19} | {'B: shm + fill (ms)':>18} | {'A / B':>7}")
    print("-" * 90)

    for n in SIZES:
        data = rng.random(n)

        path_a(parent_conn, data)
        path_b(parent_conn, data)

        a_times = []
        b_hand = []
        b_total = []
        for _ in range(RUNS):
            a_times.append(path_a(parent_conn, data))
            h, t = path_b(parent_conn, data)
            b_hand.append(h)
            b_total.append(t)

        a = median(a_times)
        bh = median(b_hand)
        bt = median(b_total)
        print(f"{n:>12,} | {data.nbytes / 1e6:>6.1f} | {a:>12.2f} | {bh:>19.3f} | {bt:>18.2f} | {a / bh:>6.0f}x")

    parent_conn.send(("stop",))
    p.join()