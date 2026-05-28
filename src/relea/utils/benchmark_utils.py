from typing import Callable

import time

def benchmark_time(fn: Callable):
    def wrapper(*args, **kwargs):
        elapsed_times = []
        print("Starting Benchmarking...")
        for i in range(15):
            start = time.time()
            fn(*args, **kwargs)
            end = time.time()
            elapsed_time = end - start
            elapsed_times.append(elapsed_time)
            print(f"\t[{i + 1}/{15}] Elapsed Time: {elapsed_time}")

        elapsed_times = elapsed_times[1:]   # discard first because it has obvious overheads.
        print(f"Mean Elapsed Time: {sum(elapsed_times) / len(elapsed_times)}s.")
    return wrapper
