"""Benchmarks Minimax vs Alpha-Beta Pruning with timeouts and CSV export."""

import time
import multiprocessing
import csv
from game import Game

TIMEOUT_LIMIT = 300  # 5 minutes


def worker(m, n, k, mode, queue):
    """Executes the specific game search algorithm in a separate process."""
    g = Game(m, n, k)
    if mode == "minimax":
        g.get_best_move()
    elif mode == "pruning":
        g.pruning_best_move()
    queue.put(g.node_count)


def run_with_timeout(m, n, k, mode):
    """Spawns a process to run the game logic and enforces a hard time limit."""
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=worker, args=(m, n, k, mode, queue))

    start = time.perf_counter()
    p.start()
    p.join(timeout=TIMEOUT_LIMIT)

    if p.is_alive():
        p.terminate()
        p.join()
        return f">{TIMEOUT_LIMIT}s", "N/A"

    end = time.perf_counter()
    elapsed = end - start
    nodes = queue.get() if not queue.empty() else "Error"
    return f"{elapsed:.5f}s", str(nodes)


def run_suite():
    """Iterates through test parameters, runs benchmarks, and logs results to console and CSV."""
    test_params = [
        (3, 3, 3),
        (3, 4, 3),
        (4, 3, 3),
        (4, 4, 3),
        (3, 5, 3),
        (4, 5, 3),
        (4, 4, 4),
        (5, 4, 4),
    ]

    header = f"{'Board (m,n,k)':<15} | {'Minimax Time':<14} | {'Minimax Nodes':<14} | {'Pruning Time':<14} | {'Pruning Nodes':<14}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Open CSV file for writing
    with open("benchmark_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write CSV Header
        writer.writerow(
            ["m", "n", "k", "Minimax Time", "Minimax Nodes", "Pruning Time", "Pruning Nodes"]
        )

        for m, n, k in test_params:
            mm_time, mm_nodes = run_with_timeout(m, n, k, "minimax")
            ab_time, ab_nodes = run_with_timeout(m, n, k, "pruning")

            print(
                f"({m},{n},{k})".ljust(15)
                + f" | {mm_time:<14} | {mm_nodes:<14} | {ab_time:<14} | {ab_nodes:<14}"
            )

            # Write row to CSV
            writer.writerow([m, n, k, mm_time, mm_nodes, ab_time, ab_nodes])


if __name__ == "__main__":
    run_suite()
