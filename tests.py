"""Benchmarks Minimax (Memo), No-Memo, and Alpha-Beta Pruning with timeouts and CSV export."""

import time
import multiprocessing
import csv
from game import Game

TIMEOUT_LIMIT = 100  # 5 minutes


def worker(m, n, k, mode, queue):
    """Executes the specific game search algorithm in a separate process."""
    g = Game(m, n, k)
    if mode == "minimax":
        g.get_best_move()
    elif mode == "pruning":
        g.pruning_best_move()
    elif mode == "nomemo":
        g.get_best_move_no_memo()
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

    # Adjusting formatting to fit the new columns
    header = (
        f"{'Board':<9} | "
        f"{'Memo Time':<11} | {'Memo Nodes':<11} | "
        f"{'NoMemo Time':<12} | {'NoMemo Nodes':<12} | "
        f"{'Pruning Time':<12} | {'Pruning Nodes':<13}"
    )

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Open CSV file for writing
    with open("benchmark_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write CSV Header
        writer.writerow(
            [
                "m",
                "n",
                "k",
                "Memo Time",
                "Memo Nodes",
                "NoMemo Time",
                "NoMemo Nodes",
                "Pruning Time",
                "Pruning Nodes",
            ]
        )

        for m, n, k in test_params:
            # 1. Standard Minimax (With Memo)
            mm_time, mm_nodes = run_with_timeout(m, n, k, "minimax")

            # 2. Minimax (No Memo)
            nm_time, nm_nodes = run_with_timeout(m, n, k, "nomemo")

            # 3. Alpha-Beta Pruning
            ab_time, ab_nodes = run_with_timeout(m, n, k, "pruning")

            print(
                f"({m},{n},{k})".ljust(9)
                + " | "
                + f"{mm_time:<11} | {mm_nodes:<11} | "
                + f"{nm_time:<12} | {nm_nodes:<12} | "
                + f"{ab_time:<12} | {ab_nodes:<13}"
            )

            # Write row to CSV
            writer.writerow([m, n, k, mm_time, mm_nodes, nm_time, nm_nodes, ab_time, ab_nodes])


if __name__ == "__main__":
    run_suite()
