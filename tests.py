"""Benchmarks Minimax (Memo), No-Memo, and Alpha-Beta Pruning with timeouts and CSV export."""

import time
import multiprocessing
import csv
from game import Game  # importing from our previous file

TIMEOUT_LIMIT = 100  # 5 minutes (or 100 seconds technically) - hard cutoff


def worker(m, n, k, mode, queue):
    """Executes the specific game search algorithm in a separate process."""
    g = Game(m, n, k)  # <--- Initialization happens here, BEFORE the timer

    start_time = time.perf_counter()  # <--- Timer starts ONLY for the move search

    # pick the algorithm based on mode string
    if mode == "minimax":
        g.get_best_move()
    elif mode == "pruning":
        g.pruning_best_move()
    elif mode == "nomemo":
        g.get_best_move_no_memo()

    end_time = time.perf_counter()  # <--- Timer stops

    # send time and node count back to parent process
    queue.put((end_time - start_time, g.node_count))


def run_with_timeout(m, n, k, mode):
    """Spawns a process to run the game logic and enforces a hard time limit."""
    queue = multiprocessing.Queue()
    # create independent process so we can kill it if it hangs
    p = multiprocessing.Process(target=worker, args=(m, n, k, mode, queue))

    p.start()
    p.join(timeout=TIMEOUT_LIMIT)  # wait for completion or timeout

    if p.is_alive():
        p.terminate()  # kill the process if it's still running
        p.join()  # cleanup resources
        return f">{TIMEOUT_LIMIT}s", "N/A"  # return timeout string

    if not queue.empty():
        elapsed, nodes = queue.get()
        # format time to 5 decimal places for precision
        return f"{elapsed:.5f}s", str(nodes)

    return "Error", "Error"


def run_suite():
    """Iterates through test parameters, runs benchmarks, and logs results to console and CSV."""
    # list of board configurations to test: (rows, cols, win_length)
    test_params = [
        (2, 2, 2),
        (2, 3, 2),
        (3, 2, 2),
        (3, 3, 2),
        (3, 3, 3),
        (4, 3, 3),
        (3, 4, 3),
        (4, 4, 3),
        (4, 4, 4),  # getting harder...
        (5, 4, 4),
        (5, 5, 4),  # this one might time out without pruning
    ]

    # Adjusting formatting to fit the new columns for console output
    header = (
        f"{'Board':<9} | "
        f"{'Memo Time':<11} | {'Memo Nodes':<11} | "
        f"{'NoMemo Time':<12} | {'NoMemo Nodes':<12} | "
        f"{'Pruning Time':<12} | {'Pruning Nodes':<13}"
    )

    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Open CSV file for writing results permanently
    with open("benchmark_results.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write CSV Header row
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
            # 1. Standard Minimax (With Memoization)
            mm_time, mm_nodes = run_with_timeout(m, n, k, "minimax")

            # 2. Minimax (No Memoization) - usually the slowest
            nm_time, nm_nodes = run_with_timeout(m, n, k, "nomemo")

            # 3. Alpha-Beta Pruning - usually the fastest
            ab_time, ab_nodes = run_with_timeout(m, n, k, "pruning")

            # print row to console so user sees progress
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

