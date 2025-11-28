import time
import math
import platform
import timeit
import sys

import numba

# --- Try import Numba for compile time optimization ---
try:
    from numba import njit
    import numpy as np
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print('Note: Numpy/Numba not installed. Numba benchmarks will be skipped')



class Game:
    """
    Minimax Game Class
    Using a bitboard to represent the board state
    """

    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k

        # The State: Two integers representing the board
        self.mask_x = 0
        self.mask_o = 0

        # Pre Calculation of all Winning Masks (Integers)
        self.win_masks = self._generate_win_masks()
        # Full Board mask for detecting draws
        self.full_board_mask = (1 << (self.m * self.n)) - 1

        # Generate Search Order (Center-Out Indices)
        center_r, center_c = self.m // 2, self.n // 2
        coords = [(r, c) for r in range(self.m) for c in range(self.n)]
        coords.sort(key=lambda x: abs(x[0] - center_r) + abs(x[1] - center_c))

        # Store a flattened bit indices (i.e 0 to 15 for 4x4)
        self.search_indices = [r * self.n + c for r, c in coords]

        # Memory for Memoization
        self.memo = {}

    def _generate_win_masks(self):
        """Generates a list of 64-bit integers representing all winning lines"""
        masks = []
        get_idx = lambda r, c: r * self.n + c

        # Horizontal
        for r in range(self.m):
            for c in range(self.n - self.k + 1):
                mask = 0
                for i in range(self.k): mask |= (1 << get_idx(r, c + i))
                masks.append(mask)
        # Vertical
        for r in range(self.m - self.k + 1):
            for c in range(self.n):
                mask = 0
                for i in range(self.k): mask |= (1 << get_idx(r + i, c))
                masks.append(mask)
        # Diagonal
        for r in range(self.m - self.k + 1):
            for c in range(self.n - self.k + 1):
                mask = 0
                for i in range(self.k): mask |= (1 << get_idx(r + i, c + i))
                masks.append(mask)
        # Anti-Diagonal
        for r in range(self.m - self.k + 1):
            for c in range(self.k - 1, self.n):
                mask = 0
                for i in range(self.k): mask |= (1 << get_idx(r + i, c - i))
                masks.append(mask)
        return masks

    def drawboard(self):
        """Decodes the integers back into a visual grid for the user."""
        print("  " + " ".join([str(i) for i in range(self.n)]))
        for r in range(self.m):
            row_str = f"{r} "
            for c in range(self.n):
                idx = r * self.n + c
                if (self.mask_x >> idx) & 1:
                    row_str += "X "
                elif (self.mask_o >> idx) & 1:
                    row_str += "O "
                else:
                    row_str += ". "
            print(row_str)
        print()

    def check_winner(self, mx, mo):
        """Checks if mask_x (mx) or mask_o (mo) matches a winning pattern."""
        for wm in self.win_masks:
            if (mx & wm) == wm: return 'X'
            if (mo & wm) == wm: return 'O'
        if (mx | mo) == self.full_board_mask: return 'Draw'
        return None

    # =================================================================
    # METHOD 1: PURE MINIMAX (Bitboard)
    # =================================================================
    def run_pure(self):
        """Wrapper to start Pure Minimax."""
        return self._max_pure(self.mask_x, self.mask_o)

    def _max_pure(self, mx, mo):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        # Iterate through bit indices
        # Note: Unordered search for Pure compairson
        for i in range(self.m * self.n):
            # Check if bit 'i' is 0 in both masks
            if not (( mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._min_pure(mx | (1 << i), mo)
                if v > max_v: max_v = v; best_move = i
        return (max_v, best_move)

    def _min_pure(self, mx, mo):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        # Iterate through bit indices
        # Note: Unordered search for Pure compairson
        for i in range(self.m * self.n):
            # Check if bit 'i' is 0 in both masks
            if not ((mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._max_pure(mx, mo | (1 << i))
                if v < min_v: min_v = v; best_move = i
        return (min_v, best_move)

    # =================================================================
    # METHOD 2: PURE MINIMAX Extension with Center out order (Bitboard)
    # =================================================================

    def run_pure_ordered(self):
        """Wrapper to start Pure Minimax Ordered."""
        return self._max_pure_ordered(self.mask_x, self.mask_o)

    def _max_pure_ordered(self, mx, mo):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        # Iterate through bit indices
        for i in self.search_indices:
            # Check if bit 'i' is 0 in both masks
            if not ((mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._min_pure_ordered(mx | (1 << i), mo)
                if v > max_v: max_v = v; best_move = i
        return (max_v, best_move)

    def _min_pure_ordered(self, mx, mo):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        # Iterate through bit indices
        for i in self.search_indices:
            # Check if bit 'i' is 0 in both masks
            if not ((mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._max_pure_ordered(mx, mo | (1 << i))
                if v < min_v: min_v = v; best_move = i
        return (min_v, best_move)

    # =================================================================
    # METHOD 3: PURE MINIMAX Extension with Center out order + Memory (Bitboard)
    # =================================================================

    def run_pure_ordered_and_memory(self):
        """Wrapper to start Pure Minimax Ordered."""
        return self._max_pure_ordered(self.mask_x, self.mask_o)

    def _max_pure_ordered_and_memory(self, mx, mo):
        # Hash Lookup O(1)
        state_key = (mx, mo)
        if state_key in self.memo: return self.memo[state_key]

        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        # Iterate through bit indices
        for i in self.search_indices:
            # Check if bit 'i' is 0 in both masks
            if not ((mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._min_pure_ordered_and_memory(mx | (1 << i), mo)
                if v > max_v: max_v = v; best_move = i
        self.memo[state_key] = (max_v, best_move)
        return (max_v, best_move)

    def _min_pure_ordered_and_memory(self, mx, mo):
        # Hash Lookup O(1)
        state_key = (mx, mo)
        if state_key in self.memo: return self.memo[state_key]

        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        # Iterate through bit indices
        for i in self.search_indices:
            # Check if bit 'i' is 0 in both masks
            if not ((mx | mo) >> i) & 1:
                # Recurse pass new mask (mx| 1<<i)
                v, _ = self._max_pure_ordered_and_memory(mx, mo | (1 << i))
                if v < min_v: min_v = v; best_move = i
        self.memo[state_key] = (min_v, best_move)
        return (min_v, best_move)


    # =================================================================
    # METHOD 4: BASIC ALPHA-BETA (Bitboard)
    # =================================================================
    def run_ab_basic(self):
        return self._max_ab_basic(self.mask_x, self.mask_o, -math.inf, math.inf)

    def _max_ab_basic(self, mx, mo, alpha, beta):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        for i in range(self.m * self.n):
            if not ((mx | mo) >> i) & 1:
                v, _ = self._min_ab_basic(mx | (1 << i), mo, alpha, beta)
                if v > max_v:
                    max_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if max_v >= beta: return (max_v, best_move)
                alpha = max(alpha, max_v)
        return (max_v, best_move)

    def _min_ab_basic(self, mx, mo, alpha, beta):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        for i in range(self.m * self.n):
            if not ((mx | mo) >> i) & 1:
                v, _ = self._max_ab_basic(mx, mo | (1 << i), alpha, beta)
                if v < min_v:
                    min_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if min_v <= alpha: return (min_v, best_move)
                beta = min(beta, min_v)
        return (min_v, best_move)


    # =================================================================
    # METHOD 5: Ordered (Center out) ALPHA-BETA (Bitboard)
    # =================================================================
    def run_ab_ordered(self):
        return self._max_ab_ordered(self.mask_x, self.mask_o, -math.inf, math.inf)

    def _max_ab_ordered(self, mx, mo, alpha, beta):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        for i in self.search_indices:
            if not ((mx | mo) >> i) & 1:
                v, _ = self._min_ab_ordered(mx | (1 << i), mo, alpha, beta)
                if v > max_v:
                    max_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if max_v >= beta: return (max_v, best_move)
                alpha = max(alpha, max_v)
        return (max_v, best_move)

    def _min_ab_ordered(self, mx, mo, alpha, beta):
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        for i in self.search_indices:
            if not ((mx | mo) >> i) & 1:
                v, _ = self._max_ab_ordered(mx, mo | (1 << i), alpha, beta)
                if v < min_v:
                    min_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if min_v <= alpha: return (min_v, best_move)
                beta = min(beta, min_v)
        return (min_v, best_move)


    # =================================================================
    # METHOD 6: Ordered (Center out) + Memory ALPHA-BETA (Bitboard)
    # Called Max and Min for Assingment as best models (without numba)
    # =================================================================
    def run_ab_ordered_and_memory(self):
        return self.max(self.mask_x, self.mask_o, -math.inf, math.inf)

    def max(self, mx, mo, alpha, beta):

        state_key = (mx, mo)
        if state_key in self.memo: return self.memo[state_key]

        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        max_v = -math.inf
        best_move = None

        for i in self.search_indices:
            if not ((mx | mo) >> i) & 1:
                v, _ = self.min(mx | (1 << i), mo, alpha, beta)
                if v > max_v:
                    max_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if max_v >= beta: return (max_v, best_move)
                alpha = max(alpha, max_v)

        self.memo[state_key] = (max_v, best_move)
        return (max_v, best_move)

    def min(self, mx, mo, alpha, beta):

        state_key = (mx, mo)
        if state_key in self.memo: return self.memo[state_key]
        for wm in self.win_masks:
            if (mx & wm) == wm: return (1, None)
            if (mo & wm) == wm: return (-1, None)
        if (mx | mo) == self.full_board_mask: return (0, None)

        min_v = math.inf
        best_move = None

        for i in self.search_indices:
            if not ((mx | mo) >> i) & 1:
                v, _ = self.max(mx, mo | (1 << i), alpha, beta)
                if v < min_v:
                    min_v = v
                    best_move = i
                # Alpha-Beta Pruning
                if min_v <= alpha: return (min_v, best_move)
                beta = min(beta, min_v)

        self.memo[state_key] = (min_v, best_move)
        return (min_v, best_move)


# =================================================================
# NUMBA JIT (Standalone, Functional Style)
# =================================================================
if NUMBA_AVAILABLE:
    # Note we cant use Dictionary Hash for Memory here sowe use a ZOBRIST Hash
    # Memory Constants
    TT_SIZE = 1_000_003
    ZOBRIST_TABLE = np.random.randint(1, 2 ** 62, size=(3, 64), dtype=np.int64)

    @njit
    def compute_hash_numba(mask_x, mask_o, m, n):
        """Zobrist Hash Calculation for Numba"""
        h = 0
        for i in range(m * n):
            if (mask_x >> i) & 1:
                h ^= ZOBRIST_TABLE[1, i]
            elif (mask_o >> i) & 1:
                h ^= ZOBRIST_TABLE[2, i]
        return h


    @njit
    def solve_numba(mask_x, mask_o, m, n, win_masks, full_mask, search_indices, is_max, alpha, beta, tt_table,
                    tt_flags):
        """
        Numba-Optimized Solver.
        Uses Zobrist Array for Memoization, Bitwise logic for Moves, and Numpy Arrays for Masks.
        """
        # 1. Zobrist Lookup
        h = compute_hash_numba(mask_x, mask_o, m, n)
        idx = h % TT_SIZE
        if tt_flags[idx] == h:
            return tt_table[idx]

        # 2. Terminal Check
        for wm in win_masks:
            if (mask_x & wm) == wm: return 1.0
            if (mask_o & wm) == wm: return -1.0
        if (mask_x | mask_o) == full_mask: return 0.0

        # 3. Recursion
        if is_max:
            best = -1000.0
            for i in search_indices:
                if not ((mask_x | mask_o) >> i) & 1:
                    val = solve_numba(mask_x | (1 << i), mask_o, m, n, win_masks, full_mask, search_indices, False,
                                      alpha, beta, tt_table, tt_flags)
                    if val > best: best = val
                    if val >= beta:
                        tt_table[idx] = best
                        tt_flags[idx] = h
                        return best
                    if val > alpha: alpha = val

            tt_table[idx] = best
            tt_flags[idx] = h
            return best
        else:
            best = 1000.0
            for i in search_indices:
                if not ((mask_x | mask_o) >> i) & 1:
                    val = solve_numba(mask_x, mask_o | (1 << i), m, n, win_masks, full_mask, search_indices, True,
                                      alpha, beta, tt_table, tt_flags)
                    if val < best: best = val
                    if val <= alpha:
                        tt_table[idx] = best
                        tt_flags[idx] = h
                        return best
                    if val < beta: beta = val

            tt_table[idx] = best
            tt_flags[idx] = h
            return best


    @njit
    def get_initial_hash(mask_x, mask_o, m, n):
        h = 0
        for i in range(m * n):
            if (mask_x >> i) & 1:
                h ^= ZOBRIST_TABLE[1, i]
            elif (mask_o >> i) & 1:
                h ^= ZOBRIST_TABLE[2, i]
        return h


    @njit
    def solve_numba_with_updating_hash(mask_x, mask_o, current_hash, win_masks, full_mask, search_indices, is_max, alpha,
                                    beta, tt_table, tt_states):
        """
        Collision-Proof Solver.
        Stores the ACTUAL board state (mask_x, mask_o) in memory to guarantee correctness.
        """

        #  Direct State Lookup
        # We still use hash to find the "bucket" (Index)
        idx = current_hash % TT_SIZE

        # Check if the stored board matches the current board exactly
        if tt_states[idx, 0] == mask_x and tt_states[idx, 1] == mask_o:
            return tt_table[idx]

        #  Terminal Check (Same as before)
        for wm in win_masks:
            if (mask_x & wm) == wm: return 1.0
            if (mask_o & wm) == wm: return -1.0
        if (mask_x | mask_o) == full_mask: return 0.0

        # Recursion
        if is_max:
            best = -1000.0
            for i in search_indices:
                if not ((mask_x | mask_o) >> i) & 1:
                    # Update Hash Incrementally
                    new_hash = current_hash ^ ZOBRIST_TABLE[1, i]

                    val = solve_numba_with_updating_hash(mask_x | (1 << i), mask_o, new_hash, win_masks, full_mask,
                                                      search_indices, False, alpha, beta, tt_table, tt_states)

                    if val > best: best = val
                    if val >= beta:
                        # STORE THE FULL STATE
                        tt_table[idx] = best
                        tt_states[idx, 0] = mask_x
                        tt_states[idx, 1] = mask_o
                        return best
                    if val > alpha: alpha = val

            tt_table[idx] = best
            tt_states[idx, 0] = mask_x
            tt_states[idx, 1] = mask_o
            return best
        else:
            best = 1000.0
            for i in search_indices:
                if not ((mask_x | mask_o) >> i) & 1:
                    # Update Hash Incrementally
                    new_hash = current_hash ^ ZOBRIST_TABLE[2, i]

                    val = solve_numba_with_updating_hash(mask_x, mask_o | (1 << i), new_hash, win_masks, full_mask,
                                                      search_indices, True, alpha, beta, tt_table, tt_states)

                    if val < best: best = val
                    if val <= alpha:
                        tt_table[idx] = best
                        tt_states[idx, 0] = mask_x
                        tt_states[idx, 1] = mask_o
                        return best
                    if val < beta: beta = val

            tt_table[idx] = best
            tt_states[idx, 0] = mask_x
            tt_states[idx, 1] = mask_o
            return best


# =================================================================
# BENCHMARK RUNNER
# =================================================================
def run_benchmark():
    m, n, k = 3, 3, 3
    interpreter = platform.python_implementation()
    print(f"\nBENCHMARK: {m}x{n} Board (k={k}) | Running on: {interpreter}")
    print("=" * 70)
    print(f"{'Method':<30} | {'Type':<20} | {'Time (s)':<10}")
    print("-" * 70)
    #
    # 1. Pure Minimax (Bitboard)
    # Using 3x3 for pure minimax to avoid hanging
    g_small = Game(3, 3, 3)
    s = time.time()
    g_small.run_pure()
    print(f"{'1. Pure Minimax (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

    # 2. Pure Minimax Ordered
    g_small = Game(3, 3, 3)
    s = time.time()
    g_small.run_pure_ordered()
    print(f"{'2. Pure Ordered (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

    # 3. Pure Minimax Ordered + Memory
    g_small = Game(3, 3, 3)
    s = time.time()
    g_small.run_pure_ordered_and_memory()
    print(f"{'3. Pure Ord+Memo (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

    # 4. Basic Alpha Beta
    g = Game(m, n, k)
    s = time.time()
    g.run_ab_basic()
    print(f"{'4. Basic Alpha-Beta':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

    # 5. Ordered Alpha Beta
    g = Game(m, n, k)
    s = time.time()
    g.run_ab_ordered()
    print(f"{'5. Ordered Alpha-Beta':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")
    #
    # 6. The BEST Version (Bitboard + Memo + Ordered)
    g = Game(m, n, k)
    s = time.time()
    g.run_ab_ordered_and_memory()
    print(f"{'6. BEST (Ord+Memo+AB)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

    # 7. Numba
    if NUMBA_AVAILABLE and interpreter != 'PyPy':
        # Setup Data for Numba
        win_masks_np = np.array(g.win_masks, dtype=np.int64)
        search_indices_np = np.array(g.search_indices, dtype=np.int64)
        full_mask = (1 << (m * n)) - 1

        # Setup Memory
        tt_table = np.zeros(TT_SIZE, dtype=np.float64)  # Float because Numba fn returns floats
        tt_flags = np.full(TT_SIZE, -1, dtype=np.int64)

        # Warmup (Run ones to get it to complie)
        solve_numba(0, 0, m, n, win_masks_np, full_mask, search_indices_np, True, -100.0, 100.0, tt_table, tt_flags)

        # Rest Memory and Run
        tt_table.fill(0)
        tt_flags.fill(-1)
        s = time.time()
        solve_numba(0, 0, m, n, win_masks_np, full_mask, search_indices_np, True, -100.0, 100.0, tt_table, tt_flags)
        print(f"{'7. Numba version of best':<30} | {'Machine Code':<20} | {time.time() - s:.5f}")


        # --- NEW: STATE STORE (2D Array) ---
        # Dimensions: [1,000,003 rows x 2 columns]
        # 18 bits per slot  ( double the above with 8 bytes)
        # Column 0 = mask_x, Column 1 = mask_o
        tt_states = np.zeros((TT_SIZE, 2), dtype=np.int64)
        tt_states.fill(-1)  # Init with -1 so empty board (0,0) doesn't collide

        root_hash = get_initial_hash(0, 0, m, n)

        # Warmup (Run ones to get it to complie)
        solve_numba_with_updating_hash(0, 0, root_hash, win_masks_np, full_mask, search_indices_np, True, -100.0, 100.0,
                                    tt_table, tt_states)

        # Rest memory and Run
        tt_table.fill(0)
        tt_states.fill(-1)
        s = time.time()
        solve_numba_with_updating_hash(0, 0, root_hash, win_masks_np, full_mask, search_indices_np, True, -100.0, 100.0,
                                    tt_table, tt_states)
        print(f"{'8. Numba State Check':<30} | {'Machine Code':<20} | {time.time() - s:.5f}")



def minimax_time_comparison():
    test_configs = [
        (2, 2, 2),
        (3, 2, 2),
        (3, 3, 2),
        (3, 3, 3),
    ]

    interpreter = platform.python_implementation()
    print(f"\nBENCHMARK: accross (m,n,k) compose | Running on: {interpreter}")
    print("=" * 70)
    print(f"(m,n,k) | {'Method':<30} | {'Type':<20} | {'Time (s)':<10}")
    print("-" * 70)

    for m, n, k in test_configs:

        # 1. Pure Minimax (Bitboard)
        # Using 3x3 for pure minimax to avoid hanging
        g_small = Game(m, n, k)
        s = time.time()
        g_small.run_pure()
        print(f"{(m,n,k)} | {'1. Pure Minimax (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

        # 2. Pure Minimax Ordered
        g_small = Game(m, n, k)
        s = time.time()
        g_small.run_pure_ordered()
        print(f"{(m,n,k)} | {'2. Pure Ordered (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

        # 3. Pure Minimax Ordered + Memory
        g_small = Game(m, n, k)
        s = time.time()
        g_small.run_pure_ordered_and_memory()
        print(f"{(m,n,k)} | {'3. Pure Ord+Memo (3x3)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")


def alpha_beta_time_comparison():
    test_configs = [
        (2, 2, 2),
        (3, 2, 2),
        (3, 3, 2),
        (3, 3, 3),
        (4, 3, 3),
        (4, 4, 3),
        (4, 4, 4),
    ]

    interpreter = platform.python_implementation()
    print(f"\nBENCHMARK: accross (m,n,k) compose | Running on: {interpreter}")
    print("=" * 70)
    print(f"(m,n,k) | {'Method':<30} | {'Type':<20} | {'Time (s)':<10}")
    print("-" * 70)

    for m, n, k in test_configs:
        # 4. Basic Alpha Beta
        g = Game(m, n, k)
        s = time.time()
        g.run_ab_basic()
        print(f"{(m,n,k)} | {' Basic Alpha-Beta':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")

        # 5. Ordered Alpha Beta
        g = Game(m, n, k)
        s = time.time()
        g.run_ab_ordered()
        print(f"{(m,n,k)} | {' Ordered Alpha-Beta':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")
        #
        # 6. The BEST Version (Bitboard + Memo + Ordered)
        g = Game(m, n, k)
        s = time.time()
        g.run_ab_ordered_and_memory()
        print(f"{(m,n,k)} | {' BEST (Ord+Memo+AB)':<30} | {'Bitboard':<20} | {time.time() - s:.5f}")




if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    run_benchmark()

    minimax_time_comparison()
    alpha_beta_time_comparison()