"""Implements a generalized (m,n,k) game solver using bitboards and pruning."""

import time


class Game:
    """Bitboard game engine with Memoization and Alpha-Beta Pruning."""

    def __init__(self, m: int, n: int, k: int) -> None:
        """Initialize the game parameters, bitmasks, and winning shift directions."""
        self.m = m
        self.n = n
        self.k = k
        self.w = n + 1
        self.max_board = 0
        self.min_board = 0
        self.memo = {}

        shifts = []
        count = 1
        while count < k:
            step = min(count, k - count)
            shifts.append(step)
            count += step
        self.check_shifts = tuple(shifts)
        self.directions = (1, self.w, self.w + 1, self.w - 1)

        valid_indices = [r * self.w + c for r in range(m) for c in range(n)]
        self.full_mask = sum(1 << i for i in valid_indices)
        self.move_masks = tuple((1 << i) for i in valid_indices)

    def make_max_move(self, row: int, col: int) -> None:
        """Place a piece for the maximizing player at the given coordinates."""
        self.max_board |= 1 << (row * self.w + col)

    def make_min_move(self, row: int, col: int) -> None:
        """Place a piece for the minimizing player at the given coordinates."""
        self.min_board |= 1 << (row * self.w + col)

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if the cell at the specified row and column is empty."""
        return not ((self.max_board | self.min_board) & (1 << (row * self.w + col)))

    def check_win(self, board: int) -> bool:
        """Determine if the provided bitboard contains a winning line of length k."""
        for d in self.directions:
            b = board
            for step in self.check_shifts:
                b &= b >> (d * step)
                if not b:
                    break
            if b:
                return True
        return False

    def get_best_move(self):
        """Calculate the optimal game value using standard Minimax recursion."""
        return self.max_value(self.max_board, self.min_board)

    def max_value(self, max_b: int, min_b: int) -> int:
        """recursively compute the best score for the maximizing player."""
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(min_b):
            self.memo[state_key] = -1
            return -1

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = 0
            return 0

        best_score = -2

        for mask in self.move_masks:
            if not (occupied & mask):
                score = self.min_value(max_b | mask, min_b)

                if score > best_score:
                    best_score = score

                if best_score == 1:
                    self.memo[state_key] = 1
                    return 1

        self.memo[state_key] = best_score
        return best_score

    def min_value(self, max_b: int, min_b: int) -> int:
        """Recursively compute the best score for the minimizing player."""
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(max_b):
            self.memo[state_key] = 1
            return 1

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = 0
            return 0

        best_score = 2

        for mask in self.move_masks:
            if not (occupied & mask):
                score = self.max_value(max_b, min_b | mask)

                if score < best_score:
                    best_score = score

                if best_score == -1:
                    self.memo[state_key] = -1
                    return -1

        self.memo[state_key] = best_score
        return best_score

    def pruning_best_move(self):
        """Calculate the optimal game value using Alpha-Beta pruning."""
        self.memo = {}
        return self.pruning_max(self.max_board, self.min_board, -2, 2)

    def pruning_max(self, max_b: int, min_b: int, alpha: int, beta: int) -> int:
        """Execute the maximizing step with Alpha-Beta pruning optimization."""
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(min_b):
            return -1

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return 0

        for mask in self.move_masks:
            if not (occupied & mask):
                score = self.pruning_min(max_b | mask, min_b, alpha, beta)
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break

        self.memo[state_key] = alpha
        return alpha

    def pruning_min(self, max_b: int, min_b: int, alpha: int, beta: int) -> int:
        """Execute the minimizing step with Alpha-Beta pruning optimization."""
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(max_b):
            return 1

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return 0

        for mask in self.move_masks:
            if not (occupied & mask):
                score = self.pruning_max(max_b, min_b | mask, alpha, beta)
                if score < beta:
                    beta = score
                if beta <= alpha:
                    break

        self.memo[state_key] = beta
        return beta

    def drawboard(self) -> None:
        """Render a text-based representation of the current board state."""
        w = self.w
        max_b = self.max_board
        min_b = self.min_board
        for r in range(self.m):
            line = "|"
            for c in range(self.n):
                idx = r * w + c
                if (max_b >> idx) & 1:
                    line += "X|"
                elif (min_b >> idx) & 1:
                    line += "O|"
                else:
                    line += " |"
            print(line)


if __name__ == "__main__":
    g = Game(4, 4, 4)

    # print("--- Minimax ---")
    # start = time.perf_counter()
    # val = g.get_best_move()
    # end = time.perf_counter()
    # print(f"Val: {val}")
    # print(f"Time: {(end - start) * 1000:.4f}ms")
    # print(f"States: {len(g.memo)}")

    print("\n--- Alpha-Beta ---")
    start = time.perf_counter()
    val_prune = g.pruning_best_move()
    end = time.perf_counter()
    print(f"Val: {val_prune}")
    print(f"Time: {(end - start) * 1000:.4f}ms")
    print(f"States: {len(g.memo)}")
