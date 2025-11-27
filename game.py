class Game:
    """Bitboard game engine with Memoization and Pre-computed masks."""

    __slots__ = (
        "m",
        "n",
        "k",
        "w",
        "check_shifts",
        "directions",
        "max_board",
        "min_board",
        "full_mask",
        "move_masks",
        "memo",
    )

    def __init__(self, m: int, n: int, k: int) -> None:
        """Initialize game dimensions, pre-compute bitmasks, and set up the memoization table."""
        self.m = m
        self.n = n
        self.k = k
        self.w = n + 1

        shifts = []
        count = 1
        while count < self.k:
            step = min(count, self.k - count)
            shifts.append(step)
            count += step
        self.check_shifts = tuple(shifts)
        self.directions = (1, self.w, self.w + 1, self.w - 1)

        valid_indices = [r * self.w + c for r in range(m) for c in range(n)]
        self.full_mask = sum(1 << i for i in valid_indices)

        self.move_masks = tuple((1 << i) for i in valid_indices)

        self.initialize_game()

    def initialize_game(self) -> None:
        """Reset the board state and clear the memoization cache."""
        self.max_board = 0
        self.min_board = 0
        self.memo = {}

    def make_max_move(self, row: int, col: int) -> None:
        """Update the board with a move for the Max player at the given coordinates."""
        self.max_board |= 1 << (row * self.w + col)

    def make_min_move(self, row: int, col: int) -> None:
        """Update the board with a move for the Min player at the given coordinates."""
        self.min_board |= 1 << (row * self.w + col)

    def is_valid_move(self, row: int, col: int) -> bool:
        """Check if the specified cell is currently empty."""
        return not ((self.max_board | self.min_board) & (1 << (row * self.w + col)))

    def check_win(self, board: int) -> bool:
        """Check if the provided board state contains a winning pattern."""
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
        """Initiate the recursive minimax search from the current board state."""
        return self.max_value(self.max_board, self.min_board)

    def max_value(self, max_b: int, min_b: int) -> int:
        """Recursive function to find the best score for the maximizing player using memoization."""
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
        """Recursive function to find the best score for the minimizing player using memoization."""
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

    def drawboard(self) -> None:
        """Render the current board state to the console using ASCII characters."""
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
    g = Game(3, 3, 3)

    import time

    start = time.perf_counter()

    val = g.get_best_move()

    end = time.perf_counter()
    print(f"Board Value: {val}")
    print(f"Time taken: {(end - start) * 1000:.4f}ms")
    print(f"Memoized States: {len(g.memo)}")
