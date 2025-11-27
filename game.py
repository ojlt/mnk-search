class Game:
    """Bitboard game engine with Memoization and Alpha-Beta Pruning."""

    def __init__(self, m, n, k):
        """Initialize the game parameters, bitmasks, and winning shift directions."""
        self.m = m
        self.n = n
        self.k = k
        self.w = n + 1
        self.max_board = 0
        self.min_board = 0
        self.memo = {}
        self.node_count = 0

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

    def make_max_move(self, row, col):
        """Place a piece for the maximizing player at the given coordinates."""
        self.max_board |= 1 << (row * self.w + col)

    def make_min_move(self, row, col):
        """Place a piece for the minimizing player at the given coordinates."""
        self.min_board |= 1 << (row * self.w + col)

    def is_valid_move(self, row, col):
        """Check if the cell at the specified row and column is empty."""
        return not ((self.max_board | self.min_board) & (1 << (row * self.w + col)))

    def check_win(self, board):
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
        self.memo = {}
        self.node_count = 0
        return self.max_value(self.max_board, self.min_board)

    def max_value(self, max_b, min_b):
        """Recursively compute the best score for the maximizing player."""
        self.node_count += 1
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
                # Early stopping removed here.
                # It will now check ALL moves even if it found a win.

        self.memo[state_key] = best_score
        return best_score

    def min_value(self, max_b, min_b):
        """Recursively compute the best score for the minimizing player."""
        self.node_count += 1
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
                # Early stopping removed here.
                # It will now check ALL moves even if it found a win.

        self.memo[state_key] = best_score
        return best_score

    def pruning_best_move(self):
        """Calculate the optimal game value using Alpha-Beta pruning."""
        self.memo = {}
        self.node_count = 0
        return self.pruning_max(self.max_board, self.min_board, -2, 2)

    def pruning_max(self, max_b, min_b, alpha, beta):
        """Execute the maximizing step with Alpha-Beta pruning optimization."""
        self.node_count += 1
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

    def pruning_min(self, max_b, min_b, alpha, beta):
        """Execute the minimizing step with Alpha-Beta pruning optimization."""
        self.node_count += 1
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

    def drawboard(self):
        """Render a text-based representation of the current board state."""
        w = self.w
        max_b = self.max_board
        min_b = self.min_board
        print(f"\nBoard ({self.m}x{self.n}):")
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
        print("-" * (self.n * 2 + 1))

    def play(self):
        """Main game loop for Human vs AI."""
        print(f"Starting Game. You are Max (X), AI is Min (O).")
        self.drawboard()

        while True:
            while True:
                try:
                    user_input = input("Enter move (row col): ").split()
                    if not user_input:
                        continue
                    r, c = int(user_input[0]), int(user_input[1])
                    if 0 <= r < self.m and 0 <= c < self.n and self.is_valid_move(r, c):
                        self.make_max_move(r, c)
                        break
                    else:
                        print("Invalid move.")
                except (ValueError, IndexError):
                    print("Invalid format. Use: row col")

            self.drawboard()

            if self.check_win(self.max_board):
                print("You Win!")
                break
            if (self.max_board | self.min_board) & self.full_mask == self.full_mask:
                print("Draw!")
                break

            print("AI is thinking...")
            best_val = 2
            best_move = None

            # Reset counters for the AI's turn
            self.memo = {}
            self.node_count = 0

            # Find the best move by checking all children of the current state
            for r in range(self.m):
                for c in range(self.n):
                    if self.is_valid_move(r, c):
                        mask = 1 << (r * self.w + c)
                        # We are Minimizing, so we check the Max Value of the resulting state
                        val = self.pruning_max(self.max_board, self.min_board | mask, -2, 2)
                        if val < best_val:
                            best_val = val
                            best_move = (r, c)

            print(f"States visited: {self.node_count}")

            if best_move:
                print(f"AI plays: {best_move}")
                self.make_min_move(best_move[0], best_move[1])
            else:
                print("AI has no moves left.")

            self.drawboard()

            if self.check_win(self.min_board):
                print("AI Wins!")
                break
            if (self.max_board | self.min_board) & self.full_mask == self.full_mask:
                print("Draw!")
                break


if __name__ == "__main__":
    g = Game(3, 3, 3)
    g.play()
