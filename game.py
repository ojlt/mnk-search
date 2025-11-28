class Game:
    """Bitboard game engine with Alpha-Beta Pruning (No Memoization)."""

    def __init__(self, m, n, k):
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

        # Generate Search Order Center-Out Indices
        center_r, center_c = self.m // 2, self.n // 2
        coords = [(r, c) for r in range(self.m) for c in range(self.n)]
        coords.sort(key=lambda x: abs(x[0] - center_r) + abs(x[1] - center_c))

        # Store a flattened bit indices (i.e 0 to 15 for 4x4)
        self.search_indices = [1 << (r * self.n + c) for r, c in coords]



    def get_coords_from_mask(self, mask):
        if not mask:
            return None
        idx = mask.bit_length() - 1
        r = idx // self.w
        c = idx % self.w
        return (r, c)

    def make_max_move(self, row, col):
        self.max_board |= 1 << (row * self.w + col)

    def make_min_move(self, row, col):
        self.min_board |= 1 << (row * self.w + col)

    def is_valid_move(self, row, col):
        return not ((self.max_board | self.min_board) & (1 << (row * self.w + col)))

    def check_win(self, board):
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
        self.memo = {}
        self.node_count = 0
        return self.max_value(self.max_board, self.min_board)

    def max_value(self, max_b, min_b):
        self.node_count += 1
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(min_b):
            self.memo[state_key] = (-1, None)
            return (-1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = (0, None)
            return (0, None)

        best_score = -2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.min_value(max_b | mask, min_b)
                if score > best_score:
                    best_score = score
                    best_move = mask

        self.memo[state_key] = (best_score, best_move)
        return (best_score, best_move)

    def min_value(self, max_b, min_b):
        self.node_count += 1
        state_key = (max_b, min_b)
        if state_key in self.memo:
            return self.memo[state_key]

        if self.check_win(max_b):
            self.memo[state_key] = (1, None)
            return (1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = (0, None)
            return (0, None)

        best_score = 2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.max_value(max_b, min_b | mask)
                if score < best_score:
                    best_score = score
                    best_move = mask

        self.memo[state_key] = (best_score, best_move)
        return (best_score, best_move)

    def get_best_move_no_memo(self):
        self.node_count = 0
        return self.max_value_no_memo(self.max_board, self.min_board)

    def max_value_no_memo(self, max_b, min_b):
        self.node_count += 1

        if self.check_win(min_b):
            return (-1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = -2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.min_value_no_memo(max_b | mask, min_b)

                if score > best_score:
                    best_score = score
                    best_move = mask

        return (best_score, best_move)

    def min_value_no_memo(self, max_b, min_b):
        self.node_count += 1

        if self.check_win(max_b):
            return (1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = 2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.max_value_no_memo(max_b, min_b | mask)

                if score < best_score:
                    best_score = score
                    best_move = mask

        return (best_score, best_move)

    def pruning_best_move(self):
        self.node_count = 0
        return self.pruning_max(self.max_board, self.min_board, -2, 2)

    def pruning_max(self, max_b, min_b, alpha, beta):
        self.node_count += 1

        if self.check_win(min_b):
            return (-1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = -2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.pruning_min(max_b | mask, min_b, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_move = mask

                if best_score > alpha:
                    alpha = best_score

                if alpha >= beta:
                    break

        return (best_score, best_move)

    def pruning_min(self, max_b, min_b, alpha, beta):
        self.node_count += 1

        if self.check_win(max_b):
            return (1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = 2
        best_move = None

        for mask in self.search_indices:
            if not (occupied & mask):
                score, _ = self.pruning_max(max_b, min_b | mask, alpha, beta)

                if score < best_score:
                    best_score = score
                    best_move = mask

                if best_score < beta:
                    beta = best_score

                if beta <= alpha:
                    break

        return (best_score, best_move)

    def drawboard(self):
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
        print("Starting Game. You are Max (X), AI is Min (O).")
        self.drawboard()

        while True:
            self.node_count = 0
            advice_score, advice_mask = self.pruning_max(self.max_board, self.min_board, -2, 2)

            if advice_mask:
                adv_r, adv_c = self.get_coords_from_mask(advice_mask)
                print(f"Advice: Placing at ({adv_r}, {adv_c}) leads to score {advice_score}")
            else:
                print("Advice: No moves left or game over.")

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
            if ((self.max_board | self.min_board) & self.full_mask) == self.full_mask:
                print("Draw!")
                break

            print("AI is thinking...")
            self.node_count = 0
            ai_score, ai_mask = self.pruning_min(self.max_board, self.min_board, -2, 2)

            print(f"States visited: {self.node_count}")
            print(f"AI evaluation: {ai_score}")

            if ai_mask:
                r, c = self.get_coords_from_mask(ai_mask)
                print(f"AI plays: ({r}, {c})")
                self.make_min_move(r, c)
            else:
                print("AI cannot move.")

            self.drawboard()

            if self.check_win(self.min_board):
                print("AI Wins!")
                break
            if ((self.max_board | self.min_board) & self.full_mask) == self.full_mask:
                print("Draw!")
                break


if __name__ == "__main__":
    g = Game(3, 3, 3)
    g.play()
