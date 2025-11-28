class Game:
    """Bitboard game engine with Alpha-Beta Pruning (No Memoization)."""

    def __init__(self, m, n, k):
        self.m = m
        self.n = n
        self.k = k
        # width + 1 creates a 'buffer column' in the bit string. 
        # (this prevents horizontal win-checks from wrapping around to the next row)
        self.w = n + 1 
        
        #pPre-calculate shift amounts for win checking
        shifts = []
        count = 1
        while count < k:
            step = min(count, k - count)
            shifts.append(step)
            count += step
        self.check_shifts = tuple(shifts)
        
 
        self.directions = (1, self.w, self.w + 1, self.w - 1)# directions: Horizontal, Vertical, Diagonal, Anti-Diagonal
        
        # generate valid bit indices (skipping the buffer column)
        valid_indices = [r * self.w + c for r in range(m) for c in range(n)]
        
        # mask representing a full board (for draw detection)
        self.full_mask = sum(1 << i for i in valid_indices) 
        
        # precomputed single-bit masks for every valid position
        self.move_masks = tuple((1 << i) for i in valid_indices)
        
        # Initialize the game state
        self.initialize_game()

    def initialize_game(self):
        """Initializes the empty m x n board (Requirement 1a)."""
        self.max_board = 0  # Bitboard for Player Max
        self.min_board = 0  # Bitboard for Player Min
        self.memo = {}      
        self.node_count = 0

    def get_coords_from_mask(self, mask):
        if not mask:
            return None
        # get the index of the highest set bit
        idx = mask.bit_length() - 1
        # map flat index back to 2d coordinates (row, col)
        r = idx // self.w
        c = idx % self.w
        return (r, c)

    def make_max_move(self, row, col):
        # OR operation sets the specific bit for the max player
        self.max_board |= 1 << (row * self.w + col)

    def make_min_move(self, row, col):
        # same as above, but for the min player
        self.min_board |= 1 << (row * self.w + col)

    def is_valid_move(self, row, col):
        # check if spot is free on both boards
        # combine boards, check specific bit, invert result
        return not ((self.max_board | self.min_board) & (1 << (row * self.w + col)))

    def check_win(self, board):
        # bitwise parallel check for k-in-a-row in any direction
        for d in self.directions:
            b = board  # working copy of the board
            for step in self.check_shifts:
                # shift board by 'step' in direction 'd' and AND with original
                # this filters for bits that have a match 'step' distance away
                b &= b >> (d * step)
                if not b:
                    break  # optimization: stop early if no potential wins remain
            if b:
                return True  # if any bits remain, we have a winner
        return False

    def get_best_move(self):
        # reset stats and cache before starting a fresh search
        self.memo = {}
        self.node_count = 0
        return self.max_value(self.max_board, self.min_board)

    def max_value(self, max_b, min_b):
        self.node_count += 1
        state_key = (max_b, min_b)
        
        # check cache first
        if state_key in self.memo:
            return self.memo[state_key]

        # terminal state check: did Min just win?
        if self.check_win(min_b):
            self.memo[state_key] = (-1, None)
            return (-1, None)

        occupied = max_b | min_b
        # check for draw (board full)
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = (0, None)
            return (0, None)

        best_score = -2  # initialize lower than worst possible outcome (-1)
        best_move = None

        for mask in self.move_masks:
            # only try moves that aren't occupied
            if not (occupied & mask):
                # recurse: It's Min's turn now
                score, _ = self.min_value(max_b | mask, min_b)
                
                # Max wants to maximize the score
                if score > best_score:
                    best_score = score
                    best_move = mask

        self.memo[state_key] = (best_score, best_move)  # cache result
        return (best_score, best_move)

    def min_value(self, max_b, min_b):
        self.node_count += 1
        state_key = (max_b, min_b)
        
        if state_key in self.memo:
            return self.memo[state_key]

        # terminal state check: did Max just win?
        if self.check_win(max_b):
            self.memo[state_key] = (1, None)
            return (1, None)

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            self.memo[state_key] = (0, None)
            return (0, None)

        best_score = 2  # initialize higher than worst possible outcome (1)
        best_move = None

        for mask in self.move_masks:
            if not (occupied & mask):
                # recurse: It's Max's turn now
                score, _ = self.max_value(max_b, min_b | mask)
                
                # Min wants to minimize the score
                if score < best_score:
                    best_score = score
                    best_move = mask

        self.memo[state_key] = (best_score, best_move)
        return (best_score, best_move)

    def get_best_move_no_memo(self):
        # pure minimax without caching 
        self.node_count = 0
        return self.max_value_no_memo(self.max_board, self.min_board)

    def max_value_no_memo(self, max_b, min_b):
        self.node_count += 1

        if self.check_win(min_b):
            return (-1, None)  # loss for max

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)  # board full

        best_score = -2
        best_move = None

        for mask in self.move_masks:
            if not (occupied & mask):
                score, _ = self.min_value_no_memo(max_b | mask, min_b)

                if score > best_score:
                    best_score = score
                    best_move = mask

        return (best_score, best_move)

    def min_value_no_memo(self, max_b, min_b):
        self.node_count += 1

        if self.check_win(max_b):
            return (1, None)  # loss for min

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = 2
        best_move = None

        for mask in self.move_masks:
            if not (occupied & mask):
                score, _ = self.max_value_no_memo(max_b, min_b | mask)

                if score < best_score:
                    best_score = score
                    best_move = mask

        return (best_score, best_move)

    def pruning_best_move(self):
        self.node_count = 0
        # start search with full alpha-beta window (-infinity to +infinity here is -2 to 2)
        return self.pruning_max(self.max_board, self.min_board, -2, 2)

    def pruning_max(self, max_b, min_b, alpha, beta):
        self.node_count += 1

        if self.check_win(min_b):
            return (-1, None)  # opponent won

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)  # no moves left

        best_score = -2
        best_move = None

        for mask in self.move_masks:
            if not (occupied & mask):
                # try move, swap to min player
                score, _ = self.pruning_min(max_b | mask, min_b, alpha, beta)

                if score > best_score:
                    best_score = score
                    best_move = mask

                # update lower bound
                if best_score > alpha:
                    alpha = best_score

                # pruning: if score is better than opponent's best option, stop
                if alpha >= beta:
                    break  # beta cutoff

        return (best_score, best_move)

    def pruning_min(self, max_b, min_b, alpha, beta):
        self.node_count += 1

        if self.check_win(max_b):
            return (1, None)  # opponent won

        occupied = max_b | min_b
        if (occupied & self.full_mask) == self.full_mask:
            return (0, None)

        best_score = 2
        best_move = None

        for mask in self.move_masks:
            if not (occupied & mask):
                # try move, swap to max player
                score, _ = self.pruning_max(max_b, min_b | mask, alpha, beta)

                if score < best_score:
                    best_score = score
                    best_move = mask

                # update upper bound
                if best_score < beta:
                    beta = best_score

                # pruning: if score is worse than opponent's guaranteed best, stop
                if beta <= alpha:
                    break  # alpha cutoff

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
                # check bits to see who occupies the cell
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
            # get ai suggestion for player using alpha-beta
            advice_score, advice_mask = self.pruning_max(self.max_board, self.min_board, -2, 2)

            if advice_mask:
                adv_r, adv_c = self.get_coords_from_mask(advice_mask)
                print(f"Advice: Placing at ({adv_r}, {adv_c}) leads to score {advice_score}")
            else:
                print("Advice: No moves left or game over.")

            while True:
                # input validation loop
                try:
                    user_input = input("Enter move (row col): ").split()
                    if not user_input:
                        continue
                    r, c = int(user_input[0]), int(user_input[1])

                    # ensure move is within bounds and cell is empty
                    if 0 <= r < self.m and 0 <= c < self.n and self.is_valid_move(r, c):
                        self.make_max_move(r, c)
                        break
                    else:
                        print("Invalid move.")
                except (ValueError, IndexError):
                    print("Invalid format. Use: row col")

            self.drawboard()

            # check if human won
            if self.check_win(self.max_board):
                print("You Win!")
                break
            # check for draw
            if ((self.max_board | self.min_board) & self.full_mask) == self.full_mask:
                print("Draw!")
                break

            print("AI is thinking...")
            self.node_count = 0
            # run alpha-beta search for AI move
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

            # check if AI won
            if self.check_win(self.min_board):
                print("AI Wins!")
                break
            if ((self.max_board | self.min_board) & self.full_mask) == self.full_mask:
                print("Draw!")
                break


if __name__ == "__main__":
    g = Game(4, 3, 3)  # create a 4x3 board, 3-in-a-row wins
    g.play()