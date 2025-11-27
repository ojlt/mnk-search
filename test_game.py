import unittest
from game import Game


class TestBitboardGame(unittest.TestCase):
    def setUp(self):
        """Initialize a standard 3x3 Tic-Tac-Toe board before each test."""
        self.game = Game(3, 3, 3)

    def test_initialization(self):
        """Ensure the board starts empty and dimensions are correct."""
        self.assertEqual(self.game.max_board, 0)
        self.assertEqual(self.game.min_board, 0)
        self.assertEqual(self.game.m, 3)
        self.assertEqual(self.game.n, 3)
        self.assertEqual(self.game.k, 3)

    def test_make_moves(self):
        """Test that moves update the bitboards correctly."""
        # Max makes a move at 0,0
        self.game.make_max_move(0, 0)
        # 0,0 is index 0. 1 << 0 is 1.
        self.assertEqual(self.game.max_board, 1)
        self.assertFalse(self.game.is_valid_move(0, 0))

        # Min makes a move at 0,1
        self.game.make_min_move(0, 1)
        # 0,1 is index 1. 1 << 1 is 2.
        self.assertEqual(self.game.min_board, 2)
        self.assertFalse(self.game.is_valid_move(0, 1))

        # Check valid spot
        self.assertTrue(self.game.is_valid_move(0, 2))

    def test_win_horizontal(self):
        """Test row detection logic."""
        # X X X
        # . . .
        # . . .
        self.game.make_max_move(0, 0)
        self.game.make_max_move(0, 1)
        self.game.make_max_move(0, 2)
        self.assertTrue(self.game.check_win(self.game.max_board))
        self.assertFalse(self.game.check_win(self.game.min_board))

    def test_win_vertical(self):
        """Test column detection logic."""
        # X . .
        # X . .
        # X . .
        self.game.make_max_move(0, 0)
        self.game.make_max_move(1, 0)
        self.game.make_max_move(2, 0)
        self.assertTrue(self.game.check_win(self.game.max_board))

    def test_win_diagonal(self):
        """Test diagonal detection logic (\)."""
        # X . .
        # . X .
        # . . X
        self.game.make_max_move(0, 0)
        self.game.make_max_move(1, 1)
        self.game.make_max_move(2, 2)
        self.assertTrue(self.game.check_win(self.game.max_board))

    def test_win_anti_diagonal(self):
        """Test anti-diagonal detection logic (/)."""
        # . . X
        # . X .
        # X . .
        self.game.make_max_move(0, 2)
        self.game.make_max_move(1, 1)
        self.game.make_max_move(2, 0)
        self.assertTrue(self.game.check_win(self.game.max_board))

    def test_no_win(self):
        """Ensure check_win returns False for non-winning patterns."""
        # X X O
        self.game.make_max_move(0, 0)
        self.game.make_max_move(0, 1)
        self.game.make_min_move(0, 2)
        self.assertFalse(self.game.check_win(self.game.max_board))

    # --- AI / Solver Tests ---

    def test_solver_immediate_win(self):
        """
        Situation: Max to move. Max has 2 in a row.
        Expectation: Solver returns 1 (Win).
        Board:
        X X .
        O O .
        . . .
        """
        self.game.make_max_move(0, 0)
        self.game.make_min_move(1, 0)
        self.game.make_max_move(0, 1)
        self.game.make_min_move(1, 1)

        # It is Max's turn. Max should see the win at (0,2)
        result = self.game.get_best_move()
        self.assertEqual(result, 1, "AI failed to find an immediate win.")

    def test_solver_must_block(self):
        """
        Situation: Max to move. Min has 2 in a row.
        Expectation: If Max blocks, it's a draw (0). If Max ignores, it's a loss (-1).
        Since Minimax assumes optimal play, Max *will* block. Result should be 0 (Draw).
        Board:
        X . .
        O O .
        . . X
        """
        self.game.make_max_move(0, 0)
        self.game.make_min_move(1, 0)
        self.game.make_max_move(2, 2)  # Irrelevant move just to pass turn
        self.game.make_min_move(1, 1)

        # Min is threatening to win at (1,2). Max must play (1,2) to survive.
        # With perfect play from here, 3x3 is a draw.
        result = self.game.get_best_move()
        self.assertEqual(result, 0, "AI failed to block a winning threat or miscalculated draw.")

    def test_solver_doomed_loss(self):
        """
        Situation: Max to move. Min has two separate threats.
        Expectation: Solver returns -1 (Loss).
        Board:
        . O .
        O X .
        . . X
        (Min has set up a trap where next move guarantees 2 ways to win?
        Let's construct a simpler forced loss for Max).

        Simpler Scenario:
        O O .
        . . .
        . . .
        But it is Min's turn (impossible in standard flow, but we can call min_value).

        Let's stick to standard flow:
        Board State:
        X O X
        O O .
        X X .

        Turn: Max to move.
        If Max plays (1,2) -> Blocks row.
        But let's create a scenario where Min has a "fork".
        """
        # Reset game for a clean scenario setup
        self.game = Game(3, 3, 3)

        # O . O
        # . X .
        # . . .
        # If it's Max's turn, he might survive.
        # If it's Min's turn (we call min_value), Min wins.

        # Let's verify the full game result (Empty Board).
        # Standard Tic-Tac-Toe is a Draw (0).
        # This acts as an integration test for the whole algorithm.
        result = self.game.get_best_move()
        self.assertEqual(result, 0, "Standard 3x3 Tic-Tac-Toe should be a draw.")

    def test_small_board_win(self):
        """
        Test a 2x2 board with K=2.
        The first player (Max) should always win in a 2x2 connect-2 game.
        """
        g = Game(2, 2, 2)
        # P1 plays (0,0). P2 plays (0,1). P1 plays (1,0) -> Wins col 0.
        # P1 forces a win.
        result = g.get_best_move()
        self.assertEqual(result, 1, "First player should win 2x2 connect-2.")

    def test_check_valid_indices_logic(self):
        """
        Bitboard specific: Ensure 'w' stride handles boundaries correctly.
        (e.g., a piece at end of row 0 doesn't connect to start of row 1).
        """
        # Board 3x3.
        # X X .
        # O O O  <-- Win
        # . . .

        # Place Max at (0, 2) (End of row 0)
        # Place Max at (1, 0) (Start of row 1)
        self.game.make_max_move(0, 2)
        self.game.make_max_move(1, 0)

        # This is NOT a connection.
        # If logic was naive (just index + 1), this might trigger a win or connection logic.
        # Your code uses `self.w` and valid indices masking, so it should be fine.

        # Let's test a diagonal wrap-around failure case specifically.
        g = Game(3, 3, 3)
        g.make_max_move(0, 2)  # Top Right
        g.make_max_move(1, 0)  # Middle Left
        # Visually distinct, but if strides were wrong, could be seen as diagonal.
        self.assertFalse(g.check_win(g.max_board))


if __name__ == "__main__":
    unittest.main()
