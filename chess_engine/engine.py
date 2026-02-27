"""CLI entry point: find the best move for a given FEN and depth."""

import sys

from chess_engine.board import Board
from chess_engine.move import move_to_uci
from chess_engine.search import find_best_move

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m chess_engine.engine <FEN> [depth]")
        sys.exit(1)

    fen = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    board = Board(fen)
    print(f"FEN: {fen}")
    print(f"Depth: {depth}")
    print()

    best = find_best_move(board, depth)
    if best is None:
        print("No legal moves.")
    else:
        print(f"bestmove {move_to_uci(best)}")
