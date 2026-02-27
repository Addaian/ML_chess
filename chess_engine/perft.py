"""Perft algorithm for move generation validation."""

import sys

from chess_engine.board import Board
from chess_engine.movegen import generate_legal_moves
from chess_engine.move import move_to_uci


def perft(board: Board, depth: int) -> int:
    if depth == 0:
        return 1

    if depth == 1:
        return len(generate_legal_moves(board))

    nodes = 0
    for move in generate_legal_moves(board):
        board.make_move(move)
        nodes += perft(board, depth - 1)
        board.unmake_move()
    return nodes


def divide(board: Board, depth: int) -> dict[str, int]:
    result = {}
    total = 0
    for move in generate_legal_moves(board):
        board.make_move(move)
        count = perft(board, depth - 1)
        board.unmake_move()
        uci = move_to_uci(move)
        result[uci] = count
        total += count
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m chess_engine.perft <FEN> [depth]")
        sys.exit(1)

    fen = sys.argv[1]
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    board = Board(fen)
    print(f"FEN: {fen}")
    print(f"Depth: {depth}")
    print()

    result = divide(board, depth)
    for uci in sorted(result):
        print(f"{uci}: {result[uci]}")
    print()
    total = sum(result.values())
    print(f"Nodes: {total}")
