"""Minimax search (no pruning)."""

from chess_engine.constants import WHITE
from chess_engine.evaluator import evaluate
from chess_engine.movegen import generate_legal_moves

CHECKMATE_SCORE = 1_000_000
STALEMATE_SCORE = 0


def minimax(board, depth: int, maximizing: bool) -> int:
    """Return evaluation of *board* after searching *depth* plies.

    Score is always from White's perspective (positive = White better).
    *maximizing* must be True when it is White's turn.
    """
    moves = generate_legal_moves(board)

    # Terminal: no legal moves → checkmate or stalemate
    if not moves:
        if board.is_in_check():
            # Side to move is in checkmate.
            # If White is mated → very negative; if Black is mated → very positive.
            return -CHECKMATE_SCORE if maximizing else CHECKMATE_SCORE
        return STALEMATE_SCORE

    # Leaf node
    if depth == 0:
        return evaluate(board)

    if maximizing:
        best = -CHECKMATE_SCORE - 1
        for move in moves:
            board.make_move(move)
            score = minimax(board, depth - 1, False)
            board.unmake_move()
            if score > best:
                best = score
        return best
    else:
        best = CHECKMATE_SCORE + 1
        for move in moves:
            board.make_move(move)
            score = minimax(board, depth - 1, True)
            board.unmake_move()
            if score < best:
                best = score
        return best


def find_best_move(board, depth: int):
    """Return the best move (16-bit int) for the side to move, or None."""
    moves = generate_legal_moves(board)
    if not moves:
        return None

    maximizing = board.side == WHITE
    best_move = None

    if maximizing:
        best_score = -CHECKMATE_SCORE - 1
        for move in moves:
            board.make_move(move)
            score = minimax(board, depth - 1, False)
            board.unmake_move()
            if score > best_score:
                best_score = score
                best_move = move
    else:
        best_score = CHECKMATE_SCORE + 1
        for move in moves:
            board.make_move(move)
            score = minimax(board, depth - 1, True)
            board.unmake_move()
            if score < best_score:
                best_score = score
                best_move = move

    return best_move
