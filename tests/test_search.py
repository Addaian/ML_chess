"""Tests for minimax search."""

from chess_engine.board import Board
from chess_engine.move import move_to_uci
from chess_engine.movegen import generate_legal_moves
from chess_engine.search import find_best_move, minimax, CHECKMATE_SCORE, STALEMATE_SCORE


def _is_checkmate(board):
    """True if the side to move is in checkmate."""
    return not generate_legal_moves(board) and board.is_in_check()


def test_white_mate_in_one():
    # White: Ke6 + Qd5, Black: Ke8.  White to move, several mates exist.
    board = Board("4k3/8/4K3/3Q4/8/8/8/8 w - - 0 1")
    move = find_best_move(board, 1)
    assert move is not None
    board.make_move(move)
    assert _is_checkmate(board), f"{move_to_uci(move)} is not checkmate"


def test_black_mate_in_one():
    # Mirror: Black: Ke3 + Qd4, White: Ke1.  Black to move.
    board = Board("8/8/8/8/3q4/4k3/8/4K3 b - - 0 1")
    move = find_best_move(board, 1)
    assert move is not None
    board.make_move(move)
    assert _is_checkmate(board), f"{move_to_uci(move)} is not checkmate"


def test_captures_undefended_rook():
    # White queen can capture undefended black rook on a8
    board = Board("r3k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    move = find_best_move(board, 1)
    assert move is not None
    assert move_to_uci(move) == "a1a8"


def test_no_moves_stalemate_returns_none():
    # Stalemate: Black king on a8, White king on b6, White queen on c7
    # Black to move, no legal moves, not in check → stalemate
    board = Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
    assert not generate_legal_moves(board)  # confirm stalemate
    move = find_best_move(board, 1)
    assert move is None


def test_no_moves_checkmate_returns_none():
    # Fool's mate result: White is checkmated
    board = Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    move = find_best_move(board, 1)
    assert move is None


def test_checkmate_score_value():
    # Position where White is checkmated (no legal moves, in check)
    board = Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 1")
    score = minimax(board, 1, True)
    assert score == -CHECKMATE_SCORE


def test_stalemate_score_is_zero():
    board = Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
    score = minimax(board, 1, False)
    assert score == STALEMATE_SCORE
