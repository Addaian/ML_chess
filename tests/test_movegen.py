"""Tests for move generation."""

from chess_engine.board import Board
from chess_engine.movegen import generate_legal_moves
from chess_engine.move import move_to_uci


def _move_set(fen):
    b = Board(fen)
    return set(move_to_uci(m) for m in generate_legal_moves(b))


def test_starting_position_move_count():
    b = Board()
    moves = generate_legal_moves(b)
    assert len(moves) == 20


def test_starting_position_contains_e2e4():
    moves = _move_set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert "e2e4" in moves
    assert "e2e3" in moves
    assert "g1f3" in moves


def test_castling_available():
    moves = _move_set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    assert "e1g1" in moves  # kingside
    assert "e1c1" in moves  # queenside


def test_castling_blocked_by_piece():
    moves = _move_set("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/RN2K2R w KQkq - 0 1")
    assert "e1g1" in moves
    assert "e1c1" not in moves  # blocked by knight on b1


def test_castling_through_check():
    # White king on e1, black rook on f8 — f1 attacked, no kingside castling
    moves = _move_set("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1")
    assert "e1g1" not in moves  # f1 is attacked
    assert "e1c1" in moves


def test_en_passant():
    # White pawn on e5, black just played d7d5
    moves = _move_set("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    assert "e5d6" in moves


def test_promotion():
    moves = _move_set("8/P7/8/8/8/8/8/4K2k w - - 0 1")
    assert "a7a8q" in moves
    assert "a7a8r" in moves
    assert "a7a8b" in moves
    assert "a7a8n" in moves


def test_promotion_capture():
    moves = _move_set("1n6/P7/8/8/8/8/8/4K2k w - - 0 1")
    assert "a7b8q" in moves
    assert "a7b8n" in moves


def test_in_check_must_escape():
    # White king in check by black rook
    b = Board("4k3/8/8/8/8/8/8/r3K3 w - - 0 1")
    moves = generate_legal_moves(b)
    uci_moves = set(move_to_uci(m) for m in moves)
    # King must move, can't stay on e1
    for m in uci_moves:
        assert m.startswith("e1")


def test_stalemate_no_moves():
    # White king stalemated
    b = Board("k7/2Q5/1K6/8/8/8/8/8 b - - 0 1")
    moves = generate_legal_moves(b)
    assert len(moves) == 0


def test_make_unmake_preserves_board():
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    b = Board(fen)
    for move in generate_legal_moves(b):
        b.make_move(move)
        b.unmake_move()
        assert b.fen() == fen, f"FEN changed after make/unmake of {move_to_uci(move)}"
