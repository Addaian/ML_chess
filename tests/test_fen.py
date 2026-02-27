"""Tests for FEN parsing and serialization."""

from chess_engine.board import Board
from chess_engine.fen import STARTING_FEN


def test_starting_fen_roundtrip():
    b = Board()
    assert b.fen() == STARTING_FEN


def test_custom_fen_roundtrip():
    fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
    b = Board(fen)
    assert b.fen() == fen


def test_fen_with_ep():
    fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    b = Board(fen)
    assert b.fen() == fen


def test_fen_no_castling():
    fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
    b = Board(fen)
    assert b.fen() == fen


def test_fen_partial_castling():
    fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"
    b = Board(fen)
    assert b.fen() == fen
