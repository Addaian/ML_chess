"""Perft tests — integration validation of the entire move generation pipeline."""

import pytest

from chess_engine.board import Board
from chess_engine.perft import perft


# Standard perft test positions from the Chess Programming Wiki.
PERFT_POSITIONS = [
    # (name, FEN, {depth: expected_nodes})
    (
        "Initial position",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        {1: 20, 2: 400, 3: 8902, 4: 197281},
    ),
    (
        "Kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        {1: 48, 2: 2039, 3: 97862},
    ),
    (
        "Position 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        {1: 14, 2: 191, 3: 2812, 4: 43238},
    ),
    (
        "Position 4",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        {1: 6, 2: 264, 3: 9467, 4: 422333},
    ),
    (
        "Position 5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        {1: 44, 2: 1486, 3: 62379},
    ),
    (
        "Position 6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        {1: 46, 2: 2079, 3: 89890},
    ),
]


@pytest.mark.parametrize(
    "name, fen, expected",
    PERFT_POSITIONS,
    ids=[p[0] for p in PERFT_POSITIONS],
)
def test_perft(name, fen, expected):
    board = Board(fen)
    for depth, exp_nodes in sorted(expected.items()):
        result = perft(board, depth)
        assert result == exp_nodes, (
            f"{name} depth {depth}: got {result}, expected {exp_nodes}"
        )


# Deeper tests — slow, opt-in with pytest -m slow
@pytest.mark.slow
def test_perft_initial_depth5():
    board = Board()
    assert perft(board, 5) == 4865609


@pytest.mark.slow
def test_perft_kiwipete_depth4():
    board = Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    assert perft(board, 4) == 4085603


@pytest.mark.slow
def test_perft_position4_depth5():
    board = Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    assert perft(board, 5) == 15833292


@pytest.mark.slow
def test_perft_position5_depth4():
    board = Board("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8")
    assert perft(board, 4) == 2103487


@pytest.mark.slow
def test_perft_position6_depth4():
    board = Board("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10")
    assert perft(board, 4) == 3894594
