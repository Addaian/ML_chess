"""Tests for the static evaluator."""

from chess_engine.board import Board
from chess_engine.evaluator import _hce_evaluate, PST_KNIGHT, PST, MATERIAL


def test_starting_position_is_zero():
    board = Board()
    assert _hce_evaluate(board) == 0


def test_white_up_a_queen():
    # White has an extra queen, Black is missing hers
    board = Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    score = _hce_evaluate(board)
    assert score > 800


def test_black_up_a_queen():
    # Black has an extra queen, White is missing hers
    board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    score = _hce_evaluate(board)
    assert score < -800


def test_pst_tables_have_64_elements():
    for table in PST:
        assert len(table) == 64


def test_knight_pst_horizontal_symmetry():
    # Knights PST should be horizontally symmetric: pst[i] == pst[i^7] when
    # i and i^7 are in the same rank (which they always are since XOR 7 flips
    # file within a rank).
    for rank in range(8):
        for file in range(4):
            sq = rank * 8 + file
            mirror_sq = rank * 8 + (7 - file)
            assert PST_KNIGHT[sq] == PST_KNIGHT[mirror_sq], (
                f"Knight PST not symmetric: sq={sq} val={PST_KNIGHT[sq]} "
                f"mirror={mirror_sq} val={PST_KNIGHT[mirror_sq]}"
            )


def test_white_pawn_e4_better_than_a2():
    from chess_engine.evaluator import PST_PAWN
    from chess_engine.constants import E4, A2
    assert PST_PAWN[E4] > PST_PAWN[A2]


def test_material_values():
    assert MATERIAL[0] == 100   # pawn
    assert MATERIAL[1] == 320   # knight
    assert MATERIAL[2] == 330   # bishop
    assert MATERIAL[3] == 500   # rook
    assert MATERIAL[4] == 900   # queen
    assert MATERIAL[5] == 0     # king
