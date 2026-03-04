"""Static evaluation: neural network (v3) with hand-coded fallback (v2).

Set CHESS_EVAL=hce in the environment to force the hand-coded evaluator.
"""

import os as _os

_nn_model = None
_nn_loaded = False
_NN_ENABLED = _os.environ.get("CHESS_EVAL", "nn") != "hce"
_EVAL_SCALE = 1500  # centipawns — must match training/config.py EVAL_CLAMP


def _load_nn():
    global _nn_model, _nn_loaded
    _nn_loaded = True
    weights_path = _os.path.join(_os.path.dirname(__file__), "nn", "weights.npz")
    if _os.path.exists(weights_path):
        from chess_engine.nn import NumpyChessNet
        _nn_model = NumpyChessNet(weights_path)


def evaluate(board) -> int:
    """Return static evaluation in centipawns from White's perspective.

    Uses the neural network evaluator if weights are present, otherwise
    falls back to the hand-coded evaluator (_hce_evaluate).
    """
    if _NN_ENABLED:
        global _nn_model, _nn_loaded
        if not _nn_loaded:
            _load_nn()
        if _nn_model is not None:
            from chess_engine.nn.features import board_to_tensor
            features = board_to_tensor(board)
            raw = _nn_model.forward(features)
            return int(raw * _EVAL_SCALE)
    return _hce_evaluate(board)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-coded evaluator (v2 baseline) — kept intact as fallback
# ─────────────────────────────────────────────────────────────────────────────

"""Hand-coded evaluation: material counting + piece-square tables (Michniewski) + positional terms."""

from chess_engine.bitboard import iter_bits, popcount
from chess_engine.constants import (
    WHITE, BLACK,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_ROOK,
    FILES, RANKS,
    piece_side, piece_type, square_file, square_rank,
)

# --- Material values (centipawns) ---
# Indexed by piece type 0-5 (pawn, knight, bishop, rook, queen, king)
MATERIAL = [100, 320, 330, 500, 900, 0]

# --- Piece-square tables (Michniewski values) ---
# LERF order: index 0 = a1, index 63 = h8.
# Values are from White's perspective; for Black, mirror with sq ^ 56.

PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 1
     5, 10, 10,-20,-20, 10, 10,  5,   # rank 2
     5, -5,-10,  0,  0,-10, -5,  5,   # rank 3
     0,  0,  0, 20, 20,  0,  0,  0,   # rank 4
     5,  5, 10, 25, 25, 10,  5,  5,   # rank 5
    10, 10, 20, 30, 30, 20, 10, 10,   # rank 6
    50, 50, 50, 50, 50, 50, 50, 50,   # rank 7
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,   # rank 1
    -10,  5,  0,  0,  0,  0,  5,-10,   # rank 2
    -10, 10, 10, 10, 10, 10, 10,-10,   # rank 3
    -10,  0, 10, 10, 10, 10,  0,-10,   # rank 4
    -10,  5,  5, 10, 10,  5,  5,-10,   # rank 5
    -10,  0,  5, 10, 10,  5,  0,-10,   # rank 6
    -10,  0,  0,  0,  0,  0,  0,-10,   # rank 7
    -20,-10,-10,-10,-10,-10,-10,-20,   # rank 8
]

PST_ROOK = [
     0,  0,  0,  5,  5,  0,  0,  0,   # rank 1
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 2
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 3
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 4
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 5
    -5,  0,  0,  0,  0,  0,  0, -5,   # rank 6
     5, 10, 10, 10, 10, 10, 10,  5,   # rank 7
     0,  0,  0,  0,  0,  0,  0,  0,   # rank 8
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,   # rank 1
    -10,  0,  5,  0,  0,  0,  0,-10,   # rank 2
    -10,  5,  5,  5,  5,  5,  0,-10,   # rank 3
      0,  0,  5,  5,  5,  5,  0, -5,   # rank 4
     -5,  0,  5,  5,  5,  5,  0, -5,   # rank 5
    -10,  0,  5,  5,  5,  5,  0,-10,   # rank 6
    -10,  0,  0,  0,  0,  0,  0,-10,   # rank 7
    -20,-10,-10, -5, -5,-10,-10,-20,   # rank 8
]

PST_KING = [
     20, 30, 10,  0,  0, 10, 30, 20,   # rank 1
     20, 20,  0,  0,  0,  0, 20, 20,   # rank 2
    -10,-20,-20,-20,-20,-20,-20,-10,   # rank 3
    -20,-30,-30,-40,-40,-30,-30,-20,   # rank 4
    -30,-40,-40,-50,-50,-40,-40,-30,   # rank 5
    -30,-40,-40,-50,-50,-40,-40,-30,   # rank 6
    -30,-40,-40,-50,-50,-40,-40,-30,   # rank 7
    -30,-40,-40,-50,-50,-40,-40,-30,   # rank 8
]

# Indexed by piece type 0-5
PST = [PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING]


def _passed_pawn_bonus(sq: int, side: int) -> int:
    """Bonus for a passed pawn: +20 per rank advanced from the start rank."""
    if side == WHITE:
        # Rank 2 (index 1) is start; rank 7 (index 6) is one step from promotion
        return 20 * (square_rank(sq) - 1)
    else:
        return 20 * (6 - square_rank(sq))


def _hce_evaluate(board) -> int:
    """Hand-coded evaluation in centipawns from White's perspective.

    Positive = White is better, negative = Black is better.
    """
    score = 0

    # ── Material + PST ──────────────────────────────────────────────────────
    for piece in range(12):
        bb = board.piece_bb[piece]
        if bb == 0:
            continue
        side = piece_side(piece)
        pt = piece_type(piece)
        table = PST[pt]
        mat = MATERIAL[pt]
        for sq in iter_bits(bb):
            pst_sq = sq if side == WHITE else sq ^ 56
            value = mat + table[pst_sq]
            if side == WHITE:
                score += value
            else:
                score -= value

    # ── Bishop pair ─────────────────────────────────────────────────────────
    if popcount(board.piece_bb[WHITE_BISHOP]) >= 2:
        score += 30
    if popcount(board.piece_bb[WHITE_BISHOP + 6]) >= 2:
        score -= 30

    # ── Rooks on open / semi-open files ─────────────────────────────────────
    white_pawns = board.piece_bb[WHITE_PAWN]
    black_pawns = board.piece_bb[BLACK_PAWN]

    for sq in iter_bits(board.piece_bb[WHITE_ROOK]):
        file_mask = FILES[square_file(sq)]
        if not (white_pawns & file_mask) and not (black_pawns & file_mask):
            score += 25   # open file
        elif not (white_pawns & file_mask):
            score += 15   # semi-open file (no friendly pawn)

    for sq in iter_bits(board.piece_bb[BLACK_ROOK]):
        file_mask = FILES[square_file(sq)]
        if not (black_pawns & file_mask) and not (white_pawns & file_mask):
            score -= 25
        elif not (black_pawns & file_mask):
            score -= 15

    # ── Passed pawns ─────────────────────────────────────────────────────────
    for sq in iter_bits(white_pawns):
        f = square_file(sq)
        r = square_rank(sq)
        # Files to check: adjacent + own
        adj_files = FILES[f]
        if f > 0:
            adj_files |= FILES[f - 1]
        if f < 7:
            adj_files |= FILES[f + 1]
        # Enemy pawns ahead (ranks above sq for White)
        ahead_mask = 0
        for rank in range(r + 1, 8):
            ahead_mask |= RANKS[rank]
        if not (black_pawns & adj_files & ahead_mask):
            score += _passed_pawn_bonus(sq, WHITE)

    for sq in iter_bits(black_pawns):
        f = square_file(sq)
        r = square_rank(sq)
        adj_files = FILES[f]
        if f > 0:
            adj_files |= FILES[f - 1]
        if f < 7:
            adj_files |= FILES[f + 1]
        ahead_mask = 0
        for rank in range(0, r):
            ahead_mask |= RANKS[rank]
        if not (white_pawns & adj_files & ahead_mask):
            score -= _passed_pawn_bonus(sq, BLACK)

    # ── King pawn shield ─────────────────────────────────────────────────────
    # +10 per friendly pawn on ranks 2-3 in front of a castled king (files a-c or f-h)
    from chess_engine.constants import RANK_2, RANK_3

    wk_sq = (board.piece_bb[WHITE_KING] & -board.piece_bb[WHITE_KING]).bit_length() - 1
    if wk_sq >= 0:
        wk_file = square_file(wk_sq)
        if wk_file >= 5:  # Kingside (f, g, h)
            shield_files = FILES[5] | FILES[6] | FILES[7]
            shield_ranks = RANK_2 | RANK_3
            score += 10 * popcount(white_pawns & shield_files & shield_ranks)
        elif wk_file <= 2:  # Queenside (a, b, c)
            shield_files = FILES[0] | FILES[1] | FILES[2]
            shield_ranks = RANK_2 | RANK_3
            score += 10 * popcount(white_pawns & shield_files & shield_ranks)

    bk_sq = (board.piece_bb[WHITE_KING + 6] & -board.piece_bb[WHITE_KING + 6]).bit_length() - 1
    if bk_sq >= 0:
        bk_file = square_file(bk_sq)
        if bk_file >= 5:
            shield_files = FILES[5] | FILES[6] | FILES[7]
            shield_ranks = RANK_2 | RANK_3  # from Black's POV: ranks 6-7 (indices 5-6)
            # Black's shield pawns are on ranks 6-7 (rank index 5 and 6)
            from chess_engine.constants import RANK_6, RANK_7
            shield_ranks_b = RANK_6 | RANK_7
            score -= 10 * popcount(black_pawns & shield_files & shield_ranks_b)
        elif bk_file <= 2:
            shield_files = FILES[0] | FILES[1] | FILES[2]
            from chess_engine.constants import RANK_6, RANK_7
            shield_ranks_b = RANK_6 | RANK_7
            score -= 10 * popcount(black_pawns & shield_files & shield_ranks_b)

    return score
