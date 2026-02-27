"""Static evaluation: material counting + piece-square tables (Michniewski)."""

from chess_engine.bitboard import iter_bits
from chess_engine.constants import (
    WHITE, BLACK,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    piece_side, piece_type,
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


def evaluate(board) -> int:
    """Return static evaluation in centipawns from White's perspective.

    Positive = White is better, negative = Black is better.
    """
    score = 0
    for piece in range(12):
        bb = board.piece_bb[piece]
        if bb == 0:
            continue
        side = piece_side(piece)
        pt = piece_type(piece)
        table = PST[pt]
        mat = MATERIAL[pt]
        for sq in iter_bits(bb):
            # For White: PST index = sq (LERF, a1=0, rank 1 at bottom)
            # For Black: mirror rank with sq ^ 56
            pst_sq = sq if side == WHITE else sq ^ 56
            value = mat + table[pst_sq]
            if side == WHITE:
                score += value
            else:
                score -= value
    return score
