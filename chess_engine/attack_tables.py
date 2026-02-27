"""Pre-computed attack tables and sliding piece attack generation."""

from chess_engine.constants import (
    BB_ALL, BB_EMPTY, FILE_A, FILE_H,
    WHITE, BLACK, square_file, square_rank,
)
from chess_engine.bitboard import set_bit

# --- Direction offsets ---
NORTH = 8
SOUTH = -8
EAST = 1
WEST = -1
NORTH_EAST = 9
NORTH_WEST = 7
SOUTH_EAST = -7
SOUTH_WEST = -9

# Ray direction indices
DIR_N, DIR_NE, DIR_E, DIR_SE, DIR_S, DIR_SW, DIR_W, DIR_NW = range(8)
RAY_OFFSETS = [NORTH, NORTH_EAST, EAST, SOUTH_EAST, SOUTH, SOUTH_WEST, WEST, NORTH_WEST]


def _init_knight_attacks() -> list[int]:
    table = [BB_EMPTY] * 64
    offsets = [17, 15, 10, 6, -6, -10, -15, -17]
    for sq in range(64):
        bb = BB_EMPTY
        r, f = square_rank(sq), square_file(sq)
        for off in offsets:
            t = sq + off
            if 0 <= t < 64:
                tr, tf = square_rank(t), square_file(t)
                if abs(tr - r) + abs(tf - f) == 3:
                    bb = set_bit(bb, t)
        table[sq] = bb
    return table


def _init_king_attacks() -> list[int]:
    table = [BB_EMPTY] * 64
    offsets = [NORTH, SOUTH, EAST, WEST, NORTH_EAST, NORTH_WEST, SOUTH_EAST, SOUTH_WEST]
    for sq in range(64):
        bb = BB_EMPTY
        r, f = square_rank(sq), square_file(sq)
        for off in offsets:
            t = sq + off
            if 0 <= t < 64:
                tr, tf = square_rank(t), square_file(t)
                if abs(tr - r) <= 1 and abs(tf - f) <= 1:
                    bb = set_bit(bb, t)
        table[sq] = bb
    return table


def _init_pawn_attacks() -> list[list[int]]:
    """PAWN_ATTACKS[side][sq] — squares a pawn on sq attacks."""
    table = [[BB_EMPTY] * 64, [BB_EMPTY] * 64]
    for sq in range(64):
        r, f = square_rank(sq), square_file(sq)
        # White pawns attack NE/NW
        for df, dr in [(1, 1), (-1, 1)]:
            tf, tr_ = f + df, r + dr
            if 0 <= tf < 8 and 0 <= tr_ < 8:
                table[WHITE][sq] = set_bit(table[WHITE][sq], tr_ * 8 + tf)
        # Black pawns attack SE/SW
        for df, dr in [(1, -1), (-1, -1)]:
            tf, tr_ = f + df, r + dr
            if 0 <= tf < 8 and 0 <= tr_ < 8:
                table[BLACK][sq] = set_bit(table[BLACK][sq], tr_ * 8 + tf)
    return table


def _init_ray_masks() -> list[list[int]]:
    """RAY_MASKS[direction][sq] — ray from sq in direction, excluding sq."""
    table = [[BB_EMPTY] * 64 for _ in range(8)]
    for d_idx, offset in enumerate(RAY_OFFSETS):
        for sq in range(64):
            bb = BB_EMPTY
            r, f = square_rank(sq), square_file(sq)
            cr, cf = r, f
            while True:
                # step
                t = (cr * 8 + cf) + offset
                if t < 0 or t >= 64:
                    break
                tr, tf_ = square_rank(t), square_file(t)
                # Validate no wrap-around
                if abs(tr - cr) > 1 or abs(tf_ - cf) > 1:
                    break
                bb = set_bit(bb, t)
                cr, cf = tr, tf_
            table[d_idx][sq] = bb
    return table


# Pre-computed tables (module-level initialization)
KNIGHT_ATTACKS: list[int] = _init_knight_attacks()
KING_ATTACKS: list[int] = _init_king_attacks()
PAWN_ATTACKS: list[list[int]] = _init_pawn_attacks()
RAY_MASKS: list[list[int]] = _init_ray_masks()


def sliding_attacks(sq: int, occupied: int, directions: list[int]) -> int:
    """Compute sliding attacks from sq given occupied bitboard.

    directions: list of direction indices (DIR_N, DIR_NE, etc.)
    """
    attacks = BB_EMPTY
    for d in directions:
        ray = RAY_MASKS[d][sq]
        # Find the first blocker along the ray
        blockers = ray & occupied
        if blockers:
            if d in (DIR_N, DIR_NE, DIR_E, DIR_NW):
                # Positive ray — first blocker is LSB
                blocker_sq = (blockers & -blockers).bit_length() - 1
            else:
                # Negative ray — first blocker is MSB
                blocker_sq = blockers.bit_length() - 1
            # Include blocker square, exclude everything beyond
            attacks |= ray ^ RAY_MASKS[d][blocker_sq]
        else:
            attacks |= ray
    return attacks


BISHOP_DIRS = [DIR_NE, DIR_SE, DIR_SW, DIR_NW]
ROOK_DIRS = [DIR_N, DIR_E, DIR_S, DIR_W]
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS


def bishop_attacks(sq: int, occupied: int) -> int:
    return sliding_attacks(sq, occupied, BISHOP_DIRS)


def rook_attacks(sq: int, occupied: int) -> int:
    return sliding_attacks(sq, occupied, ROOK_DIRS)


def queen_attacks(sq: int, occupied: int) -> int:
    return sliding_attacks(sq, occupied, QUEEN_DIRS)


def is_square_attacked(sq: int, by_side: int, piece_bbs: list[int], occupied: int) -> bool:
    """Check if square is attacked by by_side.

    piece_bbs: list of 12 bitboards indexed by piece constant.
    """
    offset = 6 if by_side == BLACK else 0

    # Pawn attacks: can sq be attacked by a pawn of by_side?
    if PAWN_ATTACKS[1 - by_side][sq] & piece_bbs[0 + offset]:
        return True
    # Knight
    if KNIGHT_ATTACKS[sq] & piece_bbs[1 + offset]:
        return True
    # King
    if KING_ATTACKS[sq] & piece_bbs[5 + offset]:
        return True
    # Bishop/Queen (diagonal)
    diag = bishop_attacks(sq, occupied)
    if diag & (piece_bbs[2 + offset] | piece_bbs[4 + offset]):
        return True
    # Rook/Queen (orthogonal)
    orth = rook_attacks(sq, occupied)
    if orth & (piece_bbs[3 + offset] | piece_bbs[4 + offset]):
        return True
    return False
