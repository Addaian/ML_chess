"""16-bit move encoding/decoding.

Layout: [15:12 flags] [11:6 to_sq] [5:0 from_sq]
"""

from chess_engine.constants import SQUARE_NAMES, square_name

# Move flags (4 bits)
FLAG_QUIET = 0
FLAG_DOUBLE_PAWN = 1
FLAG_KING_CASTLE = 2
FLAG_QUEEN_CASTLE = 3
FLAG_CAPTURE = 4
FLAG_EP_CAPTURE = 5
# 6, 7 unused
FLAG_PROMO_KNIGHT = 8
FLAG_PROMO_BISHOP = 9
FLAG_PROMO_ROOK = 10
FLAG_PROMO_QUEEN = 11
FLAG_PROMO_CAPTURE_KNIGHT = 12
FLAG_PROMO_CAPTURE_BISHOP = 13
FLAG_PROMO_CAPTURE_ROOK = 14
FLAG_PROMO_CAPTURE_QUEEN = 15

PROMO_PIECES = "nbrq"


def encode_move(from_sq: int, to_sq: int, flags: int = FLAG_QUIET) -> int:
    return (flags << 12) | (to_sq << 6) | from_sq


def decode_from(move: int) -> int:
    return move & 0x3F


def decode_to(move: int) -> int:
    return (move >> 6) & 0x3F


def decode_flags(move: int) -> int:
    return (move >> 12) & 0xF


def is_capture(move: int) -> bool:
    return bool(decode_flags(move) & FLAG_CAPTURE)


def is_promotion(move: int) -> bool:
    return decode_flags(move) >= FLAG_PROMO_KNIGHT


def is_castling(move: int) -> bool:
    flags = decode_flags(move)
    return flags == FLAG_KING_CASTLE or flags == FLAG_QUEEN_CASTLE


def is_ep(move: int) -> bool:
    return decode_flags(move) == FLAG_EP_CAPTURE


def promotion_piece_type(move: int) -> int:
    """Return promotion piece type index (1=N,2=B,3=R,4=Q) or 0 if not a promotion."""
    flags = decode_flags(move)
    if flags >= FLAG_PROMO_KNIGHT:
        return (flags & 3) + 1  # 0->N(1), 1->B(2), 2->R(3), 3->Q(4)
    return 0


def move_to_uci(move: int) -> str:
    from_sq = decode_from(move)
    to_sq = decode_to(move)
    s = square_name(from_sq) + square_name(to_sq)
    flags = decode_flags(move)
    if flags >= FLAG_PROMO_KNIGHT:
        s += PROMO_PIECES[flags & 3]
    return s
