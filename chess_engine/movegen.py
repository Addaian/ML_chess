"""Pseudo-legal and legal move generation."""

from chess_engine.constants import (
    WHITE, BLACK, BB_EMPTY,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
    RANK_2, RANK_7, RANK_4, RANK_5,
    A1, B1, C1, D1, E1, F1, G1, H1,
    A8, B8, C8, D8, E8, F8, G8, H8,
    square_rank,
)
from chess_engine.bitboard import iter_bits
from chess_engine.move import (
    encode_move,
    FLAG_QUIET, FLAG_DOUBLE_PAWN, FLAG_KING_CASTLE, FLAG_QUEEN_CASTLE,
    FLAG_CAPTURE, FLAG_EP_CAPTURE,
    FLAG_PROMO_KNIGHT, FLAG_PROMO_BISHOP, FLAG_PROMO_ROOK, FLAG_PROMO_QUEEN,
    FLAG_PROMO_CAPTURE_KNIGHT, FLAG_PROMO_CAPTURE_BISHOP,
    FLAG_PROMO_CAPTURE_ROOK, FLAG_PROMO_CAPTURE_QUEEN,
)
from chess_engine.attack_tables import (
    KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS,
    bishop_attacks, rook_attacks, queen_attacks,
    is_square_attacked,
)


def _gen_pawn_moves(board, moves: list) -> None:
    side = board.side
    if side == WHITE:
        pawns = board.piece_bb[WHITE_PAWN]
        enemy = board.occupied_side[BLACK]
        promo_rank = RANK_7   # Pawns on rank 7 will promote
        start_rank = RANK_2   # Double push from rank 2
        push = 8
    else:
        pawns = board.piece_bb[BLACK_PAWN]
        enemy = board.occupied_side[WHITE]
        promo_rank = RANK_2   # Pawns on rank 2 will promote (to rank 1)
        start_rank = RANK_7   # Double push from rank 7
        push = -8

    empty = ~board.occupied & ((1 << 64) - 1)

    for sq in iter_bits(pawns):
        is_promo = bool((1 << sq) & promo_rank)

        # Single push
        to = sq + push
        if 0 <= to < 64 and (1 << to) & empty:
            if is_promo:
                moves.append(encode_move(sq, to, FLAG_PROMO_QUEEN))
                moves.append(encode_move(sq, to, FLAG_PROMO_ROOK))
                moves.append(encode_move(sq, to, FLAG_PROMO_BISHOP))
                moves.append(encode_move(sq, to, FLAG_PROMO_KNIGHT))
            else:
                moves.append(encode_move(sq, to, FLAG_QUIET))
                # Double push
                if (1 << sq) & start_rank:
                    to2 = sq + 2 * push
                    if (1 << to2) & empty:
                        moves.append(encode_move(sq, to2, FLAG_DOUBLE_PAWN))

        # Captures
        for cap_sq in iter_bits(PAWN_ATTACKS[side][sq] & enemy):
            if is_promo:
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_QUEEN))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_ROOK))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_BISHOP))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_KNIGHT))
            else:
                moves.append(encode_move(sq, cap_sq, FLAG_CAPTURE))

        # En passant
        if board.ep_square != -1:
            if (1 << board.ep_square) & PAWN_ATTACKS[side][sq]:
                moves.append(encode_move(sq, board.ep_square, FLAG_EP_CAPTURE))


def _gen_knight_moves(board, moves: list) -> None:
    side = board.side
    knights = board.piece_bb[WHITE_KNIGHT + side * 6]
    friendly = board.occupied_side[side]

    for sq in iter_bits(knights):
        targets = KNIGHT_ATTACKS[sq] & ~friendly
        for to in iter_bits(targets & board.occupied):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))
        for to in iter_bits(targets & ~board.occupied):
            moves.append(encode_move(sq, to, FLAG_QUIET))


def _gen_bishop_moves(board, moves: list) -> None:
    side = board.side
    bishops = board.piece_bb[WHITE_BISHOP + side * 6]
    friendly = board.occupied_side[side]

    for sq in iter_bits(bishops):
        targets = bishop_attacks(sq, board.occupied) & ~friendly
        for to in iter_bits(targets & board.occupied):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))
        for to in iter_bits(targets & ~board.occupied):
            moves.append(encode_move(sq, to, FLAG_QUIET))


def _gen_rook_moves(board, moves: list) -> None:
    side = board.side
    rooks = board.piece_bb[WHITE_ROOK + side * 6]
    friendly = board.occupied_side[side]

    for sq in iter_bits(rooks):
        targets = rook_attacks(sq, board.occupied) & ~friendly
        for to in iter_bits(targets & board.occupied):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))
        for to in iter_bits(targets & ~board.occupied):
            moves.append(encode_move(sq, to, FLAG_QUIET))


def _gen_queen_moves(board, moves: list) -> None:
    side = board.side
    queens = board.piece_bb[WHITE_QUEEN + side * 6]
    friendly = board.occupied_side[side]

    for sq in iter_bits(queens):
        targets = queen_attacks(sq, board.occupied) & ~friendly
        for to in iter_bits(targets & board.occupied):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))
        for to in iter_bits(targets & ~board.occupied):
            moves.append(encode_move(sq, to, FLAG_QUIET))


def _gen_king_moves(board, moves: list) -> None:
    side = board.side
    king_bb = board.piece_bb[WHITE_KING + side * 6]
    king_sq = (king_bb & -king_bb).bit_length() - 1
    friendly = board.occupied_side[side]

    targets = KING_ATTACKS[king_sq] & ~friendly
    for to in iter_bits(targets & board.occupied):
        moves.append(encode_move(king_sq, to, FLAG_CAPTURE))
    for to in iter_bits(targets & ~board.occupied):
        moves.append(encode_move(king_sq, to, FLAG_QUIET))


def _gen_castling_moves(board, moves: list) -> None:
    side = board.side
    occ = board.occupied
    enemy = side ^ 1

    if side == WHITE:
        if board.castling & CASTLE_WK:
            # King on e1, need f1 and g1 empty, e1/f1/g1 not attacked
            if not (occ & ((1 << F1) | (1 << G1))):
                if (not is_square_attacked(E1, enemy, board.piece_bb, occ)
                    and not is_square_attacked(F1, enemy, board.piece_bb, occ)
                    and not is_square_attacked(G1, enemy, board.piece_bb, occ)):
                    moves.append(encode_move(E1, G1, FLAG_KING_CASTLE))
        if board.castling & CASTLE_WQ:
            # Need b1, c1, d1 empty; e1/d1/c1 not attacked
            if not (occ & ((1 << B1) | (1 << C1) | (1 << D1))):
                if (not is_square_attacked(E1, enemy, board.piece_bb, occ)
                    and not is_square_attacked(D1, enemy, board.piece_bb, occ)
                    and not is_square_attacked(C1, enemy, board.piece_bb, occ)):
                    moves.append(encode_move(E1, C1, FLAG_QUEEN_CASTLE))
    else:
        if board.castling & CASTLE_BK:
            if not (occ & ((1 << F8) | (1 << G8))):
                if (not is_square_attacked(E8, enemy, board.piece_bb, occ)
                    and not is_square_attacked(F8, enemy, board.piece_bb, occ)
                    and not is_square_attacked(G8, enemy, board.piece_bb, occ)):
                    moves.append(encode_move(E8, G8, FLAG_KING_CASTLE))
        if board.castling & CASTLE_BQ:
            if not (occ & ((1 << B8) | (1 << C8) | (1 << D8))):
                if (not is_square_attacked(E8, enemy, board.piece_bb, occ)
                    and not is_square_attacked(D8, enemy, board.piece_bb, occ)
                    and not is_square_attacked(C8, enemy, board.piece_bb, occ)):
                    moves.append(encode_move(E8, C8, FLAG_QUEEN_CASTLE))


def generate_pseudo_legal_captures(board) -> list[int]:
    """Generate pseudo-legal captures and promotions only (for quiescence search)."""
    moves: list[int] = []
    side = board.side

    if side == WHITE:
        pawns = board.piece_bb[WHITE_PAWN]
        enemy = board.occupied_side[BLACK]
        promo_rank = RANK_7
        push = 8
    else:
        pawns = board.piece_bb[BLACK_PAWN]
        enemy = board.occupied_side[WHITE]
        promo_rank = RANK_2
        push = -8

    empty = ~board.occupied & ((1 << 64) - 1)

    # Pawn: promotion pushes + captures + EP captures (no quiet pushes/double pushes)
    for sq in iter_bits(pawns):
        is_promo = bool((1 << sq) & promo_rank)
        if is_promo:
            to = sq + push
            if 0 <= to < 64 and (1 << to) & empty:
                moves.append(encode_move(sq, to, FLAG_PROMO_QUEEN))
                moves.append(encode_move(sq, to, FLAG_PROMO_ROOK))
                moves.append(encode_move(sq, to, FLAG_PROMO_BISHOP))
                moves.append(encode_move(sq, to, FLAG_PROMO_KNIGHT))
        for cap_sq in iter_bits(PAWN_ATTACKS[side][sq] & enemy):
            if is_promo:
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_QUEEN))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_ROOK))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_BISHOP))
                moves.append(encode_move(sq, cap_sq, FLAG_PROMO_CAPTURE_KNIGHT))
            else:
                moves.append(encode_move(sq, cap_sq, FLAG_CAPTURE))
        if board.ep_square != -1:
            if (1 << board.ep_square) & PAWN_ATTACKS[side][sq]:
                moves.append(encode_move(sq, board.ep_square, FLAG_EP_CAPTURE))

    # Sliding + leaping pieces: captures only
    friendly = board.occupied_side[side]

    for sq in iter_bits(board.piece_bb[WHITE_KNIGHT + side * 6]):
        for to in iter_bits(KNIGHT_ATTACKS[sq] & enemy):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))

    for sq in iter_bits(board.piece_bb[WHITE_BISHOP + side * 6]):
        for to in iter_bits(bishop_attacks(sq, board.occupied) & enemy):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))

    for sq in iter_bits(board.piece_bb[WHITE_ROOK + side * 6]):
        for to in iter_bits(rook_attacks(sq, board.occupied) & enemy):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))

    for sq in iter_bits(board.piece_bb[WHITE_QUEEN + side * 6]):
        for to in iter_bits(queen_attacks(sq, board.occupied) & enemy):
            moves.append(encode_move(sq, to, FLAG_CAPTURE))

    king_bb = board.piece_bb[WHITE_KING + side * 6]
    king_sq = (king_bb & -king_bb).bit_length() - 1
    for to in iter_bits(KING_ATTACKS[king_sq] & enemy):
        moves.append(encode_move(king_sq, to, FLAG_CAPTURE))

    return moves


def generate_legal_captures(board) -> list[int]:
    """Legal captures + promotions only (for quiescence search)."""
    legal = []
    for move in generate_pseudo_legal_captures(board):
        if board.make_move(move):
            board.unmake_move()
            legal.append(move)
    return legal


def generate_pseudo_legal_moves(board) -> list[int]:
    moves: list[int] = []
    _gen_pawn_moves(board, moves)
    _gen_knight_moves(board, moves)
    _gen_bishop_moves(board, moves)
    _gen_rook_moves(board, moves)
    _gen_queen_moves(board, moves)
    _gen_king_moves(board, moves)
    _gen_castling_moves(board, moves)
    return moves


def generate_legal_moves(board) -> list[int]:
    legal = []
    for move in generate_pseudo_legal_moves(board):
        if board.make_move(move):
            board.unmake_move()
            legal.append(move)
    return legal
