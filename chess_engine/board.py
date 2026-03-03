"""Board class — 12 piece bitboards, 3 occupancy bitboards, mailbox, game state."""

from dataclasses import dataclass

from chess_engine.constants import (
    WHITE, BLACK, PIECE_NONE,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    BLACK_PAWN, BLACK_KNIGHT, BLACK_BISHOP, BLACK_ROOK, BLACK_QUEEN, BLACK_KING,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ, CASTLE_MASK,
    A1, C1, D1, E1, F1, G1, H1,
    A8, C8, D8, E8, F8, G8, H8,
    piece_side, piece_type,
)
from chess_engine.bitboard import set_bit, clear_bit, get_bit, bitscan_forward
from chess_engine.move import (
    decode_from, decode_to, decode_flags,
    FLAG_QUIET, FLAG_DOUBLE_PAWN, FLAG_KING_CASTLE, FLAG_QUEEN_CASTLE,
    FLAG_CAPTURE, FLAG_EP_CAPTURE, FLAG_PROMO_KNIGHT,
    is_capture, is_promotion, is_castling, is_ep, promotion_piece_type,
)
from chess_engine.attack_tables import is_square_attacked
from chess_engine.fen import parse_fen, board_to_fen, STARTING_FEN


@dataclass(slots=True)
class _UndoInfo:
    move: int
    captured_piece: int
    castling: int
    ep_square: int
    halfmove: int
    hash_key: int


class Board:
    __slots__ = (
        'piece_bb', 'occupied', 'occupied_side',
        'mailbox', 'side', 'castling', 'ep_square',
        'halfmove', 'fullmove', 'history', 'hash_key',
    )

    def __init__(self, fen: str = STARTING_FEN):
        self.piece_bb: list[int] = [0] * 12
        self.occupied: int = 0
        self.occupied_side: list[int] = [0, 0]
        self.mailbox: list[int] = [PIECE_NONE] * 64
        self.side: int = WHITE
        self.castling: int = 0
        self.ep_square: int = -1
        self.halfmove: int = 0
        self.fullmove: int = 1
        self.history: list[_UndoInfo] = []
        self.hash_key: int = 0
        self._load_fen(fen)

    def _load_fen(self, fen: str) -> None:
        info = parse_fen(fen)
        self.side = info['side']
        self.castling = info['castling']
        self.ep_square = info['ep_square']
        self.halfmove = info['halfmove']
        self.fullmove = info['fullmove']

        self.piece_bb = [0] * 12
        self.occupied = 0
        self.occupied_side = [0, 0]
        self.mailbox = [PIECE_NONE] * 64

        for sq, piece in enumerate(info['pieces']):
            if piece != PIECE_NONE:
                self._place_piece(piece, sq)

        # Compute Zobrist hash after all pieces are placed
        from chess_engine.zobrist import compute_hash
        self.hash_key = compute_hash(self)

    def _place_piece(self, piece: int, sq: int) -> None:
        self.piece_bb[piece] = set_bit(self.piece_bb[piece], sq)
        side = piece_side(piece)
        self.occupied_side[side] = set_bit(self.occupied_side[side], sq)
        self.occupied = set_bit(self.occupied, sq)
        self.mailbox[sq] = piece

    def _remove_piece(self, piece: int, sq: int) -> None:
        self.piece_bb[piece] = clear_bit(self.piece_bb[piece], sq)
        side = piece_side(piece)
        self.occupied_side[side] = clear_bit(self.occupied_side[side], sq)
        self.occupied = clear_bit(self.occupied, sq)
        self.mailbox[sq] = PIECE_NONE

    def _move_piece(self, piece: int, from_sq: int, to_sq: int) -> None:
        mask = (1 << from_sq) | (1 << to_sq)
        self.piece_bb[piece] ^= mask
        side = piece_side(piece)
        self.occupied_side[side] ^= mask
        self.occupied ^= mask
        self.mailbox[from_sq] = PIECE_NONE
        self.mailbox[to_sq] = piece

    def make_move(self, move: int) -> bool:
        """Apply move. Returns True if the move is legal (doesn't leave king in check)."""
        from chess_engine.zobrist import PIECE_KEYS, SIDE_KEY, CASTLE_KEYS, EP_KEYS, NO_EP_KEY

        from_sq = decode_from(move)
        to_sq = decode_to(move)
        flags = decode_flags(move)
        moving_piece = self.mailbox[from_sq]
        captured_piece = PIECE_NONE

        h = self.hash_key

        # XOR out old castling and EP keys
        h ^= CASTLE_KEYS[self.castling]
        if self.ep_square != -1:
            h ^= EP_KEYS[self.ep_square & 7]
        else:
            h ^= NO_EP_KEY

        # Save undo info (with current hash, before modifications)
        undo = _UndoInfo(
            move=move,
            captured_piece=PIECE_NONE,
            castling=self.castling,
            ep_square=self.ep_square,
            halfmove=self.halfmove,
            hash_key=self.hash_key,
        )

        # Handle captures
        if flags == FLAG_EP_CAPTURE:
            if self.side == WHITE:
                cap_sq = to_sq - 8
                captured_piece = BLACK_PAWN
            else:
                cap_sq = to_sq + 8
                captured_piece = WHITE_PAWN
            self._remove_piece(captured_piece, cap_sq)
            h ^= PIECE_KEYS[captured_piece][cap_sq]
        elif is_capture(move):
            captured_piece = self.mailbox[to_sq]
            self._remove_piece(captured_piece, to_sq)
            h ^= PIECE_KEYS[captured_piece][to_sq]

        undo.captured_piece = captured_piece

        # Move the piece: XOR out from_sq, XOR in to_sq
        h ^= PIECE_KEYS[moving_piece][from_sq]
        h ^= PIECE_KEYS[moving_piece][to_sq]
        self._move_piece(moving_piece, from_sq, to_sq)

        # Handle promotions
        if is_promotion(move):
            promo_type = promotion_piece_type(move)  # 1=N,2=B,3=R,4=Q
            promo_piece = (self.side * 6) + promo_type
            # XOR out the pawn at to_sq, XOR in the promoted piece
            h ^= PIECE_KEYS[moving_piece][to_sq]
            h ^= PIECE_KEYS[promo_piece][to_sq]
            self._remove_piece(moving_piece, to_sq)
            self._place_piece(promo_piece, to_sq)

        # Handle castling — move the rook
        if flags == FLAG_KING_CASTLE:
            if self.side == WHITE:
                h ^= PIECE_KEYS[WHITE_ROOK][H1]
                h ^= PIECE_KEYS[WHITE_ROOK][F1]
                self._move_piece(WHITE_ROOK, H1, F1)
            else:
                h ^= PIECE_KEYS[BLACK_ROOK][H8]
                h ^= PIECE_KEYS[BLACK_ROOK][F8]
                self._move_piece(BLACK_ROOK, H8, F8)
        elif flags == FLAG_QUEEN_CASTLE:
            if self.side == WHITE:
                h ^= PIECE_KEYS[WHITE_ROOK][A1]
                h ^= PIECE_KEYS[WHITE_ROOK][D1]
                self._move_piece(WHITE_ROOK, A1, D1)
            else:
                h ^= PIECE_KEYS[BLACK_ROOK][A8]
                h ^= PIECE_KEYS[BLACK_ROOK][D8]
                self._move_piece(BLACK_ROOK, A8, D8)

        # Update castling rights
        self.castling &= CASTLE_MASK[from_sq] & CASTLE_MASK[to_sq]

        # Update en passant square
        if flags == FLAG_DOUBLE_PAWN:
            self.ep_square = (from_sq + to_sq) // 2
        else:
            self.ep_square = -1

        # Update halfmove clock
        if piece_type(moving_piece) == 0 or captured_piece != PIECE_NONE:
            self.halfmove = 0
        else:
            self.halfmove += 1

        # Update fullmove counter
        if self.side == BLACK:
            self.fullmove += 1

        # Switch side
        self.side ^= 1
        h ^= SIDE_KEY

        # XOR in new castling and EP keys
        h ^= CASTLE_KEYS[self.castling]
        if self.ep_square != -1:
            h ^= EP_KEYS[self.ep_square & 7]
        else:
            h ^= NO_EP_KEY

        self.hash_key = h

        # Check legality — did we leave our king in check?
        king_sq = bitscan_forward(self.piece_bb[WHITE_KING + (self.side ^ 1) * 6])
        if is_square_attacked(king_sq, self.side, self.piece_bb, self.occupied):
            # Illegal — undo
            self.history.append(undo)
            self.unmake_move()
            return False

        self.history.append(undo)
        return True

    def unmake_move(self) -> None:
        undo = self.history.pop()
        move = undo.move

        # Switch side back
        self.side ^= 1

        from_sq = decode_from(move)
        to_sq = decode_to(move)
        flags = decode_flags(move)
        moving_piece = self.mailbox[to_sq]

        # Undo promotion — replace promoted piece with pawn
        if is_promotion(move):
            promo_piece = self.mailbox[to_sq]
            self._remove_piece(promo_piece, to_sq)
            pawn_piece = WHITE_PAWN if self.side == WHITE else BLACK_PAWN
            self._place_piece(pawn_piece, to_sq)
            moving_piece = pawn_piece

        # Move piece back
        self._move_piece(moving_piece, to_sq, from_sq)

        # Undo castling rook move
        if flags == FLAG_KING_CASTLE:
            if self.side == WHITE:
                self._move_piece(WHITE_ROOK, F1, H1)
            else:
                self._move_piece(BLACK_ROOK, F8, H8)
        elif flags == FLAG_QUEEN_CASTLE:
            if self.side == WHITE:
                self._move_piece(WHITE_ROOK, D1, A1)
            else:
                self._move_piece(BLACK_ROOK, D8, A8)

        # Restore captured piece
        if flags == FLAG_EP_CAPTURE:
            if self.side == WHITE:
                cap_sq = to_sq - 8
            else:
                cap_sq = to_sq + 8
            self._place_piece(undo.captured_piece, cap_sq)
        elif undo.captured_piece != PIECE_NONE:
            self._place_piece(undo.captured_piece, to_sq)

        # Restore state
        self.castling = undo.castling
        self.ep_square = undo.ep_square
        self.halfmove = undo.halfmove
        if self.side == BLACK:
            self.fullmove -= 1

        # Restore hash
        self.hash_key = undo.hash_key

    def is_in_check(self) -> bool:
        king_sq = bitscan_forward(self.piece_bb[WHITE_KING + self.side * 6])
        return is_square_attacked(king_sq, self.side ^ 1, self.piece_bb, self.occupied)

    def fen(self) -> str:
        return board_to_fen(self)

    def __repr__(self) -> str:
        return f"Board('{self.fen()}')"
