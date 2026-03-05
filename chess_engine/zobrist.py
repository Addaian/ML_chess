"""Zobrist hashing tables for incremental position hashing."""

import random

from chess_engine.bitboard import iter_bits
from chess_engine.constants import BLACK

random.seed(0xBEEF)  # deterministic for reproducibility

# 12 pieces × 64 squares
PIECE_KEYS: list[list[int]] = [
    [random.getrandbits(64) for _ in range(64)] for _ in range(12)
]
SIDE_KEY: int = random.getrandbits(64)          # XOR when Black to move
CASTLE_KEYS: list[int] = [random.getrandbits(64) for _ in range(16)]  # 4-bit → 16 combos
EP_KEYS: list[int] = [random.getrandbits(64) for _ in range(8)]        # EP file 0-7
NO_EP_KEY: int = random.getrandbits(64)


def compute_hash(board) -> int:
    """Full hash from scratch (used once at Board init)."""
    h = 0
    for piece in range(12):
        for sq in iter_bits(board.piece_bb[piece]):
            h ^= PIECE_KEYS[piece][sq]
    if board.side == BLACK:
        h ^= SIDE_KEY
    h ^= CASTLE_KEYS[board.castling]
    if board.ep_square != -1:
        h ^= EP_KEYS[board.ep_square & 7]
    else:
        h ^= NO_EP_KEY
    return h
