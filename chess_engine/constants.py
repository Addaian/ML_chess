"""Square indices, piece types, masks, and helper constants."""

# --- Square indices (LERF: Little-Endian Rank-File) ---
# a1=0, b1=1, ... h1=7, a2=8, ... h8=63
A1, B1, C1, D1, E1, F1, G1, H1 = range(8)
A2, B2, C2, D2, E2, F2, G2, H2 = range(8, 16)
A3, B3, C3, D3, E3, F3, G3, H3 = range(16, 24)
A4, B4, C4, D4, E4, F4, G4, H4 = range(24, 32)
A5, B5, C5, D5, E5, F5, G5, H5 = range(32, 40)
A6, B6, C6, D6, E6, F6, G6, H6 = range(40, 48)
A7, B7, C7, D7, E7, F7, G7, H7 = range(48, 56)
A8, B8, C8, D8, E8, F8, G8, H8 = range(56, 64)

SQUARE_NAMES = [
    f"{f}{r}" for r in range(1, 9) for f in "abcdefgh"
]

def square_name(sq: int) -> str:
    return SQUARE_NAMES[sq]

def square_from_name(name: str) -> int:
    f = ord(name[0]) - ord('a')
    r = int(name[1]) - 1
    return r * 8 + f

def square_file(sq: int) -> int:
    return sq & 7

def square_rank(sq: int) -> int:
    return sq >> 3

# --- Piece indices ---
WHITE_PAWN = 0
WHITE_KNIGHT = 1
WHITE_BISHOP = 2
WHITE_ROOK = 3
WHITE_QUEEN = 4
WHITE_KING = 5
BLACK_PAWN = 6
BLACK_KNIGHT = 7
BLACK_BISHOP = 8
BLACK_ROOK = 9
BLACK_QUEEN = 10
BLACK_KING = 11

PIECE_NONE = -1

# Side
WHITE = 0
BLACK = 1

# Piece chars for FEN
PIECE_CHARS = "PNBRQKpnbrqk"
CHAR_TO_PIECE = {c: i for i, c in enumerate(PIECE_CHARS)}

def piece_side(piece: int) -> int:
    return BLACK if piece >= 6 else WHITE

def piece_type(piece: int) -> int:
    """Return type index 0-5 (pawn..king) regardless of color."""
    return piece % 6

# --- Rank and file masks ---
FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7

FILES = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H]

RANK_1 = 0xFF
RANK_2 = RANK_1 << 8
RANK_3 = RANK_1 << 16
RANK_4 = RANK_1 << 24
RANK_5 = RANK_1 << 32
RANK_6 = RANK_1 << 40
RANK_7 = RANK_1 << 48
RANK_8 = RANK_1 << 56

RANKS = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8]

BB_ALL = (1 << 64) - 1
BB_EMPTY = 0

# --- Castling rights ---
CASTLE_WK = 1   # White kingside
CASTLE_WQ = 2   # White queenside
CASTLE_BK = 4   # Black kingside
CASTLE_BQ = 8   # Black queenside
CASTLE_ALL = CASTLE_WK | CASTLE_WQ | CASTLE_BK | CASTLE_BQ

# Castling rights update mask table: indexed by square.
# When a piece moves from or to square sq, new_rights &= CASTLE_MASK[sq].
CASTLE_MASK = [0xF] * 64
CASTLE_MASK[E1] = ~(CASTLE_WK | CASTLE_WQ) & 0xF  # King moves
CASTLE_MASK[A1] = ~CASTLE_WQ & 0xF                  # Rook a1
CASTLE_MASK[H1] = ~CASTLE_WK & 0xF                  # Rook h1
CASTLE_MASK[E8] = ~(CASTLE_BK | CASTLE_BQ) & 0xF    # King moves
CASTLE_MASK[A8] = ~CASTLE_BQ & 0xF                   # Rook a8
CASTLE_MASK[H8] = ~CASTLE_BK & 0xF                   # Rook h8
