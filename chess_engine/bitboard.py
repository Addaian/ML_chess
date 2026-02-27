"""Bitboard utility functions using native Python ints."""

from chess_engine.constants import BB_ALL


def set_bit(bb: int, sq: int) -> int:
    return bb | (1 << sq)


def clear_bit(bb: int, sq: int) -> int:
    return bb & ~(1 << sq)


def get_bit(bb: int, sq: int) -> bool:
    return bool(bb & (1 << sq))


def iter_bits(bb: int):
    """Yield each set bit index (LSB first)."""
    while bb:
        sq = (bb & -bb).bit_length() - 1
        yield sq
        bb &= bb - 1


def bitscan_forward(bb: int) -> int:
    """Return index of least significant set bit, or -1 if empty."""
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


def popcount(bb: int) -> int:
    return bb.bit_count()


def print_bitboard(bb: int) -> str:
    """Return a human-readable 8x8 grid (rank 8 on top)."""
    lines = []
    for rank in range(7, -1, -1):
        row = []
        for file in range(8):
            sq = rank * 8 + file
            row.append("1" if bb & (1 << sq) else ".")
        lines.append(f"{rank + 1}  {' '.join(row)}")
    lines.append("   a b c d e f g h")
    return "\n".join(lines)
