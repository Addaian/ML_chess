"""FEN parsing and serialization."""

from chess_engine.constants import (
    WHITE, BLACK, PIECE_NONE,
    CHAR_TO_PIECE, PIECE_CHARS,
    CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ,
    square_from_name, square_name, square_rank, square_file,
)

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def parse_fen(fen: str) -> dict:
    """Parse a FEN string into a dict with board setup info.

    Returns dict with keys:
        pieces: list[int] of 64 elements (PIECE_NONE for empty)
        side: WHITE or BLACK
        castling: int (bitmask)
        ep_square: int or -1
        halfmove: int
        fullmove: int
    """
    parts = fen.split()
    board_str = parts[0]
    side_str = parts[1]
    castle_str = parts[2]
    ep_str = parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    pieces = [PIECE_NONE] * 64
    rank = 7
    file = 0
    for ch in board_str:
        if ch == '/':
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            sq = rank * 8 + file
            pieces[sq] = CHAR_TO_PIECE[ch]
            file += 1

    side = WHITE if side_str == 'w' else BLACK

    castling = 0
    if castle_str != '-':
        if 'K' in castle_str:
            castling |= CASTLE_WK
        if 'Q' in castle_str:
            castling |= CASTLE_WQ
        if 'k' in castle_str:
            castling |= CASTLE_BK
        if 'q' in castle_str:
            castling |= CASTLE_BQ

    ep_square = -1
    if ep_str != '-':
        ep_square = square_from_name(ep_str)

    return {
        'pieces': pieces,
        'side': side,
        'castling': castling,
        'ep_square': ep_square,
        'halfmove': halfmove,
        'fullmove': fullmove,
    }


def board_to_fen(board) -> str:
    """Serialize a Board object to a FEN string."""
    parts = []

    # Piece placement
    rows = []
    for rank in range(7, -1, -1):
        row = ""
        empty = 0
        for file in range(8):
            sq = rank * 8 + file
            piece = board.mailbox[sq]
            if piece == PIECE_NONE:
                empty += 1
            else:
                if empty:
                    row += str(empty)
                    empty = 0
                row += PIECE_CHARS[piece]
        if empty:
            row += str(empty)
        rows.append(row)
    parts.append("/".join(rows))

    # Side to move
    parts.append('w' if board.side == WHITE else 'b')

    # Castling
    castle = ""
    if board.castling & CASTLE_WK:
        castle += 'K'
    if board.castling & CASTLE_WQ:
        castle += 'Q'
    if board.castling & CASTLE_BK:
        castle += 'k'
    if board.castling & CASTLE_BQ:
        castle += 'q'
    parts.append(castle if castle else '-')

    # En passant
    parts.append(square_name(board.ep_square) if board.ep_square != -1 else '-')

    # Halfmove and fullmove
    parts.append(str(board.halfmove))
    parts.append(str(board.fullmove))

    return " ".join(parts)
