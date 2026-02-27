"""Chess engine — bitboard-based board representation with legal move generation."""

from chess_engine.board import Board
from chess_engine.movegen import generate_legal_moves, generate_pseudo_legal_moves
from chess_engine.move import move_to_uci, encode_move, decode_from, decode_to, decode_flags
from chess_engine.perft import perft, divide
from chess_engine.fen import STARTING_FEN

__all__ = [
    "Board",
    "generate_legal_moves",
    "generate_pseudo_legal_moves",
    "move_to_uci",
    "encode_move",
    "decode_from",
    "decode_to",
    "decode_flags",
    "perft",
    "divide",
    "STARTING_FEN",
]
