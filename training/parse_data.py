#!/usr/bin/env python3
"""Parse a Lichess .pgn.zst file into (features, label) numpy .npz chunks.

Each game in the PGN has Stockfish eval comments like { [%eval 0.23] } after moves.
We replay the game position by position and extract the eval for each annotated move.

Usage:
    python3 -m training.parse_data training/data/lichess_db_standard_rated_2024-01.pgn.zst
    python3 -m training.parse_data <file.pgn.zst> --out training/data/parsed --max 3000000
"""

import argparse
import os
import re
import numpy as np

from chess_engine.board import Board
from chess_engine.move import move_to_uci
from chess_engine.movegen import generate_legal_moves
from chess_engine.nn.features import board_to_tensor
from training.config import EVAL_CLAMP, CHUNK_SIZE

# Matches [%eval 0.23] or [%eval -1.50] (centipawn/100 from White's POV)
# Does NOT match [%eval #3] (mate scores — skipped)
_EVAL_RE = re.compile(r'\[%eval ([+-]?\d+\.\d+)\]')

# Piece letter → type index (for SAN parsing)
_PIECE_LETTER = {'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5}
_PROMO_LETTER = {'n': 1, 'b': 2, 'r': 3, 'q': 4}


# ── SAN → move conversion ─────────────────────────────────────────────────────

def _san_to_move(board: Board, san: str):
    """Return the internal move int matching a SAN string, or None."""
    from chess_engine.move import decode_from, decode_to, decode_flags
    from chess_engine.constants import piece_type, piece_side, WHITE

    san_clean = san.rstrip('+#!? ').strip()
    if not san_clean:
        return None

    legal = generate_legal_moves(board)

    for move in legal:
        if _move_matches_san(board, move, san_clean):
            return move
    return None


def _move_matches_san(board: Board, move: int, san: str) -> bool:
    from chess_engine.move import decode_from, decode_to, decode_flags
    from chess_engine.constants import piece_type, piece_side, WHITE

    from_sq = decode_from(move)
    to_sq = decode_to(move)
    flags = decode_flags(move)

    files = 'abcdefgh'
    ranks = '12345678'

    # Castling
    if san in ('O-O', '0-0'):
        return flags == 2
    if san in ('O-O-O', '0-0-0'):
        return flags == 3

    # Destination square (last two chars before optional promotion)
    san_work = san
    promo_type = None
    if '=' in san_work:
        promo_type = _PROMO_LETTER.get(san_work[-1].lower())
        san_work = san_work[:san_work.index('=')]
    elif flags in (5, 6, 7, 8) and len(san_work) >= 1 and san_work[-1].lower() in _PROMO_LETTER:
        promo_type = _PROMO_LETTER[san_work[-1].lower()]
        san_work = san_work[:-1]

    # Check destination
    if len(san_work) < 2:
        return False
    dest = san_work[-2:]
    if len(dest) != 2 or dest[0] not in files or dest[1] not in ranks:
        return False
    exp_to = files.index(dest[0]) + ranks.index(dest[1]) * 8
    if to_sq != exp_to:
        return False

    # Check promotion
    if promo_type is not None:
        if flags not in (5, 6, 7, 8):
            return False
        if promo_type != (flags - 4):
            return False
    elif flags in (5, 6, 7, 8):
        return False  # promotion move but no promo in SAN

    # Check piece type
    piece = board.mailbox[from_sq]
    if piece < 0:
        return False
    pt = piece_type(piece)

    if san_work and san_work[0].isupper() and san_work[0] in _PIECE_LETTER:
        if pt != _PIECE_LETTER[san_work[0]]:
            return False
    else:
        if pt != 0:  # pawn
            return False

    # Disambiguation (e.g. Nbd2, R1e1)
    disambig = san_work[1:-2] if san_work[0].isupper() and san_work[0] in _PIECE_LETTER else san_work[:-2]
    disambig = disambig.lstrip('x')  # strip capture 'x'
    if disambig:
        from_file = from_sq % 8
        from_rank = from_sq // 8
        if disambig[0] in files and files.index(disambig[0]) != from_file:
            return False
        if disambig[0] in ranks and ranks.index(disambig[0]) != from_rank:
            return False

    return True


# ── Game parser ───────────────────────────────────────────────────────────────

def _parse_game(moves_text: str):
    """Yield (features, label) pairs from the moves section of one PGN game."""
    board = Board()
    tokens = re.split(r'(\{[^}]*\})', moves_text)
    pending_eval = None

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if token.startswith('{'):
            m = _EVAL_RE.search(token)
            if m:
                pending_eval = float(m.group(1)) * 100  # convert to centipawns
            else:
                pending_eval = None  # mate score or unannotated — skip
            continue

        # Token is a moves string — process word by word
        for word in token.split():
            # Skip move numbers (e.g. "1.", "12...")
            if re.match(r'^\d+\.+$', word):
                continue
            # Skip results
            if word in ('1-0', '0-1', '1/2-1/2', '*'):
                return

            # Emit pending eval for the position BEFORE this move
            if pending_eval is not None:
                cp = pending_eval
                if abs(cp) <= EVAL_CLAMP:
                    label = np.float32(cp / EVAL_CLAMP)
                    features = board_to_tensor(board)
                    yield features, label
                pending_eval = None

            # Make the move
            move = _san_to_move(board, word)
            if move is None:
                return  # unparseable — abandon game
            board.make_move(move)


# ── File processor ────────────────────────────────────────────────────────────

def process_file(pgn_zst_path: str, out_dir: str, max_positions: int = 5_000_000):
    """Stream-parse a .pgn.zst file and write .npz chunks to out_dir."""
    import zstandard as zstd

    os.makedirs(out_dir, exist_ok=True)

    chunk_features: list[np.ndarray] = []
    chunk_labels: list[np.float32] = []
    chunk_idx = 0
    total = 0
    games = 0

    dctx = zstd.ZstdDecompressor()
    buffer = ""

    with open(pgn_zst_path, 'rb') as f:
        reader = dctx.stream_reader(f)
        while total < max_positions:
            raw = reader.read(1 << 20)  # 1 MB
            if not raw:
                break
            buffer += raw.decode('utf-8', errors='replace')

            # Split on game boundaries (blank line before a tag section)
            parts = re.split(r'\n\n(?=\[)', buffer)
            buffer = parts[-1]

            # Pair up header + moves for each complete game
            i = 0
            while i < len(parts) - 1:
                header_block = parts[i]
                i += 1
                # The moves section is the next block if it doesn't start with '['
                if i < len(parts) - 1 and not parts[i].lstrip().startswith('['):
                    moves_block = parts[i]
                    i += 1
                else:
                    moves_block = ""

                games += 1
                try:
                    for features, label in _parse_game(moves_block):
                        chunk_features.append(features)
                        chunk_labels.append(label)
                        total += 1
                        if len(chunk_features) >= CHUNK_SIZE:
                            _save_chunk(chunk_features, chunk_labels, out_dir, chunk_idx)
                            chunk_idx += 1
                            chunk_features, chunk_labels = [], []
                        if total >= max_positions:
                            break
                except Exception:
                    pass

                if total >= max_positions:
                    break

            if total >= max_positions:
                break

    if chunk_features:
        _save_chunk(chunk_features, chunk_labels, out_dir, chunk_idx)
        chunk_idx += 1

    print(f"\nGames processed : {games:,}")
    print(f"Positions saved : {total:,}")
    print(f"Chunks written  : {chunk_idx}")


def _save_chunk(features, labels, out_dir, idx):
    path = os.path.join(out_dir, f"chunk_{idx:04d}.npz")
    np.savez_compressed(
        path,
        features=np.stack(features),
        labels=np.array(labels, dtype=np.float32),
    )
    print(f"  chunk_{idx:04d}.npz  ({len(features):,} positions)", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse Lichess PGN.zst → .npz training chunks")
    parser.add_argument("pgn_zst", help="Path to .pgn.zst file")
    parser.add_argument("--out", default="training/data/parsed",
                        help="Output directory for .npz chunks")
    parser.add_argument("--max", type=int, default=5_000_000,
                        help="Maximum positions to extract")
    args = parser.parse_args()
    process_file(args.pgn_zst, args.out, args.max)
