"""Convert a Board object to a 768-element float32 feature tensor.

Layout: piece_index * 64 + square
Piece indices: 0-5 = White (P,N,B,R,Q,K), 6-11 = Black (P,N,B,R,Q,K)
Square indexing: LERF — a1=0, b1=1, ..., h8=63
"""

import numpy as np


def board_to_tensor(board) -> np.ndarray:
    """Convert Board to (768,) float32 numpy array."""
    features = np.zeros(768, dtype=np.float32)
    for piece in range(12):
        bb = board.piece_bb[piece]
        base = piece * 64
        while bb:
            sq = (bb & -bb).bit_length() - 1
            features[base + sq] = 1.0
            bb &= bb - 1
    return features
