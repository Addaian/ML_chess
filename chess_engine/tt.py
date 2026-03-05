"""Transposition table with depth-preferred replacement."""

EXACT = 0   # Score is exact
UPPER = 1   # Failed low (score <= alpha) — upper bound
LOWER = 2   # Failed high (score >= beta)  — lower bound


class TTEntry:
    __slots__ = ('key', 'depth', 'score', 'flag', 'best_move')

    def __init__(self):
        self.key: int = 0
        self.depth: int = -1
        self.score: int = 0
        self.flag: int = 0
        self.best_move: int = 0


class TTable:
    def __init__(self, size: int = 1 << 20):  # ~1M entries
        self.mask = size - 1
        self.table = [TTEntry() for _ in range(size)]

    def probe(self, key: int):
        """Return entry if it matches key and is valid, else None."""
        e = self.table[key & self.mask]
        return e if e.key == key and e.depth >= 0 else None

    def store(self, key: int, depth: int, score: int, flag: int, best_move: int) -> None:
        """Store with depth-preferred replacement."""
        e = self.table[key & self.mask]
        if depth >= e.depth or key != e.key:
            e.key = key
            e.depth = depth
            e.score = score
            e.flag = flag
            e.best_move = best_move
