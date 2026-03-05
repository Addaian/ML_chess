#!/usr/bin/env python3
"""
Estimate ML_chess engine Elo by playing it against Stockfish at
calibrated strength levels (Skill Level 0 + depth, and UCI_LimitStrength).

Usage:
    python3 elo_test.py              # engine at depth 2, 10 games/level
    python3 elo_test.py --depth 3   # slower but stronger engine
    python3 elo_test.py --games 20  # more games = tighter estimate
"""

import argparse
import math
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

from chess_engine import Board, generate_legal_moves, find_best_move
from chess_engine.server import resolve_uci_move
from chess_engine.constants import WHITE, BLACK

# ── Tunable defaults ─────────────────────────────────────────────────────────
DEFAULT_DEPTH = 2   # our engine search depth
DEFAULT_GAMES = 10  # games per Stockfish level (half as White, half as Black)
MAX_PLY       = 300 # safety cutoff (150 full moves)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Level:
    label: str
    ref_elo: int          # approximate Elo for this SF configuration
    skill: Optional[int]  # 0-20 (SF Skill Level), None = don't set
    uci_elo: Optional[int]  # 1320-3190 (UCI_LimitStrength), None = don't set
    sf_depth: Optional[int] = None    # `go depth N`
    movetime_ms: Optional[int] = None # `go movetime N`


# Tested levels span a wide range so we can bracket our engine.
# ref_elo values for Skill-Level entries are community approximations (±200).
LEVELS = [
    Level("SF Skill 0  depth 1",  ref_elo=900,  skill=0, uci_elo=None, sf_depth=1),
    Level("SF Skill 0  depth 3",  ref_elo=1100, skill=0, uci_elo=None, sf_depth=3),
    Level("SF Skill 0  depth 6",  ref_elo=1250, skill=0, uci_elo=None, sf_depth=6),
    Level("SF UCI Elo 1320",      ref_elo=1320, skill=None, uci_elo=1320, movetime_ms=100),
    Level("SF UCI Elo 1500",      ref_elo=1500, skill=None, uci_elo=1500, movetime_ms=100),
]


# ── Stockfish process wrapper ─────────────────────────────────────────────────

class Stockfish:
    def __init__(self, level: Level):
        self.level = level
        self.proc = subprocess.Popen(
            ["stockfish"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._expect("uciok")
        self._send("setoption name Threads value 1")
        self._send("setoption name Hash value 1")

        if level.uci_elo is not None:
            self._send("setoption name UCI_LimitStrength value true")
            self._send(f"setoption name UCI_Elo value {level.uci_elo}")
        else:
            self._send("setoption name UCI_LimitStrength value false")
            if level.skill is not None:
                self._send(f"setoption name Skill Level value {level.skill}")

        self._send("isready")
        self._expect("readyok")

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _expect(self, prefix: str) -> str:
        while True:
            line = self.proc.stdout.readline().strip()
            if line.startswith(prefix):
                return line

    def best_move(self, fen: str) -> Optional[str]:
        self._send("position fen " + fen)
        if self.level.movetime_ms is not None:
            self._send(f"go movetime {self.level.movetime_ms}")
        else:
            self._send(f"go depth {self.level.sf_depth}")
        line = self._expect("bestmove")
        parts = line.split()
        mv = parts[1] if len(parts) >= 2 else None
        return None if mv in (None, "(none)") else mv

    def new_game(self):
        self._send("ucinewgame")

    def close(self):
        try:
            self._send("quit")
            self.proc.wait(timeout=3)
        except Exception:
            self.proc.kill()


# ── Game logic ────────────────────────────────────────────────────────────────

def play_game(sf: Stockfish, our_color: int, our_depth: int) -> int:
    """
    Play a single game.
    Returns +1 (our win), 0 (draw), -1 (our loss).
    """
    board = Board()
    sf.new_game()

    for _ in range(MAX_PLY):
        legal = generate_legal_moves(board)

        if not legal:
            if board.is_in_check():          # checkmate
                return 1 if board.side != our_color else -1
            return 0                          # stalemate

        if board.halfmove >= 100:            # 50-move rule
            return 0

        if board.side == our_color:
            move = find_best_move(board, our_depth)
            if move is None:
                return 0
        else:
            uci = sf.best_move(board.fen())
            if uci is None:
                return 0
            move = resolve_uci_move(board, uci)
            if move is None:
                return 0

        board.make_move(move)

    return 0  # move-limit draw


# ── Statistics ────────────────────────────────────────────────────────────────

def score_pct(w, d, l) -> float:
    n = w + d + l
    return (w + 0.5 * d) / n * 100 if n else 0.0


def elo_from_score(ref_elo: int, w: int, d: int, l: int) -> Optional[float]:
    n = w + d + l
    if n == 0:
        return None
    s = w + 0.5 * d
    # Clamp to avoid log(0) / log(neg)
    if s <= 0:
        return float(ref_elo - 677)   # ~1% score lower bound
    if s >= n:
        return float(ref_elo + 677)   # ~99% score upper bound
    return ref_elo + 400 * math.log10(s / (n - s))


# ── Match runner ──────────────────────────────────────────────────────────────

def run_match(level: Level, games: int, our_depth: int):
    sf = Stockfish(level)
    w = d = l = 0
    for i in range(games):
        our_color = WHITE if i % 2 == 0 else BLACK
        side_str  = "White" if our_color == WHITE else "Black"
        result    = play_game(sf, our_color, our_depth)
        if result == 1:
            w += 1; tag = "Win "
        elif result == -1:
            l += 1; tag = "Loss"
        else:
            d += 1; tag = "Draw"
        print(f"    [{i+1:2}/{games}] as {side_str:5}  {tag}   W{w} D{d} L{l}", flush=True)
    sf.close()
    return w, d, l


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="Engine search depth")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="Games per level (even = balanced colors)")
    args = parser.parse_args()

    our_depth = args.depth
    games     = args.games + (args.games % 2)   # round up to even

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║         ML_chess Elo Estimation vs Stockfish 18          ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Engine depth : {our_depth:<3}  │  Games / level : {games:<3}                 ║")
    est_time = len(LEVELS) * games * 40 * 0.13   # rough estimate
    print(f"║  Est. runtime : {est_time/60:.0f}-{est_time/60*1.5:.0f} min                              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    results = []
    total_t  = time.time()

    for level in LEVELS:
        print(f"  ▸ {level.label}  (ref Elo ≈ {level.ref_elo})")
        t0      = time.time()
        w, d, l = run_match(level, games, our_depth)
        elapsed = time.time() - t0
        elo     = elo_from_score(level.ref_elo, w, d, l)
        pct     = score_pct(w, d, l)
        results.append((level, w, d, l, elo))
        print(f"    → Score {pct:.0f}%  ({w}W {d}D {l}L)  "
              f"Est Elo: {elo:.0f}  [{elapsed:.0f}s]")
        print()

    # ── Summary ──────────────────────────────────────────────────────────────
    W = 62
    print("═" * W)
    print("  SUMMARY")
    print("═" * W)
    print(f"  {'Opponent':<24} {'W':>3} {'D':>3} {'L':>3}  {'Score':>6}  {'Est. Elo':>10}")
    print("  " + "─" * (W - 2))

    elo_vals, weights = [], []
    for level, w, d, l, elo in results:
        pct  = score_pct(w, d, l)
        estr = f"{elo:.0f}" if elo else "—"
        print(f"  {level.label:<24} {w:>3} {d:>3} {l:>3}  {pct:>5.0f}%  {estr:>10}")
        if elo is not None:
            elo_vals.append(elo)
            # Matches near 50% score are most statistically reliable
            wt = 1.0 - abs(pct - 50) / 50
            weights.append(max(0.05, wt))

    print("  " + "─" * (W - 2))

    if elo_vals:
        avg = sum(elo_vals) / len(elo_vals)
        wt_elo = sum(e * wt for e, wt in zip(elo_vals, weights)) / sum(weights)

        # Find the Elo band where we cross 50%
        bracket_lo = bracket_hi = None
        for i in range(len(results) - 1):
            s0 = score_pct(*results[i][1:4])
            s1 = score_pct(*results[i+1][1:4])
            if s0 >= 50 >= s1 or s0 <= 50 <= s1:
                bracket_lo = results[i][0].ref_elo
                bracket_hi = results[i+1][0].ref_elo
                break

        print()
        print(f"  Weighted Elo estimate : ~{wt_elo:.0f}")
        print(f"  Simple average        : ~{avg:.0f}")
        if bracket_lo is not None:
            print(f"  50%-score bracket     : {bracket_lo}–{bracket_hi}")
        print()
        print("  Note: Skill-Level reference Elos are approximate (±200).")
        print("  UCI_LimitStrength results (1320, 1500) are more reliable.")

    total_elapsed = time.time() - total_t
    print(f"\n  Total time: {total_elapsed:.0f}s")
    print("═" * W)


if __name__ == "__main__":
    main()
