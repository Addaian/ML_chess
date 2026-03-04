# ML_chess

A chess engine that evolves through three versions — from hand-coded evaluation to neural network — with measurable Elo progression at each stage.

---

## Elo Progression

| Version | Description | Est. Elo | Jump |
|---|---|---|---|
| v1 | Minimax + hand-coded eval (depth 2) | ~840 | — |
| v2 | Alpha-beta + TT + move ordering + quiescence (depth 2) | ~1400–1450 | +560–610 |
| v3 | Alpha-beta + NN evaluator, trained on 5M Lichess positions (depth 2) | ~1450–1500 | +50 |

---

## Getting Started

```bash
git clone <repo>
cd ML_chess
pip install numpy         # runtime dependency
pip install pytest        # for tests

# Run the engine from CLI
python -m chess_engine.engine "<FEN>" [depth]

# Run all tests
python -m pytest tests/ -v

# Run web server (localhost:5000)
python -m chess_engine.server
```

To use the hand-coded evaluator instead of the neural network:
```bash
CHESS_EVAL=hce python -m chess_engine.engine "<FEN>"
```

---

## Build Plan

### Phase 1: Board Representation & Move Generation ✓

The foundation everything else sits on.

**Core data structures**
- **Bitboard** representation — 12 × 64-bit integers, one per piece type per color. Compact and fast using bitwise operations.
- Board state tracks: piece positions, whose turn, castling rights, en passant square, half-move clock (50-move rule), Zobrist hash

**Move generation**
- Full legal move generation including castling, en passant, pawn promotion, and pin detection
- Verified with **perft tests** — counts all possible positions to depth N against known correct values

---

### Phase 2: Minimax Engine (v1) ✓

```python
function minimax(position, depth, is_maximising):
    if depth == 0:
        return evaluate(position)

    if is_maximising:
        best = -infinity
        for move in legal_moves(position):
            score = minimax(make_move(position, move), depth-1, False)
            best = max(best, score)
        return best
    else:
        # mirror for minimising player
```

**Hand-coded evaluator**
- Material count: pawn=100, knight=320, bishop=330, rook=500, queen=900 centipawns
- Piece-square tables (Michniewski values): bonus/penalty for piece placement
- Positional terms: bishop pair, rooks on open/semi-open files, passed pawns, king pawn shield

**Phase 2 result (depth 2, 10 games/level vs Stockfish 18):**

| Opponent | Score | Est. Elo |
|---|---|---|
| SF Skill 0 depth 1 (~900) | 40% | 830 |
| SF Skill 0 depth 3 (~1100) | 0% | 423 |
| SF Skill 0 depth 6 (~1250) | 15% | 949 |
| SF UCI Elo 1320 | 5% | 808 |
| SF UCI Elo 1500 | 0% | 823 |

**Weighted Elo estimate: ~840**

---

### Phase 3: Alpha-Beta Pruning (v2) ✓

```python
function alphabeta(position, depth, alpha, beta, is_maximising):
    if depth == 0:
        return evaluate(position)

    for move in legal_moves(position):
        score = alphabeta(make_move(position, move), depth-1, alpha, beta, not is_maximising)

        if is_maximising:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

        if beta <= alpha:
            break  # prune — opponent would never let this happen

    return alpha if is_maximising else beta
```

**Optimisations implemented**
- **Move ordering** — TT best move → captures (MVV-LVA) → killer moves → history heuristic
- **Transposition table** — Zobrist hashing, depth-preferred replacement, ~1M entries
- **Iterative deepening** — search depth 1→N, reusing TT results to order moves better each iteration
- **Quiescence search** — extends search through captures at leaf nodes to avoid horizon effect

**Phase 3 result (depth 2, 10 games/level vs Stockfish 18):**

| Opponent | Score | Est. Elo |
|---|---|---|
| SF Skill 0 depth 1 (~900) | 100% | 1577 |
| SF Skill 0 depth 3 (~1100) | 50% | 1100 |
| SF Skill 0 depth 6 (~1250) | 80% | 1491 |
| SF UCI Elo 1320 | 80% | 1561 |
| SF UCI Elo 1500 | 45% | 1465 |

**Weighted Elo estimate: ~1352 · Best estimate (UCI brackets): ~1400–1450**

---

### Phase 4: Neural Network Evaluator (v3) ✓

Replace the hand-coded eval function with a network trained on Stockfish evaluations. The search logic is unchanged — only `evaluate()` is swapped.

**Data pipeline**
- Source: Lichess standard rated games PGN (January 2024, ~30 GB), with `[%eval X.XX]` Stockfish annotations
- Parsed 5M positions from 983K games; filtered to `|eval| ≤ 1500 cp`, skipping mate scores
- Labels normalised to `[-1, 1]`: `label = clamp(cp, ±1500) / 1500`

**Model architecture**
```
Input: 768 features (12 piece planes × 64 squares, binary)
→ Linear(768, 256) → ReLU
→ Linear(256, 128) → ReLU
→ Linear(128,  64) → ReLU
→ Linear( 64,   1) → Tanh     # output ∈ [-1, +1], scaled ×1500 → centipawns
```

**Training**
- Optimiser: Adam (lr=1e-3, weight_decay=1e-4), batch size 4096, 10 epochs
- MSE loss vs Stockfish eval; 90/10 train/val split
- Best val loss: 0.01915

**Runtime inference**: weights exported to numpy `.npz`; forward pass runs with pure numpy (no PyTorch at inference time, ~20–80 µs per call).

**Phase 4 result (depth 2, 10 games/level vs Stockfish 18):**

| Opponent | Score | Est. Elo |
|---|---|---|
| SF Skill 0 depth 1 (~900) | 80% | 1141 |
| SF Skill 0 depth 3 (~1100) | 90% | 1482 |
| SF Skill 0 depth 6 (~1250) | 90% | 1632 |
| SF UCI Elo 1320 | 75% | 1511 |
| SF UCI Elo 1500 | 50% | 1500 |

**Weighted Elo estimate: ~1450 · Best estimate (UCI brackets): ~1450–1500**

**Going further (AlphaZero-style)**
- Add a **policy head** alongside the value head — the network outputs a probability distribution over moves
- Use **Monte Carlo Tree Search (MCTS)** instead of alpha-beta — the policy head guides which branches to explore, the value head scores leaf nodes
- Larger jump in complexity but makes for an extraordinary project

---

### Phase 5: Web App

**Backend (FastAPI)**
- REST endpoint: receive FEN string, return best move
- Engine runs server-side; server already partially implemented in `chess_engine/server.py`

**Frontend (React)**
- `chess.js` — handles game logic, validates moves
- `react-chessboard` — renders the board
- On each player move, send FEN to backend, get engine response, update board

---

## Tech Stack

| Component | Technology |
|---|---|
| Engine core | Python 3.11 |
| Board representation | Bitboards (12 × 64-bit integers) |
| Search | Negamax alpha-beta, TT, iterative deepening, quiescence |
| Evaluator (v2) | Hand-coded: material + PST + positional terms |
| Evaluator (v3) | PyTorch (training) → numpy (inference) |
| Training data | Lichess standard rated games PGN |
| Backend API | FastAPI (partial) |
| Frontend | React + react-chessboard (planned) |
| Deployment | Hugging Face Spaces (planned) |

---

## Architecture

```
chess_engine/
  board.py       — Board state, make/unmake move, Zobrist hashing
  movegen.py     — Legal move generation (bitboard-based)
  search.py      — Negamax alpha-beta, TT, move ordering, quiescence
  evaluator.py   — Dispatcher: NN eval (v3) with HCE fallback (v2)
  nn/
    features.py  — board_to_tensor(): Board → 768-float numpy array
    model.py     — NumpyChessNet (runtime) + ChessNet/export_weights (training)
    weights.npz  — Trained model weights
  server.py      — FastAPI backend
  engine.py      — CLI entry point

training/
  download_data.py — Download Lichess PGN
  parse_data.py    — PGN → .npz training chunks
  train.py         — Training loop, exports weights.npz

tests/           — pytest suite (58 tests: perft, movegen, search, evaluator, NN)
elo_test.py      — Elo estimation vs Stockfish (configurable depth/games)
```
