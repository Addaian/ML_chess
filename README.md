# ML_chess

A chess engine that evolves through three versions — from hand-coded evaluation to neural network — with measurable Elo progression at each stage.

<!-- TODO: Add a GIF of the engine playing here -->
<!-- TODO: Add Elo progression chart (v1 → v2 → v3) -->

---

## Build Plan

### Phase 1: Board Representation & Move Generation

The foundation everything else sits on. Get this right before touching search or ML.

**Core data structures**
- Represent the board as a **bitboard** — 12 x 64-bit integers, one per piece type per color. Compact and fast using bitwise operations. Alternatively, an 8x8 array is simpler to start but slower.
- Board state object should track: piece positions, whose turn, castling rights, en passant square, half-move clock (for 50-move rule)

**Move generation**
- Generate all legal moves from a given position
- Handle edge cases: castling, en passant, pawn promotion, pins (a piece is pinned if moving it exposes your king)
- Write a **perft test** — a standard chess programming benchmark that counts all possible positions to depth N. There are known correct values you can check against. This is how you know your move generator is bug-free before you build anything on top of it.

---

### Phase 2: Minimax Engine (v1)

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

**Hand-coded evaluator to start**
- Material count: assign point values (pawn=1, knight=3, bishop=3, rook=5, queen=9)
- Piece-square tables: bonus/penalty for where each piece stands (knights want the centre, kings want the corner in middlegame, etc.) — hardcoded 8x8 arrays, widely available online

At this point you have a working, beatable chess engine. Measure its strength using **Elo estimation** by playing it against Stockfish at very low depth settings.

---

### Phase 3: Alpha-Beta Pruning (v2)

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

**Additional optimisations**
- **Move ordering** — search captures and checks first, since they're more likely to cause cutoffs and make alpha-beta prune more aggressively
- **Transposition table** — a hash map (Zobrist hashing) that caches positions you've already evaluated so you don't re-search them
- **Iterative deepening** — search to depth 1, then 2, then 3, using the previous result to order moves better each time

Measure Elo again. The jump from v1 to v2 should be dramatic.

---

### Phase 4: Neural Network Evaluator (v3)

Replace the hand-coded eval function with a trained network.

**Data**
- Download the **Lichess open database** (lichess.org/database) — millions of real games with Stockfish centipawn evaluations attached
- Parse PGN files into (board state → evaluation score) pairs
- Represent each board state as an **input tensor**: 12 x 8 x 8 (one binary plane per piece type per color) = 768 input features

**Model architecture (start simple)**
```python
Input: 768 features (12 piece planes x 64 squares)
→ Dense(256, relu)
→ Dense(128, relu)
→ Dense(64, relu)
→ Dense(1, tanh)   # output: score from -1 (black winning) to +1 (white winning)
```

Train with MSE loss against Stockfish evaluations using PyTorch.

**Plugging it in**
- Export the trained model
- Replace the `evaluate(position)` call in alpha-beta with a forward pass through the network
- The search logic stays identical — only the eval function changes

**Going further (AlphaZero-style)**
- Add a **policy head** alongside the value head — the network also outputs a probability distribution over moves
- Use **Monte Carlo Tree Search (MCTS)** instead of alpha-beta — the policy head guides which branches to explore, the value head scores leaf nodes
- Larger jump in complexity but makes for an extraordinary project

---

### Phase 5: Web App

**Backend (Python/FastAPI)**
- Expose a REST endpoint: receive FEN string (standard chess position notation), return best move
- Run the engine server-side

**Frontend (React)**
- `chess.js` — handles game logic, validates moves
- `react-chessboard` — renders the board
- On each player move, send the FEN to the backend, get the engine's response, update the board

---

## Tech Stack

| Component | Technology |
|---|---|
| Engine core | Python (fast to build) or Rust (fast to run) |
| ML training | PyTorch |
| Backend API | FastAPI |
| Frontend | React + react-chessboard |
| Deployment | Hugging Face Spaces (free, great for ML demos) |

---

## Architecture

<!-- TODO: Add architecture diagram -->

---

## Getting Started

<!-- TODO: Add local setup instructions -->

---

## Elo Progression

<!-- TODO: Add chart showing v1 → v2 → v3 Elo improvement -->
