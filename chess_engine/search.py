"""Negamax search with alpha-beta pruning, quiescence, move ordering, TT, iterative deepening."""

from chess_engine.constants import (
    WHITE, BLACK, PIECE_NONE,
    WHITE_PAWN, WHITE_KNIGHT, WHITE_BISHOP, WHITE_ROOK, WHITE_QUEEN, WHITE_KING,
    piece_type, piece_side,
)
from chess_engine.evaluator import evaluate
from chess_engine.movegen import generate_legal_moves, generate_legal_captures, generate_pseudo_legal_moves
from chess_engine.move import decode_from, decode_to, decode_flags, is_capture, is_promotion
from chess_engine.tt import TTable, EXACT, UPPER, LOWER

CHECKMATE_SCORE = 1_000_000
STALEMATE_SCORE = 0
MAX_PLY = 64

# MVV-LVA piece values: P=1, N=3, B=3, R=5, Q=9, K=10
_MVV_LVA = [1, 3, 3, 5, 9, 10]


class SearchInfo:
    """Mutable state shared across one search invocation."""
    __slots__ = ('tt', 'killers', 'history', 'nodes', 'ply')

    def __init__(self, tt: TTable):
        self.tt = tt
        self.killers: list[list[int]] = [[0, 0] for _ in range(MAX_PLY)]
        self.history: list[list[int]] = [[0] * 64 for _ in range(12)]
        self.nodes: int = 0
        self.ply: int = 0


# ── Public API ──────────────────────────────────────────────────────────────

def find_best_move(board, depth: int):
    """Return best move (16-bit int) for the side to move, or None."""
    tt = TTable()
    return _iterative_deepening(board, depth, tt)


def minimax(board, depth: int, maximizing: bool) -> int:
    """Backward-compatible wrapper.

    Score is always from White's perspective (positive = White better).
    *maximizing* must be True when it is White's turn.
    """
    moves = generate_legal_moves(board)

    # Terminal: no legal moves
    if not moves:
        if board.is_in_check():
            return -CHECKMATE_SCORE if maximizing else CHECKMATE_SCORE
        return STALEMATE_SCORE

    # Leaf node
    if depth == 0:
        return evaluate(board)

    tt = TTable()
    info = SearchInfo(tt)
    score = _negamax(board, depth, -CHECKMATE_SCORE - 1, CHECKMATE_SCORE + 1, info)
    # _negamax returns side-to-move POV; convert to White POV
    if board.side == BLACK:
        score = -score
    return score


# ── Internals ───────────────────────────────────────────────────────────────

def _iterative_deepening(board, max_depth: int, tt: TTable):
    """Iterative deepening: search depths 1..max_depth, return best move found."""
    best_move = None
    for d in range(1, max_depth + 1):
        info = SearchInfo(tt)
        _, move = _search_root(board, d, info)
        if move is not None:
            best_move = move
    return best_move


def _search_root(board, depth: int, info: SearchInfo):
    """Root search: returns (score_white_pov, best_move)."""
    moves = generate_legal_moves(board)
    if not moves:
        if board.is_in_check():
            score = -CHECKMATE_SCORE if board.side == WHITE else CHECKMATE_SCORE
        else:
            score = STALEMATE_SCORE
        return score, None

    # Order moves using TT best move if available
    tt_move = 0
    entry = info.tt.probe(board.hash_key)
    if entry is not None:
        tt_move = entry.best_move

    moves.sort(key=lambda m: _score_move(board, m, info, tt_move), reverse=True)

    alpha = -CHECKMATE_SCORE - 1
    beta = CHECKMATE_SCORE + 1
    best_move = moves[0]
    best_score = alpha

    info.ply = 0
    for move in moves:
        board.make_move(move)
        info.ply = 1
        score = -_negamax(board, depth - 1, -beta, -alpha, info)
        board.unmake_move()
        info.ply = 0
        if score > best_score:
            best_score = score
            best_move = move
            if score > alpha:
                alpha = score

    # Convert to White POV
    white_score = best_score if board.side == WHITE else -best_score
    return white_score, best_move


def _negamax(board, depth: int, alpha: int, beta: int, info: SearchInfo) -> int:
    """Negamax with alpha-beta. Returns score from side-to-move POV."""
    info.nodes += 1
    original_alpha = alpha

    # TT probe
    tt_move = 0
    entry = info.tt.probe(board.hash_key)
    if entry is not None and entry.depth >= depth:
        tt_move = entry.best_move
        if entry.flag == EXACT:
            return entry.score
        elif entry.flag == LOWER:
            alpha = max(alpha, entry.score)
        elif entry.flag == UPPER:
            beta = min(beta, entry.score)
        if alpha >= beta:
            return entry.score

    # Drop into quiescence at depth 0
    if depth <= 0:
        return _quiescence(board, alpha, beta, info)

    moves = generate_legal_moves(board)

    # Terminal: checkmate or stalemate
    if not moves:
        if board.is_in_check():
            return -(CHECKMATE_SCORE - info.ply)
        return STALEMATE_SCORE

    # Order moves
    if not tt_move and entry is not None:
        tt_move = entry.best_move
    moves.sort(key=lambda m: _score_move(board, m, info, tt_move), reverse=True)

    best_score = -CHECKMATE_SCORE - 1
    best_move = moves[0]

    saved_ply = info.ply
    for move in moves:
        board.make_move(move)
        info.ply = saved_ply + 1
        score = -_negamax(board, depth - 1, -beta, -alpha, info)
        board.unmake_move()
        info.ply = saved_ply

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        if alpha >= beta:
            # Beta cutoff — update killers and history for quiet moves
            flags = decode_flags(move)
            if not is_capture(move) and not is_promotion(move):
                # Killer move
                ply = saved_ply
                if info.killers[ply][0] != move:
                    info.killers[ply][1] = info.killers[ply][0]
                    info.killers[ply][0] = move
                # History heuristic
                from_sq = decode_from(move)
                to_sq = decode_to(move)
                piece = board.mailbox[from_sq]
                if piece != PIECE_NONE:
                    info.history[piece][to_sq] += depth * depth
            info.tt.store(board.hash_key, depth, best_score, LOWER, best_move)
            return best_score

    # Store in TT
    if best_score <= original_alpha:
        flag = UPPER
    elif best_score >= beta:
        flag = LOWER
    else:
        flag = EXACT
    info.tt.store(board.hash_key, depth, best_score, flag, best_move)

    return best_score


def _quiescence(board, alpha: int, beta: int, info: SearchInfo) -> int:
    """Capture-only search to resolve tactical noise."""
    info.nodes += 1

    # Generate captures first so checkmate is always detectable regardless
    # of the alpha-beta window (stand_pat >= beta could mask checkmate otherwise).
    captures = generate_legal_captures(board)

    if not captures:
        # No captures — detect checkmate (in check with no legal moves at all)
        if board.is_in_check() and not generate_legal_moves(board):
            return -(CHECKMATE_SCORE - info.ply)
        # Stand-pat: no captures to make, return static evaluation
        static = evaluate(board)
        stand_pat = static if board.side == WHITE else -static
        return stand_pat

    # Stand-pat score for pruning (only when captures are available)
    static = evaluate(board)
    stand_pat = static if board.side == WHITE else -static

    if stand_pat >= beta:
        return beta
    if stand_pat > alpha:
        alpha = stand_pat

    # Order captures by MVV-LVA
    captures.sort(key=lambda m: _mvv_lva_score(board, m), reverse=True)

    saved_ply = info.ply
    for move in captures:
        board.make_move(move)
        info.ply = saved_ply + 1
        score = -_quiescence(board, -beta, -alpha, info)
        board.unmake_move()
        info.ply = saved_ply

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def _mvv_lva_score(board, move: int) -> int:
    """MVV-LVA: victim value * 100 - attacker value. Higher = better capture."""
    to_sq = decode_to(move)
    from_sq = decode_from(move)
    victim = board.mailbox[to_sq]
    attacker = board.mailbox[from_sq]
    if victim == PIECE_NONE or attacker == PIECE_NONE:
        return 0
    return _MVV_LVA[piece_type(victim)] * 100 - _MVV_LVA[piece_type(attacker)]


def _score_move(board, move: int, info: SearchInfo, tt_move: int) -> int:
    """Move ordering score. Higher = searched first."""
    if move == tt_move:
        return 10_000_000

    if is_capture(move) or is_promotion(move):
        return 1_000_000 + _mvv_lva_score(board, move)

    ply = info.ply
    if ply < MAX_PLY:
        if move == info.killers[ply][0]:
            return 900_000
        if move == info.killers[ply][1]:
            return 800_000

    from_sq = decode_from(move)
    to_sq = decode_to(move)
    piece = board.mailbox[from_sq]
    if piece != PIECE_NONE:
        return info.history[piece][to_sq]
    return 0
