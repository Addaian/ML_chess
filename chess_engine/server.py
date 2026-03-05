"""HTTP server + JSON API for the chess engine web UI."""

import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from chess_engine import (
    Board,
    generate_legal_moves,
    move_to_uci,
    decode_from,
    decode_to,
    decode_flags,
    find_best_move,
    evaluate,
)
from chess_engine.constants import PIECE_CHARS, PIECE_NONE, WHITE, square_name
from chess_engine.move import (
    is_capture as move_is_capture,
    is_promotion as move_is_promotion,
    promotion_piece_type,
)

STATIC_DIR = Path(__file__).parent / "static"

_FLAG_KING_CASTLE = 2
_FLAG_QUEEN_CASTLE = 3


def resolve_uci_move(board, uci):
    for move in generate_legal_moves(board):
        if move_to_uci(move) == uci:
            return move
    return None


def move_to_san(board, move):
    """Simplified algebraic notation. Call BEFORE applying the move."""
    from_sq = decode_from(move)
    to_sq = decode_to(move)
    flags = decode_flags(move)

    if flags == _FLAG_KING_CASTLE:
        return "O-O"
    if flags == _FLAG_QUEEN_CASTLE:
        return "O-O-O"

    piece = board.mailbox[from_sq]
    piece_char = PIECE_CHARS[piece].upper()
    target = square_name(to_sq)
    capture = move_is_capture(move)

    if piece_char == "P":
        san = (square_name(from_sq)[0] + "x" + target) if capture else target
    else:
        san = piece_char + ("x" if capture else "") + target

    if move_is_promotion(move):
        san += "=" + " NBRQ"[promotion_piece_type(move)]

    return san


def add_check_suffix(board, san):
    """Add + or # suffix. Call AFTER applying the move."""
    if board.is_in_check():
        legal = generate_legal_moves(board)
        return san + ("#" if not legal else "+")
    return san


def game_status(board):
    legal = generate_legal_moves(board)
    in_check = board.is_in_check()
    if not legal:
        return ("checkmate", True) if in_check else ("stalemate", False)
    return ("check", True) if in_check else ("playing", False)


def game_state_json(board):
    status, in_check = game_status(board)
    pieces = []
    for sq in range(64):
        p = board.mailbox[sq]
        pieces.append(PIECE_CHARS[p] if p != PIECE_NONE else None)

    legal_moves = {}
    for move in generate_legal_moves(board):
        uci = move_to_uci(move)
        legal_moves.setdefault(uci[:2], []).append(uci)

    return {
        "fen": board.fen(),
        "pieces": pieces,
        "side": "w" if board.side == WHITE else "b",
        "status": status,
        "in_check": in_check,
        "eval": evaluate(board),
        "legal_moves": legal_moves,
        "fullmove": board.fullmove,
    }


class ChessHandler(BaseHTTPRequestHandler):
    board = Board()
    move_history = []

    def log_message(self, fmt, *args):
        pass

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, ct):
        try:
            content = path.read_bytes()
        except FileNotFoundError:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_GET(self):
        if self.path == "/":
            self._file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/new":
            ChessHandler.board = Board()
            ChessHandler.move_history = []
            self._json({"state": game_state_json(ChessHandler.board), "move_history": []})
        elif self.path == "/api/state":
            self._json({"state": game_state_json(ChessHandler.board), "move_history": ChessHandler.move_history})
        elif self.path == "/api/undo":
            self._handle_undo()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/api/move":
            self._handle_move()
        else:
            self.send_error(404)

    def _handle_move(self):
        try:
            data = json.loads(self._body())
        except (json.JSONDecodeError, ValueError):
            return self._json({"error": "Invalid JSON"}, 400)

        uci = data.get("uci")
        depth = data.get("depth", 3)
        if not uci:
            return self._json({"error": "Missing 'uci' field"}, 400)

        board = ChessHandler.board
        move = resolve_uci_move(board, uci)
        if move is None:
            return self._json({"error": f"Illegal move: {uci}"}, 400)

        # Human move
        human_san = move_to_san(board, move)
        human_capture = move_is_capture(move)
        board.make_move(move)
        human_san = add_check_suffix(board, human_san)
        ChessHandler.move_history.append({"uci": uci, "san": human_san, "capture": human_capture})

        # Engine response
        status, _ = game_status(board)
        engine_uci = None
        engine_capture = False
        engine_time = 0

        if status in ("playing", "check"):
            t0 = time.time()
            engine_move = find_best_move(board, depth)
            engine_time = round(time.time() - t0, 3)
            if engine_move is not None:
                engine_san = move_to_san(board, engine_move)
                engine_capture = move_is_capture(engine_move)
                engine_uci = move_to_uci(engine_move)
                board.make_move(engine_move)
                engine_san = add_check_suffix(board, engine_san)
                ChessHandler.move_history.append({"uci": engine_uci, "san": engine_san, "capture": engine_capture})

        self._json({
            "human_move": uci, "human_capture": human_capture,
            "engine_move": engine_uci, "engine_capture": engine_capture,
            "engine_time": engine_time,
            "state": game_state_json(board),
            "move_history": ChessHandler.move_history,
        })

    def _handle_undo(self):
        board = ChessHandler.board
        history = ChessHandler.move_history
        if not history:
            return self._json({"error": "Nothing to undo"}, 400)
        undo_count = 1 if len(history) % 2 == 1 else 2
        for _ in range(undo_count):
            if not history:
                break
            board.unmake_move()
            history.pop()
        self._json({"state": game_state_json(board), "move_history": history})


def main():
    parser = argparse.ArgumentParser(description="Chess Engine Web UI")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = HTTPServer(("", args.port), ChessHandler)
    print(f"Chess engine server running at http://localhost:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
