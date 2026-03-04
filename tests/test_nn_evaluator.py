"""Tests for Phase 4 neural network evaluator infrastructure."""

import numpy as np
import pytest

from chess_engine.board import Board
from chess_engine.nn.features import board_to_tensor


# ── Feature extraction ────────────────────────────────────────────────────────

def test_feature_shape():
    board = Board()
    features = board_to_tensor(board)
    assert features.shape == (768,)
    assert features.dtype == np.float32


def test_starting_position_has_32_pieces():
    board = Board()
    features = board_to_tensor(board)
    assert int(features.sum()) == 32


def test_only_set_bits_are_ones():
    board = Board()
    features = board_to_tensor(board)
    assert set(features.tolist()).issubset({0.0, 1.0})


def test_single_king_board():
    board = Board("8/8/8/8/8/8/8/4K3 w - - 0 1")
    features = board_to_tensor(board)
    assert int(features.sum()) == 1


def test_white_king_e1_index():
    # White King is piece index 5; e1 = square 4
    # Feature index = 5 * 64 + 4 = 324
    board = Board()
    features = board_to_tensor(board)
    assert features[5 * 64 + 4] == 1.0


def test_black_king_e8_index():
    # Black King is piece index 11; e8 = square 60
    # Feature index = 11 * 64 + 60 = 764
    board = Board()
    features = board_to_tensor(board)
    assert features[11 * 64 + 60] == 1.0


def test_white_pawn_e2_index():
    # White Pawn is piece index 0; e2 = square 12
    # Feature index = 0 * 64 + 12 = 12
    board = Board()
    features = board_to_tensor(board)
    assert features[0 * 64 + 12] == 1.0


def test_empty_board_no_pieces():
    board = Board("8/8/8/8/8/8/8/8 w - - 0 1")
    # Board() allows an empty board; if it raises, skip
    features = board_to_tensor(board)
    assert int(features.sum()) == 0


# ── evaluate() interface ──────────────────────────────────────────────────────

def test_evaluate_returns_int():
    """evaluate() must always return an int."""
    from chess_engine.evaluator import evaluate
    board = Board()
    result = evaluate(board)
    assert isinstance(result, int)


def test_evaluate_starting_position_reasonable():
    """Starting position should be close to 0 (neither side winning)."""
    import os
    os.environ["CHESS_EVAL"] = "hce"
    import importlib
    import chess_engine.evaluator as ev
    importlib.reload(ev)

    board = Board()
    result = ev.evaluate(board)
    assert isinstance(result, int)
    assert result == 0  # HCE gives exactly 0 for starting position

    # Restore
    os.environ.pop("CHESS_EVAL", None)


def test_evaluate_fallback_no_crash(monkeypatch):
    """evaluate() falls back to HCE without crashing when weights are missing."""
    import chess_engine.evaluator as ev
    monkeypatch.setattr(ev, '_nn_model', None)
    monkeypatch.setattr(ev, '_nn_loaded', False)
    monkeypatch.setattr(ev, '_NN_ENABLED', True)

    # Patch _load_nn to do nothing (simulate missing weights file)
    monkeypatch.setattr(ev, '_load_nn', lambda: None)

    board = Board()
    result = ev.evaluate(board)
    assert isinstance(result, int)
