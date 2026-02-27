"""Tests for bitboard utility functions."""

from chess_engine.bitboard import set_bit, clear_bit, get_bit, iter_bits, bitscan_forward, popcount


def test_set_and_get_bit():
    bb = 0
    bb = set_bit(bb, 0)
    assert get_bit(bb, 0)
    assert not get_bit(bb, 1)

    bb = set_bit(bb, 63)
    assert get_bit(bb, 63)
    assert not get_bit(bb, 32)


def test_clear_bit():
    bb = set_bit(0, 10)
    bb = set_bit(bb, 20)
    bb = clear_bit(bb, 10)
    assert not get_bit(bb, 10)
    assert get_bit(bb, 20)


def test_iter_bits():
    bb = set_bit(0, 0)
    bb = set_bit(bb, 5)
    bb = set_bit(bb, 63)
    assert list(iter_bits(bb)) == [0, 5, 63]


def test_iter_bits_empty():
    assert list(iter_bits(0)) == []


def test_bitscan_forward():
    assert bitscan_forward(0) == -1
    assert bitscan_forward(1) == 0
    assert bitscan_forward(1 << 42) == 42
    assert bitscan_forward(0b1100) == 2


def test_popcount():
    assert popcount(0) == 0
    assert popcount(0xFF) == 8
    assert popcount((1 << 64) - 1) == 64
