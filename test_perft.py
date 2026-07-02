#Perft and differential tests for the cozy-chess move generator.
#Runnable with pytest, or directly: `python test_perft.py`
#
#perft counts the leaf nodes of the move tree to a fixed depth; matching the
#published counts (and python-chess) validates move generation exactly.

import copy
import random

import cozy_chess as cc
import chess

import minimax


def _perft(board, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.generate_moves():
        child = copy.copy(board)
        child.play(move)
        total += _perft(child, depth - 1)
    return total


#(fen, depth, expected) from the standard perft suite (chessprogramming.org).
_PERFT_CASES = [
    (None, 4, 197281),                                                            # start position
    ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862),  # Kiwipete
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238),                      # position 3
    ("r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 3, 9467),  # position 4
]


def test_perft_published():
    for fen, depth, expected in _PERFT_CASES:
        board = cc.Board() if fen is None else cc.Board.from_fen(fen)
        got = _perft(board, depth)
        assert got == expected, f"perft({depth}) {fen}: {got} != {expected}"


def test_cozy_matches_python_chess():
    #Over many random games, cozy and python-chess must agree on the legal move
    #set at every position (python-chess is the oracle).
    random.seed(7)
    for _ in range(40):
        cb = cc.Board()
        pb = chess.Board()
        for _ in range(40):
            cozy_moves = sorted(minimax.to_uci(cb, m) for m in cb.generate_moves())
            py_moves = sorted(m.uci() for m in pb.legal_moves)
            assert cozy_moves == py_moves, f"move mismatch at {pb.fen()}:\n {cozy_moves}\n {py_moves}"
            if not py_moves:
                break
            uci = random.choice(py_moves)
            cb.play(minimax.from_uci(cb, uci))
            pb.push_uci(uci)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
