#Tests for the Rust engine extension (salmon_nnue_rs). They validate that the
#Rust hot loop mirrors the pure-Python engine: hash/UCI parity now, and
#inference + search parity as those land. Skips cleanly when the extension is
#not built, so the suite runs on a pure-Python checkout.
#Runnable with pytest, or directly: `python test_rust_engine.py`

import random

import numpy as np
import cozy_chess as cc

import common
from nnue import mcts, network

try:
    import salmon_nnue_rs
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

_NET = network.make_random_net(0)


def _skip():
    print("  (salmon_nnue_rs not built - skipped)")


def _make_engine():
    def b(a):
        return np.ascontiguousarray(a, dtype=np.float32).tobytes()
    return salmon_nnue_rs.RustEngine(
        b(_NET.ft_w), b(_NET.ft_b), b(_NET.l1_w), b(_NET.l1_b),
        b(_NET.v_w), float(_NET.v_b), b(_NET.p_w), b(_NET.p_b))


#Rust board.hash() must equal cozy_chess_py's over random games — this is what
#lets game-history repetition hashes pass straight through to the Rust search.
def test_hash_parity():
    if not AVAILABLE:
        return _skip()
    random.seed(7)
    for _ in range(8):
        board = cc.Board()
        for _ in range(40):
            assert salmon_nnue_rs.zobrist_hash(board.fen()) == board.hash(), \
                f"hash mismatch at {board.fen()}"
            moves = list(board.generate_moves())
            if not moves:
                break
            board.play(random.choice(moves))


#Rust legal_moves_uci must match common.to_uci over generate_moves as ORDERED
#lists (search tie-breaks depend on identical move order), covering castling
#and promotion UCI normalization.
def test_uci_parity():
    if not AVAILABLE:
        return _skip()
    random.seed(11)
    fixtures = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",      # castling both sides
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",            # promotion
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant
    ]
    boards = [cc.Board.from_fen(f) for f in fixtures]
    for _ in range(6):                               # plus random-game positions
        b = cc.Board()
        for _ in range(30):
            boards.append(cc.Board.from_fen(b.fen()))
            ms = list(b.generate_moves())
            if not ms:
                break
            b.play(random.choice(ms))
    for board in boards:
        py = [common.to_uci(board, m) for m in board.generate_moves()]
        rs = salmon_nnue_rs.legal_moves_uci(board.fen())
        assert py == rs, f"uci order mismatch at {board.fen()}:\n py={py}\n rs={rs}"


#Rust value + blended priors must match the NumPy net (nnue.network /
#mcts._priors) within tight tolerance on varied positions.
def test_inference_parity():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",   # castling + ep available
        "4k3/4q3/8/8/8/8/3PPP2/4K3 w - - 0 1",        # in check
    ]
    for fen in fens:
        board = cc.Board.from_fen(fen)
        acc = network.Accumulator.from_board(_NET, board)
        v_py, logits = _NET.value_policy(acc, board.side_to_move())
        moves = list(board.generate_moves())
        py = dict(zip([common.to_uci(board, m) for m in moves],
                      mcts._priors(board, moves, logits)))
        v_rs, rs = eng.eval_debug(fen)
        rs = dict(rs)
        assert abs(v_py - v_rs) < 1e-4, f"value {fen}: {v_py} vs {v_rs}"
        for uci, p in py.items():
            assert abs(p - rs[uci]) < 1e-5, f"prior {uci} {fen}: {p} vs {rs[uci]}"


#The Rust incremental accumulator (Accumulator::play, incl. castling fallback
#and en passant) must agree with a full recompute after every move: play a
#random game move-by-move incrementally, compare each value to eval_debug.
def test_incremental_accumulator():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    random.seed(5)
    for _ in range(15):
        board = cc.Board()
        played = []
        for _ in range(40):
            moves = list(board.generate_moves())
            if not moves:
                break
            mv = random.choice(moves)
            played.append(common.to_uci(board, mv))
            board.play(mv)
            v_inc = eng.eval_after_moves(cc.Board().fen(), played)   # incremental
            v_full, _ = eng.eval_debug(board.fen())                  # full recompute
            assert abs(v_inc - v_full) < 1e-5, \
                f"acc drift after {played[-1]} at {board.fen()}: {v_inc} vs {v_full}"


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
