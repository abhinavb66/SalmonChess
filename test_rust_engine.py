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


def _make_engine(quant=False, tree_reuse=False):
    def b(a):
        return np.ascontiguousarray(a, dtype=np.float32).tobytes()
    return salmon_nnue_rs.RustEngine(
        b(_NET.ft_w), b(_NET.ft_b), b(_NET.l1_w), b(_NET.l1_b),
        b(_NET.v_w), float(_NET.v_b), b(_NET.p_w), b(_NET.p_b), quant,
        tree_reuse)


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


#The int8 path is approximate by design (u8 activations, i8 weights, per-row
#scales): assert it tracks the exact f32 engine closely on value and priors —
#loose enough for quantization error, tight enough to catch a scale/sign bug.
def test_quant_inference_close():
    if not AVAILABLE:
        return _skip()
    exact = _make_engine(quant=False)
    quant = _make_engine(quant=True)
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 0 1",
        "4k3/4q3/8/8/8/8/3PPP2/4K3 w - - 0 1",
    ]
    for fen in fens:
        v_f, pri_f = exact.eval_debug(fen)
        v_q, pri_q = quant.eval_debug(fen)
        assert abs(v_f - v_q) < 0.03, f"quant value {fen}: {v_f} vs {v_q}"
        pri_q = dict(pri_q)
        for uci, p in pri_f:
            assert abs(p - pri_q[uci]) < 0.03, \
                f"quant prior {uci} {fen}: {p} vs {pri_q[uci]}"


#Quantization must not break search behavior: mate-in-1, legality (incl. in
#check), null move when mated (mirrors the f32 behavior tests).
def test_quant_search_behavior():
    if not AVAILABLE:
        return _skip()
    import math
    eng = _make_engine(quant=True)
    for fen, want in [("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
                      ("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", "e1e8")]:
        mv, _ = eng.search(fen, [], math.inf, 1500)
        assert mv == want, f"{fen}: got {mv}, want {want}"
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "4k3/4q3/8/8/8/8/3PPP2/4K3 w - - 0 1"]:
        board = cc.Board.from_fen(fen)
        mv, _ = eng.search(fen, [], math.inf, 200)
        legal = {common.to_uci(board, m) for m in board.generate_moves()}
        assert mv in legal, f"{fen}: {mv} not legal"
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert eng.search(mated, [], 1.0, None)[0] == "0000"


#Tree reuse must promote the played-move + reply subtree: after searching, the
#continuation search's cached root keeps its old visits (cold start would have
#exactly the visits of that one call), while an unrelated position must miss
#the cache and start cold. Legality throughout a reused game line too.
def test_tree_reuse():
    if not AVAILABLE:
        return _skip()
    import math
    eng = _make_engine(tree_reuse=True)
    board = cc.Board()
    m1, _ = eng.search(board.fen(), [], math.inf, 2000)
    board.play(next(m for m in board.generate_moves()
                    if common.to_uci(board, m) == m1))
    #A strong reply (from a stateless engine) — well-visited in eng's tree.
    reply, _ = _make_engine().search(board.fen(), [], math.inf, 400)
    board.play(next(m for m in board.generate_moves()
                    if common.to_uci(board, m) == reply))

    mv, sims = eng.search(board.fen(), [], math.inf, 1)
    legal = {common.to_uci(board, m) for m in board.generate_moves()}
    assert mv in legal, f"reused search returned illegal {mv}"
    assert sims == 1
    #The cache root is the child just played; with a promoted tree it keeps
    #visits inherited from the first search — one cold simulation could never
    #give it more than 1.
    _, cached_visits = eng.cache_stats()
    assert cached_visits > 1, \
        f"expected promoted subtree (cached visits > 1), got {cached_visits}"

    #Unrelated position: cache must miss and start cold — no node can exceed
    #that call's 50 simulations.
    eng.search("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1", [], math.inf, 50)
    _, cached_visits = eng.cache_stats()
    assert cached_visits <= 50, \
        f"expected cold start (<=50 visits), got {cached_visits}"

    #Reused line still finds a mate-in-1 when it appears.
    eng2 = _make_engine(tree_reuse=True)
    eng2.search("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", [], math.inf, 100)
    mv, _ = eng2.search("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", [], math.inf, 1500)
    assert mv == "a1a8", f"got {mv}, want a1a8"


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


#Behavioral parity with the Python MCTS (mirrors test_mcts.py) driven through
#the Rust search, using the random net (terminal values dominate net noise).
def test_mcts_finds_mate():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    import math
    for fen, want in [("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),
                      ("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", "e1e8")]:
        mv, _ = eng.search(fen, [], math.inf, 1500)
        assert mv == want, f"{fen}: got {mv}, want {want}"


def test_mcts_legal_and_null():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    import math
    for fen in ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "4k3/4q3/8/8/8/8/3PPP2/4K3 w - - 0 1"]:                 # in check
        board = cc.Board.from_fen(fen)
        mv, _ = eng.search(fen, [], math.inf, 200)
        legal = {common.to_uci(board, m) for m in board.generate_moves()}
        assert mv in legal, f"{fen}: {mv} not legal"
    mated = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    assert eng.search(mated, [], 1.0, None)[0] == "0000"


def test_mcts_time_budget():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    import time as _t
    fen = cc.Board().fen()
    start = _t.monotonic()
    mv, sims = eng.search(fen, [], 0.3, None)
    elapsed = _t.monotonic() - start
    assert elapsed < 1.0, f"ran {elapsed:.2f}s over a 0.3s budget"
    assert sims > 0


def test_mcts_repetition_scores_draw():
    if not AVAILABLE:
        return _skip()
    eng = _make_engine()
    board = cc.Board.from_fen("7k/8/8/8/8/8/R7/K7 w - - 0 1")
    target = next(m for m in board.generate_moves()
                  if common.to_uci(board, m) == "a2a3")
    child = board
    child = cc.Board.from_fen(board.fen())
    child.play(target)
    stats = dict((u, (v, vs)) for u, v, vs in
                 eng.root_stats(board.fen(), [child.hash()], 200))
    visits, value_sum = stats["a2a3"]
    assert visits > 0
    assert value_sum == 0.0, f"repetition child value_sum={value_sum}, expected 0"


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
