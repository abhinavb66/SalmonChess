#Tests for the NNUE network module (feature encoding, accumulator, inference).
#Runnable with pytest, or directly: `python test_nnue.py`
#All tests use a fixed-seed random net — no trained artifact required.

import copy
import os
import random
import tempfile

import numpy as np
import cozy_chess as cc

from nnue import network

NET = network.make_random_net(0)


#Core invariant: the incremental accumulator must equal a full recompute after
#every move — across random games (captures, promotions, castling, en passant)
#and targeted special-move positions.
def test_incremental_matches_full():
    random.seed(3)
    for _ in range(20):
        board = cc.Board()
        acc = network.Accumulator.from_board(NET, board)
        for _ in range(40):
            moves = list(board.generate_moves())
            if not moves:
                break
            move = random.choice(moves)
            child = copy.copy(board)
            child.play(move)
            acc = acc.play(NET, board, move, child)
            full = network.Accumulator.from_board(NET, child)
            assert np.allclose(acc.vals, full.vals, atol=1e-4), \
                f"accumulator desync after {move} at {board.fen()}"
            board = child


def test_incremental_special_moves():
    cases = [
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1h1"),      #white O-O (cozy encoding)
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1a1"),      #white O-O-O
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8h8"),      #black O-O
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),           #promotion
        ("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q"),         #promotion capture
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),         #en passant
    ]
    for fen, uci in cases:
        board = cc.Board.from_fen(fen)
        move = next(m for m in board.generate_moves()
                    if f"{m}" == uci or str(m) == uci)
        acc = network.Accumulator.from_board(NET, board)
        child = copy.copy(board)
        child.play(move)
        inc = acc.play(NET, board, move, child)
        full = network.Accumulator.from_board(NET, child)
        assert np.allclose(inc.vals, full.vals, atol=1e-4), f"{uci} at {fen}"


def test_feature_symmetry():
    #Startpos: 32 active features per perspective, and the two perspectives of
    #a symmetric position activate the same feature set.
    board = cc.Board()
    w = sorted(network.active_features(board, cc.Color.White))
    b = sorted(network.active_features(board, cc.Color.Black))
    assert len(w) == len(b) == 32
    assert w == b       #startpos is vertically symmetric


def test_npz_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "net.npz")
        NET.save(path)
        loaded = network.Net.load(path)
        #float16 on disk: round-trip within half precision
        assert np.allclose(NET.ft_w, loaded.ft_w, atol=1e-3)
        assert np.allclose(NET.p_w, loaded.p_w, atol=1e-3)
        board = cc.Board()
        acc = network.Accumulator.from_board(loaded, board)
        v, logits = loaded.value_policy(acc, board.side_to_move())
        assert 0.0 < v < 1.0
        assert logits.shape == (network.POLICY_SIZE,)


def test_policy_index():
    #Every legal move maps into 0..4095; castling maps to the g/c-file cell;
    #the Black-stm mirror of a White-stm position maps to the same indices.
    board = cc.Board.from_fen("r3k2r/8/8/3pP3/8/8/P7/R3K2R w KQkq d6 0 1")
    idxs = [network.policy_index(board, m) for m in board.generate_moves()]
    assert all(0 <= i < 4096 for i in idxs)
    castle = next(m for m in board.generate_moves() if str(m) == "e1h1")
    e1, g1 = 4, 6
    assert network.policy_index(board, castle) == e1 * 64 + g1

    mirrored = cc.Board.from_fen("r3k2r/p7/8/8/3Pp3/8/8/R3K2R b KQkq d3 0 1")
    a = sorted(network.policy_index(board, m) for m in board.generate_moves())
    b = sorted(network.policy_index(mirrored, m) for m in mirrored.generate_moves())
    assert a == b


def test_numpy_matches_torch():
    #Parity between the NumPy inference and the torch training module. Skips
    #cleanly when torch is not installed (it is a training-only dependency).
    try:
        import torch
    except ImportError:
        print("  (torch not installed - skipped)")
        return
    from nnue.training.train import TorchNet

    model = TorchNet.from_net(NET)
    fens = [
        None,
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    ]
    for fen in fens:
        board = cc.Board() if fen is None else cc.Board.from_fen(fen)
        acc = network.Accumulator.from_board(NET, board)
        v_np, logits_np = NET.value_policy(acc, board.side_to_move())
        with torch.no_grad():
            feats = [network.active_features(board, p)
                     for p in (board.side_to_move(),
                               cc.Color.Black if board.side_to_move() == cc.Color.White
                               else cc.Color.White)]
            v_t, logits_t = model.forward_features(feats[0], feats[1])
        assert abs(v_np - float(v_t)) < 1e-4, fen
        assert np.allclose(logits_np, logits_t.numpy(), atol=1e-3), fen


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
