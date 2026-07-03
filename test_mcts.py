#Tests for the MCTS search. Runnable with pytest, or directly:
#`python test_mcts.py`. Uses a fixed-seed random net: tactical assertions rely
#on terminal values (+1/-1/0) dominating whatever a random net outputs.

import math
import time as _time

import cozy_chess as cc

import common
from nnue import mcts, network

NET = network.make_random_net(0)


def test_mate_in_one():
    #Terminal wins dominate any net eval, so modest sim counts find these.
    cases = [
        ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "a1a8"),     #back-rank mate
        ("6k1/8/6K1/8/8/8/8/4R3 w - - 0 1", "e1e8"),       #supported rook mate
    ]
    for fen, expected in cases:
        board = cc.Board.from_fen(fen)
        move = mcts.bestMove(board, time=math.inf, net=NET, max_sims=1500)
        assert move == expected, f"{fen}: got {move}, want {expected}"


def test_returns_legal_move():
    fens = [
        None,                                                       #startpos
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/8/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 2",
        "4k3/4q3/8/8/8/8/3PPP2/4K3 w - - 0 1",                      #in check
    ]
    for fen in fens:
        board = cc.Board() if fen is None else cc.Board.from_fen(fen)
        move = mcts.bestMove(board, time=math.inf, net=NET, max_sims=200)
        legal = {common.to_uci(board, m) for m in board.generate_moves()}
        assert move in legal, f"{fen}: {move} not legal"


def test_no_legal_moves_returns_null():
    mate = cc.Board.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert not list(mate.generate_moves())
    assert mcts.bestMove(mate, net=NET) == "0000"


def test_does_not_mutate_board():
    board = cc.Board()
    before = board.fen()
    mcts.bestMove(board, time=math.inf, net=NET, max_sims=300)
    assert board.fen() == before


def test_time_budget():
    board = cc.Board()
    start = _time.monotonic()
    move = mcts.bestMove(board, time=0.3, net=NET)
    elapsed = _time.monotonic() - start
    assert elapsed < 1.5, f"search ran {elapsed:.2f}s, far over its 0.3s budget"
    legal = {common.to_uci(board, m) for m in board.generate_moves()}
    assert move in legal


def test_repetition_scores_draw():
    #A child whose position hash is already in the seen set (game history or
    #search path) must back up exactly 0 (a draw), not the net's opinion.
    import copy
    board = cc.Board.from_fen("7k/8/8/8/8/8/R7/K7 w - - 0 1")
    #Mark the position after a2a3 as already seen once.
    target = next(m for m in board.generate_moves()
                  if common.to_uci(board, m) == "a2a3")
    child_board = copy.copy(board)
    child_board.play(target)
    seen = {board.hash(), child_board.hash()}

    root = mcts._Node(None, 1.0)
    root.board = board
    root.acc = network.Accumulator.from_board(NET, board)
    for _ in range(200):
        mcts._simulate(root, NET, seen)

    repeat_child = next(c for c in root.children
                        if common.to_uci(board, c.move) == "a2a3")
    assert repeat_child.visits > 0
    assert repeat_child.value_sum == 0.0, \
        f"repetition child q={repeat_child.q()}, expected exact 0"
    #And the search still returns a legal move with that history in play.
    move = mcts.bestMove(board, time=math.inf, net=NET, max_sims=300,
                         history=[child_board.hash()])
    legal = {common.to_uci(board, m) for m in board.generate_moves()}
    assert move in legal


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
