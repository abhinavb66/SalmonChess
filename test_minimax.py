#Tests for the minimax (negamax + alpha-beta) search.
#Runnable with pytest, or directly: `python test_minimax.py`

import math
import time

import chess

import eval
import minimax


#Reference: plain negamax with no pruning, sharing the same terminal/leaf
#logic as the real search. Alpha-beta must return the identical value.
def _plain_negamax(board, depth, ply=0):
    term = minimax._evaluate_terminal(board, ply)
    if term is not None:
        return term
    if depth == 0:
        return minimax._sign(board) * eval.eval_board(board)
    best = -minimax.INF
    for move in board.legal_moves:
        board.push(move)
        best = max(best, -_plain_negamax(board, depth - 1, ply + 1))
        board.pop()
    return best


#The core correctness guard: pruning must not change the search value.
def test_alpha_beta_equals_minimax():
    positions = [
        chess.Board(),                                                   # start
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
        chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),        # endgame
        chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ]
    for board in positions:
        for depth in (1, 2, 3):
            ab = minimax.negamax(board.copy(), depth, -minimax.INF, minimax.INF, 0)
            ref = _plain_negamax(board.copy(), depth)
            assert ab == ref, f"depth {depth} {board.fen()}: alpha-beta {ab} != minimax {ref}"


def test_finds_mate_in_one():
    #Back-rank mate: White Ra1-a8 is checkmate.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert minimax.bestMove(board, depth=3, time=math.inf) == "a1a8"
    #The mate is delivered at ply 1, so the root score must be MATE - 1.
    assert minimax.negamax(board.copy(), 2, -minimax.INF, minimax.INF, 0) == minimax.MATE - 1


def test_mate_score_is_ply_adjusted():
    #Terminal scores are from the mated side's view: being mated sooner is worse.
    #After negamax negation the mating side therefore prefers the faster mate,
    #which is what drives the engine toward mate instead of dithering.
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mate.is_checkmate()
    assert minimax._evaluate_terminal(mate, 1) == -(minimax.MATE - 1)
    assert minimax._evaluate_terminal(mate, 3) == -(minimax.MATE - 3)
    #Mated side: sooner is worse (more negative)...
    assert minimax._evaluate_terminal(mate, 1) < minimax._evaluate_terminal(mate, 3)
    #...so after negation the mating side scores the faster mate higher.
    assert -minimax._evaluate_terminal(mate, 1) > -minimax._evaluate_terminal(mate, 3)


def test_wins_free_material():
    #White queen can capture an undefended black queen for free; it must.
    board = chess.Board("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1")
    move = minimax.bestMove(board, depth=3, time=math.inf)
    assert move == "d1d5", f"expected to capture the queen, played {move}"


def test_returns_legal_move():
    board = chess.Board()
    move = minimax.bestMove(board, depth=3, time=math.inf)
    assert chess.Move.from_uci(move) in board.legal_moves


def test_no_legal_moves_returns_null():
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mate.is_checkmate()
    assert minimax.bestMove(mate) == "0000"


def test_does_not_mutate_board():
    board = chess.Board()
    before = board.fen()
    minimax.bestMove(board, depth=3, time=math.inf)
    assert board.fen() == before


def test_does_not_mutate_board_on_timeout():
    #A timed-out search must still leave the board fully unwound, since
    #Salmon.py reuses the same board across go commands.
    board = chess.Board("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
    before = board.fen()
    minimax.bestMove(board, depth=64, time=0.2)   # forces a mid-search timeout
    assert board.fen() == before


def test_respects_time_budget():
    board = chess.Board()
    start = time.monotonic()
    move = minimax.bestMove(board, depth=64, time=0.3)   # deep cap, short budget
    elapsed = time.monotonic() - start
    assert elapsed < 2.0, f"search ran {elapsed:.2f}s, far over its 0.3s budget"
    assert chess.Move.from_uci(move) in board.legal_moves


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
