#Tests for the minimax (negamax + alpha-beta) search.
#Runnable with pytest, or directly: `python test_minimax.py`

import math
import time

import chess

import eval
import minimax


#Reference: quiescence with no alpha-beta pruning, mirroring minimax.quiescence.
def _ref_quiescence(board, ply=0):
    term = minimax._evaluate_terminal(board, ply)
    if term is not None:
        return term
    if board.is_check():
        best = -minimax.INF
        for move in board.legal_moves:
            board.push(move)
            best = max(best, -_ref_quiescence(board, ply + 1))
            board.pop()
        return best
    best = minimax._sign(board) * eval.eval_board(board)
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion is not None:
            board.push(move)
            best = max(best, -_ref_quiescence(board, ply + 1))
            board.pop()
    return best


#Reference: plain negamax with no negamax-level pruning. Its leaf calls the
#real quiescence with a full window, which returns the exact quiescence value;
#by the alpha-beta composition theorem the pruned negamax must match this. Using
#the (fast, pruned) quiescence here keeps the reference tractable, while
#quiescence's own pruning is verified separately against _ref_quiescence.
def _plain_negamax(board, depth, ply=0):
    term = minimax._evaluate_terminal(board, ply)
    if term is not None:
        return term
    if depth == 0:
        return minimax.quiescence(board, -minimax.INF, minimax.INF, ply)
    best = -minimax.INF
    for move in board.legal_moves:
        board.push(move)
        best = max(best, -_plain_negamax(board, depth - 1, ply + 1))
        board.pop()
    return best


#Guard 1: negamax alpha-beta pruning must not change the search value.
#Depth caps are per-position so the un-pruned reference stays fast in pure
#Python: sparse positions go to depth 3, the dense (many-move) one to depth 2.
def test_alpha_beta_equals_minimax():
    cases = [
        (chess.Board(), 3),                                                       # start
        (chess.Board("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"), 3),            # sparse endgame
        (chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 w - - 0 1"), 3),                    # live recapture
        (chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"), 2),
    ]
    for board, max_depth in cases:
        for depth in range(1, max_depth + 1):
            ab = minimax.negamax(board.copy(), depth, -minimax.INF, minimax.INF, 0)
            ref = _plain_negamax(board.copy(), depth)
            assert ab == ref, f"depth {depth} {board.fen()}: alpha-beta {ab} != minimax {ref}"


#Guard 2: quiescence's own alpha-beta pruning must not change its value.
def test_quiescence_equals_unpruned():
    positions = [
        chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 b - - 0 1"),               # recapture
        chess.Board("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1"),                # promotion/capture
        chess.Board("4k3/8/3n4/4P3/2B5/8/8/4K3 w - - 0 1"),            # minor-piece captures
    ]
    for board in positions:
        pruned = minimax.quiescence(board.copy(), -minimax.INF, minimax.INF, 0)
        ref = _ref_quiescence(board.copy(), 0)
        assert pruned == ref, f"{board.fen()}: pruned {pruned} != unpruned {ref}"


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


def test_quiescence_quiet_position_is_static():
    #With no captures available and not in check, quiescence == static eval.
    board = chess.Board()
    q = minimax.quiescence(board, -minimax.INF, minimax.INF, 0)
    assert q == minimax._sign(board) * eval.eval_board(board)


def test_quiescence_resolves_recapture():
    #White just grabbed a pawn with the queen (Qd5); Black can recapture
    #...exd5 winning the queen. The static eval says Black is far behind,
    #but quiescence sees the recapture and reports Black is actually winning.
    board = chess.Board("4k3/8/4p3/3Q4/8/8/8/4K3 b - - 0 1")
    naive = minimax._sign(board) * eval.eval_board(board)
    q = minimax.quiescence(board, -minimax.INF, minimax.INF, 0)
    assert naive < 0, "static eval should show Black behind on material"
    assert q > naive, "quiescence should improve Black's score via the recapture"
    assert q > 0, "after winning the queen Black is winning"


def test_quiescence_detects_mate_at_leaf():
    mate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert mate.is_checkmate()
    assert minimax.quiescence(mate, -minimax.INF, minimax.INF, 0) == -minimax.MATE


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
