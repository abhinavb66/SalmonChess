#Quick search-speed benchmark for the cozy-chess engine: nodes/sec at a fixed
#depth, plus nodes reached in a fixed time budget. Run: `python bench_speed.py`

import time

import cozy_chess as cc

import minimax

INF = minimax.INF


def reset():
    minimax.nodes = 0
    minimax._tt = {}
    minimax._history = {}
    for k in minimax._killers:
        k[0] = k[1] = None


def fixed_depth(board, depth):
    reset()
    t = time.monotonic()
    minimax.negamax(board, depth, -INF, INF, 0, delta=True, use_tt=True)
    return minimax.nodes, time.monotonic() - t


POSITIONS = [
    ("startpos", cc.Board()),
    ("italian", cc.Board.from_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")),
]

print("--- fixed depth 5 ---")
for label, board in POSITIONS:
    nodes, el = fixed_depth(board, 5)
    print(f"{label:10s} nodes={nodes:>8d}  time={el:6.2f}s  nps={nodes / el:>8.0f}")

#Practical metric: nodes (~depth) reached in a fixed 2s budget from the start.
b = cc.Board()
t = time.monotonic()
minimax.bestMove(b, depth=64, time=2.0)
el = time.monotonic() - t
print(f"--- 2s budget from startpos ---\nnodes={minimax.nodes}  ({minimax.nodes / el:.0f} nps)")
