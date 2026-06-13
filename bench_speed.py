import math
import time

import chess

import minimax

INF = minimax.INF


def reset():
    minimax.nodes = 0
    minimax._tt = {}
    minimax._history = {}
    for k in minimax._killers:
        k[0] = k[1] = None


def fixed_depth(fen, depth):
    b = chess.Board(fen)
    reset()
    t = time.monotonic()
    minimax.negamax(b, depth, -INF, INF, 0, delta=True, use_tt=True)
    el = time.monotonic() - t
    return minimax.nodes, el


POSITIONS = [
    ("startpos", chess.STARTING_FEN),
    ("italian", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
]

#Warm up the JIT (no-op cost on CPython) before timing.
for _ in range(2):
    fixed_depth(chess.STARTING_FEN, 4)

print(f"interpreter: {'PyPy' if hasattr(__import__('sys'), 'pypy_version_info') else 'CPython'}")
print("--- fixed depth 5 (identical work) ---")
for label, fen in POSITIONS:
    nodes, el = fixed_depth(fen, 5)
    print(f"{label:10s} nodes={nodes:>8d}  time={el:6.2f}s  nps={nodes/el:>8.0f}")

#Practical metric: how many nodes (≈ how deep) in a fixed 2s budget from the start.
b = chess.Board()
t = time.monotonic()
minimax.bestMove(b, depth=64, time=2.0)
el = time.monotonic() - t
print(f"--- 2s budget from startpos ---\nnodes={minimax.nodes}  ({minimax.nodes/el:.0f} nps)")
