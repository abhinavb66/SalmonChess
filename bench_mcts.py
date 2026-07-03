#Quick MCTS speed benchmark for SalmonNNUE: simulations/sec at a fixed sims
#cap, plus simulations completed in a fixed time budget.
#Run: `python bench_mcts.py`. Uses the trained net if present, else random.

import math
import os
import time

import cozy_chess as cc

from nnue import mcts, network

_NET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "nets", "salmon_nnue.npz")
net = network.Net.load(_NET_PATH) if os.path.exists(_NET_PATH) \
    else network.make_random_net(0)
print("net:", _NET_PATH if os.path.exists(_NET_PATH) else "random (untrained)")

POSITIONS = [
    ("startpos", cc.Board()),
    ("italian", cc.Board.from_fen(
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")),
]

print("--- 2000 simulations ---")
for label, board in POSITIONS:
    t = time.monotonic()
    mcts.bestMove(board, time=math.inf, net=net, max_sims=2000)
    el = time.monotonic() - t
    print(f"{label:10s} time={el:6.2f}s  sims/s={2000 / el:>7.0f}")

print("--- 2s budget from startpos ---")
board = cc.Board()
t = time.monotonic()
mcts.bestMove(board, time=2.0, net=net)
el = time.monotonic() - t
print(f"sims={mcts.nodes}  ({mcts.nodes / el:.0f} sims/s)")
