#Adapter to the Rust engine extension (salmon_nnue_rs), when it is built. Keeps
#the same bestMove contract as nnue/mcts.py so common.run_uci_loop can drive
#either backend. Imports guard so the engine stays pure-Python without the
#extension.

import math
import os

import numpy as np

import common

try:
    import salmon_nnue_rs
    AVAILABLE = True
except ImportError:
    AVAILABLE = False

SIMS_PER_DEPTH = 800            #matches nnue.mcts: "go depth N" -> N*800 sims
nodes = 0                       #simulations of the last search (bench parity)


#Build a RustEngine holding the net weights (loaded once per process).
#quant=True runs the L1/policy matmuls in int8 (faster, ~1/127-step
#approximate); the default follows $SALMON_NNUE_QUANT (on unless "0", so
#SALMON_NNUE_QUANT=0 forces exact-f32 inference for A/B runs). Pass
#quant=False explicitly where exact NumPy parity matters (tests).
#tree_reuse=True carries the search tree across moves within a game (promoted
#when the position continues the last search); default follows
#$SALMON_NNUE_TREE_REUSE the same way.
def make_engine(net, quant=None, tree_reuse=None):
    if quant is None:
        quant = os.environ.get("SALMON_NNUE_QUANT", "1") != "0"
    if tree_reuse is None:
        tree_reuse = os.environ.get("SALMON_NNUE_TREE_REUSE", "1") != "0"
    def b(arr):
        return np.ascontiguousarray(arr, dtype=np.float32).tobytes()
    return salmon_nnue_rs.RustEngine(
        b(net.ft_w), b(net.ft_b), b(net.l1_w), b(net.l1_b),
        b(net.v_w), float(net.v_b), b(net.p_w), b(net.p_b), quant, tree_reuse)


#Same call shape as nnue.mcts.bestMove; engine is a RustEngine from make_engine.
def bestMove(board, depth=0, time=common.default_movetime, history=None,
             engine=None, max_sims=None):
    global nodes
    if engine is None:
        raise ValueError("rust_engine.bestMove requires engine=")
    if max_sims is None and depth and time == math.inf:
        max_sims = depth * SIMS_PER_DEPTH
    hist = [int(h) for h in history] if history else []
    move, sims = engine.search(board.fen(), hist, float(time), max_sims)
    nodes = sims
    return move
