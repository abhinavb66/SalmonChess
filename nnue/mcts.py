#Monte Carlo Tree Search (PUCT) for SalmonNNUE.
#
#The tree is guided by the network's policy head (move priors) and evaluated
#at the leaves by its value head — the Lc0 search shape with an NNUE-style
#net. Child boards and accumulators are materialized lazily on first visit
#(cozy boards advance by copy + play; accumulators thread incrementally from
#the parent). Draw-by-repetition uses the same 2-fold-on-the-search-path
#convention as minimax, seeded from the game history.

import copy
import math
import time as _time

import numpy as np
import cozy_chess as cc

import common
from nnue.network import Accumulator, policy_index

default_movetime = 1            #seconds per move when the GUI gives no limit

CPUCT = 1.5                     #exploration constant
FPU_REDUCTION = 0.3             #first-play-urgency: unvisited child Q penalty
SIMS_PER_DEPTH = 800            #"go depth N" maps to N*SIMS_PER_DEPTH simulations

#Move-prior source: "policy" (net head), "heuristic" (uniform + capture/
#promotion boost), or "blend" of both. Heuristic mode needs no retraining, so
#it is the fallback if the policy head underdelivers.
PRIOR_MODE = "blend"
BLEND_POLICY_WEIGHT = 0.75

nodes = 0                       #simulations run in the current search


class _Node:
    __slots__ = ("move", "prior", "visits", "value_sum", "children",
                 "board", "acc", "terminal")

    def __init__(self, move, prior):
        self.move = move            #move that leads here (None at the root)
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0        #from the PARENT's side-to-move perspective
        self.children = None        #None = not expanded
        self.board = None           #materialized lazily
        self.acc = None
        self.terminal = None        #cached common.terminal_result

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0


def _is_capture(board, move):
    if board.piece_on(move.to_square) is not None:
        return True
    return (board.piece_on(move.from_square) == cc.Piece.Pawn
            and (int(move.from_square) % 8) != (int(move.to_square) % 8))


#Move priors for the children of `board`. Returns a list aligned with moves.
def _priors(board, moves, logits):
    n = len(moves)
    if PRIOR_MODE in ("policy", "blend"):
        idx = np.fromiter((policy_index(board, m) for m in moves),
                          dtype=np.int64, count=n)
        masked = logits[idx]
        masked = masked - masked.max()
        policy = np.exp(masked)
        policy /= policy.sum()
    if PRIOR_MODE in ("heuristic", "blend"):
        heur = np.ones(n)
        for i, m in enumerate(moves):
            if _is_capture(board, m):
                heur[i] += 2.0
            if m.promotion is not None:
                heur[i] += 2.0
        heur /= heur.sum()
    if PRIOR_MODE == "policy":
        return policy
    if PRIOR_MODE == "heuristic":
        return heur
    return BLEND_POLICY_WEIGHT * policy + (1.0 - BLEND_POLICY_WEIGHT) * heur


#Expand a node: evaluate the net once, attach child stubs with priors, and
#return the leaf value from the node's side-to-move perspective in [-1,1].
def _expand(node, net):
    value, logits = net.value_policy(node.acc, node.board.side_to_move())
    moves = list(node.board.generate_moves())
    priors = _priors(node.board, moves, logits)
    node.children = [_Node(m, float(p)) for m, p in zip(moves, priors)]
    return 2.0 * value - 1.0


#One simulation from `node` (board/acc already materialized). Returns the
#value from node's side-to-move perspective in [-1,1]. `seen` holds the
#position hashes on the path from the root plus the game history.
def _simulate(node, net, seen):
    if node.terminal is not None:
        return -1.0 if node.terminal == "loss" else 0.0

    if node.children is None:
        node.terminal = common.terminal_result(node.board)
        if node.terminal is not None:
            return -1.0 if node.terminal == "loss" else 0.0
        return _expand(node, net)

    #Select the child maximizing PUCT from this node's perspective.
    sqrt_visits = math.sqrt(node.visits + 1)
    fpu = max(-1.0, node.q() - FPU_REDUCTION)
    best, best_score = None, -math.inf
    for child in node.children:
        q = child.q() if child.visits else fpu
        score = q + CPUCT * child.prior * sqrt_visits / (1 + child.visits)
        if score > best_score:
            best, best_score = child, score

    if best.board is None:
        best.board = copy.copy(node.board)
        best.board.play(best.move)
        best.acc = node.acc.play(net, node.board, best.move, best.board)

    h = best.board.hash()
    if h in seen:                       #repetition on the path -> draw
        v_child = 0.0
    else:
        seen.add(h)
        try:
            v_child = _simulate(best, net, seen)
        finally:
            seen.discard(h)

    v = -v_child                        #flip to this node's perspective
    best.visits += 1                    #the child gets its single visit here;
    best.value_sum += v                 #the node's own visit was counted by its
    return v                            #parent (or, for the root, in bestMove)


#Find the best move by MCTS within the time budget. Same call shape as
#minimax.bestMove so common.run_uci_loop can drive either engine. depth caps
#the simulation count (depth * SIMS_PER_DEPTH); time is the budget in seconds
#(math.inf for a pure sims-capped search); history is the list of position
#hashes preceding board, for repetition detection.
def bestMove(board, depth=0, time=default_movetime, history=None, net=None,
             max_sims=None):
    global nodes
    nodes = 0
    if net is None:
        raise ValueError("bestMove requires a net (nnue.network.Net)")
    moves = list(board.generate_moves())
    if not moves:
        return "0000"
    if len(moves) == 1:
        return common.to_uci(board, moves[0])

    if max_sims is None:
        #Sims cap only in pure fixed-depth mode; timed searches use the clock.
        max_sims = depth * SIMS_PER_DEPTH if (depth and time == math.inf) else math.inf
    deadline = None if time == math.inf else _time.monotonic() + max(0.0, time)

    root = _Node(None, 1.0)
    root.board = board
    root.acc = Accumulator.from_board(net, board)   #root always full-recomputes
    seen = set(history) if history else set()
    seen.add(board.hash())

    while nodes < max_sims:
        _simulate(root, net, seen)
        root.visits += 1                #the root's own visit (interior/leaf nodes
        nodes += 1                      #are counted by their parent in _simulate)
        if deadline is not None and _time.monotonic() >= deadline:
            break

    best = max(root.children, key=lambda c: (c.visits, c.q()))
    return common.to_uci(board, best.move)
