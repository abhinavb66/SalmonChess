#Minimax (negamax + alpha-beta) search for the Salmon chess engine.
#
#eval.eval_board returns an absolute score (+ favors White, - favors Black).
#Search uses the negamax convention, so the leaf score is flipped to be
#relative to the side to move. Iterative deepening drives the search within
#a time budget, keeping the best move from the last fully completed depth.

import math
import time as _time

import chess
import chess.polyglot

import eval

default_movetime = 1            #seconds per move when the GUI gives no limit

MATE = 1_000_000                #score for being checkmated at ply 0
MATE_THRESHOLD = MATE - 1000    #scores at least this large represent a forced mate
INF = MATE + 1

DELTA_MARGIN = 200              #centipawn safety margin for quiescence delta pruning

nodes = 0                       #positions visited in the current search

#Transposition table: zobrist key -> (depth, value, flag, best_move).
#Values are stored relative to the node (mate scores ply-adjusted) so entries
#are independent of where in the tree the position was reached.
_EXACT, _LOWER, _UPPER = 0, 1, 2
_MAX_PLY = 256
_tt = {}
#Move-ordering heuristics, reset per search: two killer moves per ply and a
#(color, piece, to-square) -> cutoff-weight history table.
_killers = [[None, None] for _ in range(_MAX_PLY)]
_history = {}


class _Timeout(Exception):
    """Raised internally to unwind the search when the time budget is spent."""


def _sign(board):
    return 1 if board.turn == chess.WHITE else -1


#Return a terminal score relative to the side to move, or None if the
#position is not terminal. Shared by negamax and the test reference search.
def _evaluate_terminal(board, ply):
    if board.is_checkmate():
        return -(MATE - ply)        #prefer faster mates / longest defense
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_fifty_moves()
            or (ply > 0 and board.is_repetition(2))):
        return 0
    return None


#Order moves best-first for the side to move to improve alpha-beta pruning.
#Tiers (high to low): the hint move (TT/PV) first, then captures/promotions by
#eval_move delta, then killer moves, then quiet moves by history score.
def _ordered_moves(board, hint_move=None, ply=None):
    sign = _sign(board)
    killers = _killers[ply] if (ply is not None and ply < _MAX_PLY) else ()

    def score(m):
        if hint_move is not None and m == hint_move:
            return 3_000_000
        if board.is_capture(m) or m.promotion is not None:
            return 2_000_000 + sign * eval.eval_move(board, m)
        if m in killers:
            return 1_000_000
        piece = board.piece_at(m.from_square)
        return min(_history.get((piece.color, piece.piece_type, m.to_square), 0), 999_999)

    return sorted(board.legal_moves, key=score, reverse=True)


#Record a quiet move that caused a beta cutoff: keep two killers per ply and
#weight history by depth squared (deeper cutoffs are stronger evidence).
def _record_killer(move, ply):
    if ply < _MAX_PLY and _killers[ply][0] != move:
        _killers[ply][1] = _killers[ply][0]
        _killers[ply][0] = move


def _record_history(board, move, depth):
    piece = board.piece_at(move.from_square)
    if piece is not None:
        key = (piece.color, piece.piece_type, move.to_square)
        _history[key] = _history.get(key, 0) + depth * depth


#Mate scores encode distance from the root; convert to/from node-relative so a
#transposition-table entry is valid wherever the position recurs.
def _tt_store_score(score, ply):
    if score >= MATE_THRESHOLD:
        return score + ply
    if score <= -MATE_THRESHOLD:
        return score - ply
    return score


def _tt_load_score(score, ply):
    if score >= MATE_THRESHOLD:
        return score - ply
    if score <= -MATE_THRESHOLD:
        return score + ply
    return score


#Noisy moves (captures and promotions) ordered best-first for the side to move.
#These are the only moves quiescence explores in a quiet (not-in-check) position.
def _ordered_captures(board):
    sign = _sign(board)
    moves = [m for m in board.legal_moves
             if board.is_capture(m) or m.promotion is not None]
    moves.sort(key=lambda m: sign * eval.eval_move(board, m), reverse=True)
    return moves


#The material gained by a capture, used by delta pruning. En passant captures a
#pawn that is not on the destination square, so handle it explicitly.
def _captured_value(board, move):
    if board.is_en_passant(move):
        return eval.pieceVals[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return eval.pieceVals[victim.piece_type] if victim is not None else 0


#Quiescence search: at a search leaf, keep resolving captures/promotions until
#the position is "quiet" before evaluating, so the static eval is never read in
#the middle of a capture sequence (the horizon effect). Fail-soft, like negamax.
#delta enables delta pruning (skipping captures too small to raise alpha); pass
#delta=False for an exact, value-preserving search.
def quiescence(board, alpha, beta, ply=0, deadline=None, delta=True):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1

    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term

    #In check there is no safe "do nothing" option, so search every evasion
    #(this also lets a checkmate at the leaf be found and scored).
    if board.is_check():
        best = -INF
        for move in _ordered_moves(board):
            board.push(move)
            try:
                score = -quiescence(board, -beta, -alpha, ply + 1, deadline, delta)
            finally:
                board.pop()
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
            if alpha >= beta:
                break
        return best

    #Stand-pat: the side to move is never forced to capture, so the static
    #evaluation is a lower bound on what it can achieve.
    stand_pat = _sign(board) * eval.eval_board(board)
    best = stand_pat
    if best > alpha:
        alpha = best
    if alpha >= beta:
        return best

    for move in _ordered_captures(board):
        #Delta pruning: if even winning the captured piece (plus a safety
        #margin) cannot raise alpha, this capture is hopeless, so skip it.
        #Promotions can swing far more than the captured piece, so never prune
        #them this way.
        if delta and move.promotion is None:
            if stand_pat + _captured_value(board, move) + DELTA_MARGIN <= alpha:
                continue
        board.push(move)
        try:
            score = -quiescence(board, -beta, -alpha, ply + 1, deadline, delta)
        finally:
            board.pop()
        if score > best:
            best = score
            if best > alpha:
                alpha = best
        if alpha >= beta:
            break
    return best


#Fail-soft alpha-beta negamax with a transposition table. With a full window at
#the root, the returned value equals the true minimax value of the position
#(when delta=False; delta pruning in the quiescence leaf is heuristic and may
#change the value slightly). use_tt=False disables the table for exact, history-
#independent searches (used by the correctness tests).
def negamax(board, depth, alpha, beta, ply=0, deadline=None, delta=True, use_tt=True):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1

    alpha_orig = alpha
    key = None
    tt_move = None
    if use_tt:
        key = chess.polyglot.zobrist_hash(board)
        entry = _tt.get(key)
        if entry is not None:
            e_depth, e_value, e_flag, tt_move = entry
            if e_depth >= depth:
                val = _tt_load_score(e_value, ply)
                if e_flag == _EXACT:
                    return val
                if e_flag == _LOWER and val >= beta:
                    return val
                if e_flag == _UPPER and val <= alpha:
                    return val

    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term
    if depth == 0:
        return quiescence(board, alpha, beta, ply, deadline, delta)

    best = -INF
    best_move = None
    for move in _ordered_moves(board, tt_move, ply):
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline, delta, use_tt)
        finally:
            board.pop()         #always restore, even when _Timeout unwinds
        if score > best:
            best = score
            best_move = move
            if best > alpha:
                alpha = best
        if alpha >= beta:
            if not board.is_capture(move) and move.promotion is None:
                _record_killer(move, ply)
                _record_history(board, move, depth)
            break

    if use_tt:
        if best <= alpha_orig:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        prev = _tt.get(key)
        if prev is None or prev[0] <= depth:       #depth-preferred replacement
            _tt[key] = (depth, _tt_store_score(best, ply), flag, best_move)
    return best


#Search every root move at a fixed depth, returning (best_move, best_score).
#pv_move (the previous iteration's best) is tried first.
def search_root(board, depth, deadline=None, pv_move=None):
    alpha, beta = -INF, INF
    moves = _ordered_moves(board, pv_move, 0)
    best_move = moves[0]
    best_score = -INF
    for move in moves:
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, 1, deadline)
        finally:
            board.pop()         #always restore, even when _Timeout unwinds
        if score > best_score:
            best_score = score
            best_move = move
            if best_score > alpha:
                alpha = best_score
    return best_move, best_score


#Find the best move via iterative deepening within the time budget.
#depth caps the maximum search depth; time is the budget in seconds
#(pass math.inf for a pure fixed-depth search).
def bestMove(board, depth=64, time=default_movetime):
    global nodes, _tt, _history
    nodes = 0
    _tt = {}                    #table persists across deepening iterations, not searches
    _history = {}
    for k in _killers:
        k[0] = k[1] = None
    legal = list(board.legal_moves)
    if not legal:
        return "0000"

    deadline = None if time == math.inf else _time.monotonic() + max(0.0, time)
    best_move = legal[0]
    pv_move = None

    for d in range(1, depth + 1):
        try:
            move, score = search_root(board, d, deadline, pv_move)
        except _Timeout:
            break                       #discard the incomplete depth, keep last best
        best_move = move
        pv_move = move
        if abs(score) >= MATE_THRESHOLD:
            break                       #forced mate found, no need to search deeper
        if deadline is not None and _time.monotonic() >= deadline:
            break

    return best_move.uci()
