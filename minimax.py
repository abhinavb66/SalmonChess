#Minimax (negamax + alpha-beta) search for the Salmon chess engine.
#
#eval.eval_board returns an absolute score (+ favors White, - favors Black).
#Search uses the negamax convention, so the leaf score is flipped to be
#relative to the side to move. Iterative deepening drives the search within
#a time budget, keeping the best move from the last fully completed depth.

import math
import time as _time

import chess

import eval

default_movetime = 1            #seconds per move when the GUI gives no limit

MATE = 1_000_000                #score for being checkmated at ply 0
MATE_THRESHOLD = MATE - 1000    #scores at least this large represent a forced mate
INF = MATE + 1

DELTA_MARGIN = 200              #centipawn safety margin for quiescence delta pruning

nodes = 0                       #positions visited in the current search


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
#eval_move is a white-relative delta, so scale by the side's sign.
def _ordered_moves(board, pv_move=None):
    sign = _sign(board)
    moves = list(board.legal_moves)
    moves.sort(key=lambda m: sign * eval.eval_move(board, m), reverse=True)
    if pv_move is not None and pv_move in moves:
        moves.remove(pv_move)
        moves.insert(0, pv_move)
    return moves


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


#Fail-soft alpha-beta negamax. With a full window at the root, the returned
#value equals the true minimax value of the position (when delta=False; delta
#pruning in the quiescence leaf is heuristic and may change the value slightly).
def negamax(board, depth, alpha, beta, ply=0, deadline=None, delta=True):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1

    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term
    if depth == 0:
        return quiescence(board, alpha, beta, ply, deadline, delta)

    best = -INF
    for move in _ordered_moves(board):
        board.push(move)
        try:
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline, delta)
        finally:
            board.pop()         #always restore, even when _Timeout unwinds
        if score > best:
            best = score
            if best > alpha:
                alpha = best
        if alpha >= beta:
            break
    return best


#Search every root move at a fixed depth, returning (best_move, best_score).
#pv_move (the previous iteration's best) is tried first.
def search_root(board, depth, deadline=None, pv_move=None):
    alpha, beta = -INF, INF
    moves = _ordered_moves(board, pv_move)
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
    global nodes
    nodes = 0
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
