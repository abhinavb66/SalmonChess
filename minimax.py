#Minimax (negamax + alpha-beta) search for the Salmon chess engine.
#
#Board representation is cozy-chess (a fast, native Rust move generator). Boards
#are advanced by copy + play (cozy boards are cheap value types with no unmake),
#and the transposition-table key is cozy's built-in incremental Zobrist hash.
#
#eval.eval_board returns an absolute score (+ favors White, - favors Black);
#the search flips it to the side-to-move (negamax) convention.

import copy
import math
import time as _time

import cozy_chess as cc

import eval

default_movetime = 1            #seconds per move when the GUI gives no limit

MATE = 1_000_000                #score for being checkmated at ply 0
MATE_THRESHOLD = MATE - 1000    #scores at least this large represent a forced mate
INF = MATE + 1

DELTA_MARGIN = 200              #centipawn safety margin for quiescence delta pruning

nodes = 0                       #positions visited in the current search

#Transposition table: zobrist key -> (depth, value, flag, best_move).
_EXACT, _LOWER, _UPPER = 0, 1, 2
_MAX_PLY = 256
_tt = {}
#Move-ordering heuristics, reset per search: two killer moves per ply and a
#(color, piece, to-square) -> cutoff-weight history table.
_killers = [[None, None] for _ in range(_MAX_PLY)]
_history = {}

_SQ_NAMES = ["abcdefgh"[i % 8] + str(i // 8 + 1) for i in range(64)]
_PROMO_CHAR = {cc.Piece.Knight: "n", cc.Piece.Bishop: "b",
               cc.Piece.Rook: "r", cc.Piece.Queen: "q"}


class _Timeout(Exception):
    """Raised internally to unwind the search when the time budget is spent."""


def _sign(board):
    return 1 if board.side_to_move() == cc.Color.White else -1


#cozy encodes castling as the king moving onto its own rook (e1h1/e1a1). Convert
#to/from standard UCI (e1g1/e1c1) at the engine boundary only.
def to_uci(board, move):
    fs, ts = int(move.from_square), int(move.to_square)
    if (board.piece_on(move.from_square) == cc.Piece.King
            and board.piece_on(move.to_square) == cc.Piece.Rook
            and board.color_on(move.to_square) == board.color_on(move.from_square)):
        rank = fs - (fs % 8)
        ts = rank + (6 if (ts % 8) == 7 else 2)     #king lands on g- or c-file
    s = _SQ_NAMES[fs] + _SQ_NAMES[ts]
    if move.promotion is not None:
        s += _PROMO_CHAR[move.promotion]
    return s


def from_uci(board, uci):
    for m in board.generate_moves():
        if to_uci(board, m) == uci:
            return m
    raise ValueError(f"illegal or unknown move: {uci}")


#A move is a capture if the destination is occupied, or it is an en-passant
#pawn push (a pawn changing file onto an empty square).
def _is_capture(board, move):
    if board.piece_on(move.to_square) is not None:
        return True
    return (board.piece_on(move.from_square) == cc.Piece.Pawn
            and (int(move.from_square) % 8) != (int(move.to_square) % 8))


def _victim_value(board, move):
    victim = board.piece_on(move.to_square)
    return eval.pieceVals[victim] if victim is not None else eval.pieceVals[cc.Piece.Pawn]


#Most-valuable-victim / least-valuable-attacker score for a noisy move.
def _mvv_lva(board, move):
    victim = _victim_value(board, move) if _is_capture(board, move) else 0
    attacker = eval.pieceVals[board.piece_on(move.from_square)]
    promo = (eval.pieceVals[move.promotion] - eval.pieceVals[cc.Piece.Pawn]
             if move.promotion is not None else 0)
    return 10 * victim - attacker + promo


#cozy's status() does not flag insufficient material, so detect the basic draws.
def _insufficient_material(board):
    if (len(board.pieces(cc.Piece.Pawn)) or len(board.pieces(cc.Piece.Rook))
            or len(board.pieces(cc.Piece.Queen))):
        return False
    minors = len(board.pieces(cc.Piece.Knight)) + len(board.pieces(cc.Piece.Bishop))
    return minors <= 1


#Terminal score relative to the side to move, or None if not terminal.
def _evaluate_terminal(board, ply):
    st = board.status()
    if st == cc.GameStatus.Won:         #side to move is checkmated
        return -(MATE - ply)
    if st == cc.GameStatus.Drawn:       #stalemate or fifty-move
        return 0
    if _insufficient_material(board):
        return 0
    return None


#Order moves best-first: hint move (TT/PV), then captures/promotions by MVV-LVA,
#then killer moves, then quiet moves by history score.
def _ordered_moves(board, hint_move=None, ply=None):
    killers = _killers[ply] if (ply is not None and ply < _MAX_PLY) else ()
    stm = board.side_to_move()

    def score(m):
        if hint_move is not None and m == hint_move:
            return 3_000_000
        if _is_capture(board, m) or m.promotion is not None:
            return 2_000_000 + _mvv_lva(board, m)
        if m in killers:
            return 1_000_000
        return min(_history.get((stm, board.piece_on(m.from_square), int(m.to_square)), 0),
                   999_999)

    return sorted(board.generate_moves(), key=score, reverse=True)


def _ordered_captures(board):
    moves = [m for m in board.generate_moves()
             if _is_capture(board, m) or m.promotion is not None]
    moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    return moves


def _record_killer(move, ply):
    if ply < _MAX_PLY and _killers[ply][0] != move:
        _killers[ply][1] = _killers[ply][0]
        _killers[ply][0] = move


def _record_history(board, move, depth):
    key = (board.side_to_move(), board.piece_on(move.from_square), int(move.to_square))
    _history[key] = _history.get(key, 0) + depth * depth


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


#Quiescence search: resolve captures/promotions until the position is quiet
#before evaluating, avoiding the horizon effect. Fail-soft, like negamax.
def quiescence(board, alpha, beta, ply=0, deadline=None, delta=True):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1

    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term

    #In check there is no safe "do nothing" option, so search every evasion.
    if len(board.checkers()) > 0:
        best = -INF
        for move in _ordered_moves(board):
            child = copy.copy(board)
            child.play(move)
            score = -quiescence(child, -beta, -alpha, ply + 1, deadline, delta)
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
            if alpha >= beta:
                break
        return best

    stand_pat = _sign(board) * eval.eval_board(board)
    best = stand_pat
    if best > alpha:
        alpha = best
    if alpha >= beta:
        return best

    for move in _ordered_captures(board):
        #Delta pruning: skip captures too small to raise alpha (never promotions).
        if delta and move.promotion is None:
            if stand_pat + _victim_value(board, move) + DELTA_MARGIN <= alpha:
                continue
        child = copy.copy(board)
        child.play(move)
        score = -quiescence(child, -beta, -alpha, ply + 1, deadline, delta)
        if score > best:
            best = score
            if best > alpha:
                alpha = best
        if alpha >= beta:
            break
    return best


#Fail-soft alpha-beta negamax with a transposition table. With a full window at
#the root and delta=False, the returned value equals the true minimax value.
#use_tt=False disables the table; seen holds position hashes on the path from the
#root (plus game history) for draw-by-repetition detection.
def negamax(board, depth, alpha, beta, ply=0, deadline=None, delta=True, use_tt=True, seen=None):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1
    if seen is None:
        seen = set()

    h = board.hash()
    if h in seen:                       #repetition on the search path -> draw
        return 0
    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term
    if depth == 0:
        return quiescence(board, alpha, beta, ply, deadline, delta)

    alpha_orig = alpha
    tt_move = None
    if use_tt:
        entry = _tt.get(h)
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

    best = -INF
    best_move = None
    seen.add(h)
    try:
        for move in _ordered_moves(board, tt_move, ply):
            child = copy.copy(board)
            child.play(move)
            score = -negamax(child, depth - 1, -beta, -alpha, ply + 1, deadline,
                             delta, use_tt, seen)
            if score > best:
                best = score
                best_move = move
                if best > alpha:
                    alpha = best
            if alpha >= beta:
                if not _is_capture(board, move) and move.promotion is None:
                    _record_killer(move, ply)
                    _record_history(board, move, depth)
                break
    finally:
        seen.discard(h)

    if use_tt:
        if best <= alpha_orig:
            flag = _UPPER
        elif best >= beta:
            flag = _LOWER
        else:
            flag = _EXACT
        prev = _tt.get(h)
        if prev is None or prev[0] <= depth:       #depth-preferred replacement
            _tt[h] = (depth, _tt_store_score(best, ply), flag, best_move)
    return best


#Search every root move at a fixed depth, returning (best_move, best_score).
def search_root(board, depth, deadline=None, pv_move=None, seen=None):
    if seen is None:
        seen = set()
    alpha, beta = -INF, INF
    moves = _ordered_moves(board, pv_move, 0)
    best_move = moves[0]
    best_score = -INF
    h = board.hash()
    seen.add(h)
    try:
        for move in moves:
            child = copy.copy(board)
            child.play(move)
            score = -negamax(child, depth - 1, -beta, -alpha, 1, deadline,
                             True, True, seen)
            if score > best_score:
                best_score = score
                best_move = move
                if best_score > alpha:
                    alpha = best_score
    finally:
        seen.discard(h)
    return best_move, best_score


#Find the best move via iterative deepening within the time budget.
#depth caps the maximum search depth; time is the budget in seconds (pass
#math.inf for a pure fixed-depth search). history is the list of position
#hashes preceding the current one, for draw-by-repetition detection.
def bestMove(board, depth=64, time=default_movetime, history=None):
    global nodes, _tt, _history
    nodes = 0
    _tt = {}                    #table persists across deepening iterations, not searches
    _history = {}
    for k in _killers:
        k[0] = k[1] = None
    legal = list(board.generate_moves())
    if not legal:
        return "0000"

    base_seen = set(history) if history else set()
    deadline = None if time == math.inf else _time.monotonic() + max(0.0, time)
    best_move = legal[0]
    pv_move = None

    for d in range(1, depth + 1):
        try:
            move, score = search_root(board, d, deadline, pv_move, set(base_seen))
        except _Timeout:
            break                       #discard the incomplete depth, keep last best
        best_move = move
        pv_move = move
        if abs(score) >= MATE_THRESHOLD:
            break                       #forced mate found, no need to search deeper
        if deadline is not None and _time.monotonic() >= deadline:
            break

    return to_uci(board, best_move)
