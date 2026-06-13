#Minimax (negamax + alpha-beta) search for the Salmon chess engine.
#
#eval.eval_board returns an absolute score (+ favors White, - favors Black).
#Search uses the negamax convention, so the leaf score is flipped to be
#relative to the side to move. Iterative deepening drives the search within
#a time budget, keeping the best move from the last fully completed depth.
#
#For speed the search threads three running values through make/unmake instead
#of recomputing them at every node: the evaluation (ev, via eval_move), the
#non-pawn material (mat, for the game phase), and the piece-placement Zobrist
#key (pkey, for the transposition table). See _child_eval_mat / _piece_key_delta.

import math
import random as _random
import time as _time

import chess

import eval

default_movetime = 1            #seconds per move when the GUI gives no limit

MATE = 1_000_000                #score for being checkmated at ply 0
MATE_THRESHOLD = MATE - 1000    #scores at least this large represent a forced mate
INF = MATE + 1

DELTA_MARGIN = 200              #centipawn safety margin for quiescence delta pruning
ENDGAME_MATERIAL = 1300         #non-pawn material at/below which it is the endgame

nodes = 0                       #positions visited in the current search

#Transposition table: zobrist key -> (depth, value, flag, best_move).
_EXACT, _LOWER, _UPPER = 0, 1, 2
_MAX_PLY = 256
_tt = {}
#Move-ordering heuristics, reset per search: two killer moves per ply and a
#(color, piece, to-square) -> cutoff-weight history table.
_killers = [[None, None] for _ in range(_MAX_PLY)]
_history = {}

#Zobrist tables (our own, for the TT key). The piece-placement part is kept
#incrementally; the cheap structural part (side/castling/ep) is recomputed.
_zrng = _random.Random(0xC0FFEE)
_Z_PIECE = [[[_zrng.getrandbits(64) for _ in range(64)] for _ in range(7)] for _ in range(2)]
_Z_SIDE = _zrng.getrandbits(64)
_Z_CASTLE = [_zrng.getrandbits(64) for _ in range(64)]
_Z_EP = [_zrng.getrandbits(64) for _ in range(8)]


class _Timeout(Exception):
    """Raised internally to unwind the search when the time budget is spent."""


def _sign(board):
    return 1 if board.turn == chess.WHITE else -1


def _zp(piece_type, color, square):
    return _Z_PIECE[1 if color else 0][piece_type][square]


#Piece-placement Zobrist (full scan); kept incrementally during search.
def _zobrist_pieces(board):
    k = 0
    for square, piece in board.piece_map().items():
        k ^= _zp(piece.piece_type, piece.color, square)
    return k


#Cheap structural part of the key (side to move, castling rights, ep file),
#recomputed per node so the incremental part never has to track it.
def _structural_key(board):
    k = _Z_SIDE if board.turn == chess.BLACK else 0
    cr = board.castling_rights
    while cr:
        sq = (cr & -cr).bit_length() - 1
        k ^= _Z_CASTLE[sq]
        cr &= cr - 1
    if board.ep_square is not None:
        k ^= _Z_EP[chess.square_file(board.ep_square)]
    return k


#XOR delta to the piece-placement key for a move (call BEFORE board.push).
def _piece_key_delta(board, move):
    piece = board.piece_at(move.from_square)
    color = piece.color
    d = _zp(piece.piece_type, color, move.from_square)
    final_type = move.promotion if move.promotion is not None else piece.piece_type
    d ^= _zp(final_type, color, move.to_square)
    if board.is_en_passant(move):
        cap_sq = move.to_square + (-8 if color == chess.WHITE else 8)
        d ^= _zp(chess.PAWN, not color, cap_sq)
    elif board.is_capture(move):
        victim = board.piece_at(move.to_square)
        d ^= _zp(victim.piece_type, victim.color, move.to_square)
    if board.is_castling(move):
        if board.is_kingside_castling(move):
            rf, rt = (chess.H1, chess.F1) if color == chess.WHITE else (chess.H8, chess.F8)
        else:
            rf, rt = (chess.A1, chess.D1) if color == chess.WHITE else (chess.A8, chess.D8)
        d ^= _zp(chess.ROOK, color, rf) ^ _zp(chess.ROOK, color, rt)
    return d


#Running eval/material for the child of a move (call BEFORE board.push).
#Returns (ev_after, mat_after); ev_after is None when the move flips the game
#phase, signalling the caller to recompute eval_board on the pushed position.
def _child_eval_mat(board, move, ev, mat, is_endgame):
    mat_after = mat
    if board.is_capture(move) and not board.is_en_passant(move):
        vt = board.piece_type_at(move.to_square)
        if vt != chess.PAWN:
            mat_after -= eval.pieceVals[vt]
    if move.promotion is not None:
        mat_after += eval.pieceVals[move.promotion]
    if (mat <= ENDGAME_MATERIAL) == (mat_after <= ENDGAME_MATERIAL):
        return ev + eval.eval_move(board, move, is_endgame), mat_after
    return None, mat_after


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


#Most-valuable-victim / least-valuable-attacker score for a noisy move.
def _mvv_lva(board, move):
    if board.is_en_passant(move):
        victim_v = eval.pieceVals[chess.PAWN]
    else:
        vt = board.piece_type_at(move.to_square)
        victim_v = eval.pieceVals[vt] if vt is not None else 0
    attacker_v = eval.pieceVals[board.piece_type_at(move.from_square)]
    promo_v = (eval.pieceVals[move.promotion] - eval.pieceVals[chess.PAWN]
               if move.promotion is not None else 0)
    return 10 * victim_v - attacker_v + promo_v


#Order moves best-first: hint move (TT/PV), then captures/promotions by MVV-LVA,
#then killer moves, then quiet moves by history score.
def _ordered_moves(board, hint_move=None, ply=None):
    killers = _killers[ply] if (ply is not None and ply < _MAX_PLY) else ()

    def score(m):
        if hint_move is not None and m == hint_move:
            return 3_000_000
        if board.is_capture(m) or m.promotion is not None:
            return 2_000_000 + _mvv_lva(board, m)
        if m in killers:
            return 1_000_000
        piece = board.piece_at(m.from_square)
        return min(_history.get((piece.color, piece.piece_type, m.to_square), 0), 999_999)

    return sorted(board.legal_moves, key=score, reverse=True)


#Noisy moves (captures and promotions) ordered by MVV-LVA for quiescence.
def _ordered_captures(board):
    moves = [m for m in board.legal_moves
             if board.is_capture(m) or m.promotion is not None]
    moves.sort(key=lambda m: _mvv_lva(board, m), reverse=True)
    return moves


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


def _captured_value(board, move):
    if board.is_en_passant(move):
        return eval.pieceVals[chess.PAWN]
    victim = board.piece_at(move.to_square)
    return eval.pieceVals[victim.piece_type] if victim is not None else 0


#Quiescence search: resolve captures/promotions until the position is quiet
#before evaluating, avoiding the horizon effect. Fail-soft, like negamax.
#ev/mat are the running evaluation / non-pawn material of `board`.
def quiescence(board, alpha, beta, ply=0, deadline=None, delta=True, ev=None, mat=None):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1
    if ev is None:
        ev, mat = eval.eval_board(board), eval.nonpawn_material(board)

    term = _evaluate_terminal(board, ply)
    if term is not None:
        return term

    is_endgame = mat <= ENDGAME_MATERIAL

    #In check there is no safe "do nothing" option, so search every evasion.
    if board.is_check():
        best = -INF
        for move in _ordered_moves(board):
            ev_c, mat_c = _child_eval_mat(board, move, ev, mat, is_endgame)
            board.push(move)
            try:
                if ev_c is None:
                    ev_c = eval.eval_board(board)
                score = -quiescence(board, -beta, -alpha, ply + 1, deadline, delta, ev_c, mat_c)
            finally:
                board.pop()
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
            if alpha >= beta:
                break
        return best

    #Stand-pat from the running eval (O(1)).
    stand_pat = _sign(board) * ev
    best = stand_pat
    if best > alpha:
        alpha = best
    if alpha >= beta:
        return best

    for move in _ordered_captures(board):
        #Delta pruning: skip captures too small to raise alpha (never promotions).
        if delta and move.promotion is None:
            if stand_pat + _captured_value(board, move) + DELTA_MARGIN <= alpha:
                continue
        ev_c, mat_c = _child_eval_mat(board, move, ev, mat, is_endgame)
        board.push(move)
        try:
            if ev_c is None:
                ev_c = eval.eval_board(board)
            score = -quiescence(board, -beta, -alpha, ply + 1, deadline, delta, ev_c, mat_c)
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
#the root and delta=False, the returned value equals the true minimax value.
#use_tt=False disables the table for exact, history-independent searches.
def negamax(board, depth, alpha, beta, ply=0, deadline=None, delta=True, use_tt=True,
            ev=None, mat=None, pkey=None):
    global nodes
    if deadline is not None and _time.monotonic() >= deadline:
        raise _Timeout()
    nodes += 1
    if ev is None:
        ev, mat = eval.eval_board(board), eval.nonpawn_material(board)
        pkey = _zobrist_pieces(board)

    alpha_orig = alpha
    key = None
    tt_move = None
    if use_tt:
        key = pkey ^ _structural_key(board)
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
        return quiescence(board, alpha, beta, ply, deadline, delta, ev, mat)

    is_endgame = mat <= ENDGAME_MATERIAL
    best = -INF
    best_move = None
    for move in _ordered_moves(board, tt_move, ply):
        ev_c, mat_c = _child_eval_mat(board, move, ev, mat, is_endgame)
        pkey_c = pkey ^ _piece_key_delta(board, move)
        board.push(move)
        try:
            if ev_c is None:
                ev_c = eval.eval_board(board)
            score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, deadline,
                             delta, use_tt, ev_c, mat_c, pkey_c)
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
    ev, mat = eval.eval_board(board), eval.nonpawn_material(board)
    pkey = _zobrist_pieces(board)
    is_endgame = mat <= ENDGAME_MATERIAL
    alpha, beta = -INF, INF
    moves = _ordered_moves(board, pv_move, 0)
    best_move = moves[0]
    best_score = -INF
    for move in moves:
        ev_c, mat_c = _child_eval_mat(board, move, ev, mat, is_endgame)
        pkey_c = pkey ^ _piece_key_delta(board, move)
        board.push(move)
        try:
            if ev_c is None:
                ev_c = eval.eval_board(board)
            score = -negamax(board, depth - 1, -beta, -alpha, 1, deadline,
                             True, True, ev_c, mat_c, pkey_c)
        finally:
            board.pop()
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
