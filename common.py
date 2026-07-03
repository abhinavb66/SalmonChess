#Shared engine plumbing used by both Salmon engines: cozy-chess <-> UCI move
#translation, terminal/draw detection, go-command time-control parsing, and the
#UCI stdin/stdout loop scaffold. Engine-specific playing logic stays out of
#this module.

import math
import sys

import cozy_chess as cc

default_movetime = 1            #seconds per move when the GUI gives no limit

_SQ_NAMES = ["abcdefgh"[i % 8] + str(i // 8 + 1) for i in range(64)]
_PROMO_CHAR = {cc.Piece.Knight: "n", cc.Piece.Bishop: "b",
               cc.Piece.Rook: "r", cc.Piece.Queen: "q"}


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


#cozy's status() does not flag insufficient material, so detect the basic draws.
def insufficient_material(board):
    if (len(board.pieces(cc.Piece.Pawn)) or len(board.pieces(cc.Piece.Rook))
            or len(board.pieces(cc.Piece.Queen))):
        return False
    minors = len(board.pieces(cc.Piece.Knight)) + len(board.pieces(cc.Piece.Bishop))
    return minors <= 1


#Terminal state of the position from the side to move's view:
#"loss" (checkmated), "draw" (stalemate/fifty-move/insufficient material), or
#None when the game is still on. Score mapping is left to the caller.
def terminal_result(board):
    st = board.status()
    if st == cc.GameStatus.Won:         #side to move is checkmated
        return "loss"
    if st == cc.GameStatus.Drawn:       #stalemate or fifty-move
        return "draw"
    if insufficient_material(board):
        return "draw"
    return None


#Parse the integer-valued go parameters into (max_depth, movetime_seconds).
#movetime is math.inf for a pure fixed-depth search ("go depth N").
def parse_go(arg, default_depth, white_to_move):
    params = {}
    i = 1
    while i < len(arg):
        if arg[i] in ("movetime", "depth", "wtime", "btime", "winc", "binc", "nodes"):
            if i + 1 < len(arg):
                try:
                    params[arg[i]] = int(arg[i + 1])
                except ValueError:
                    pass
            i += 2
        else:
            i += 1

    max_depth = params.get("depth", default_depth)

    #Decide the time budget (seconds) from the GUI's instructions
    if "movetime" in params:
        movetime = params["movetime"] / 1000.0
    elif "wtime" in params or "btime" in params:
        remaining = params.get("wtime" if white_to_move else "btime", 0)
        inc = params.get("winc" if white_to_move else "binc", 0)
        movetime = max(0.05, (remaining / 20.0 + inc / 2.0) / 1000.0)
    elif "depth" in params:
        movetime = math.inf     #pure fixed-depth search
    else:
        movetime = default_movetime

    return max_depth, movetime


#UCI stdin/stdout main loop, parameterized by the engine's move function:
#bestmove_fn(board, depth=..., time=..., history=[...]) -> UCI move string.
#UCI protocol: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html
def run_uci_loop(name, author, bestmove_fn, default_depth=10):
    sys.stdout.reconfigure(line_buffering=True)

    board = cc.Board()
    history = []            #position hashes preceding the current one (for repetition)
    depth = default_depth

    while True:
        line = input()
        if not line:
            continue
        arg = line.split()

        #Respond to request for engine info
        if(arg[0] == "uci"):
            print("id name " + name)
            print("id author " + author)
            print("uciok")

        #end program if requested
        elif(arg[0] == "quit"):
            break

        #respond to ready check
        elif(arg[0] == "isready"):
            print("readyok")

        #handle internal parameter change requests
        elif(arg[0] == "setoption"):
            if(arg[2] == "depth"):
                depth = int(arg[4])

        #Set up requested position
        elif(arg[0] == "position"):
            if(arg[1] == "startpos"):
                board = cc.Board()
                moves_start = 2
            elif(arg[1] == "fen"):
                board = cc.Board.from_fen(" ".join(arg[2:8]))
                moves_start = 8
            else:
                continue #bad command

            history = []
            if len(arg) > moves_start and arg[moves_start] == "moves":
                for move in arg[moves_start + 1:]:
                    history.append(board.hash())
                    board.play(from_uci(board, move))

        #Start calculating current position
        elif(arg[0] == "go"):
            white_to_move = board.side_to_move() == cc.Color.White
            max_depth, movetime = parse_go(arg, depth, white_to_move)
            move = bestmove_fn(board, depth=max_depth, time=movetime, history=history)
            print(f"bestmove {move}")
