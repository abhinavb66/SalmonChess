#Main python file for Salmon chess engine
#Designed to interface through Universal Chess Interface (UCI) with a chess GUI
#UCI protocol: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html
#This class handles communication with the GUI through the UCI protocol
#Built by Abhinav Brahmarouthu

import sys
import math
import chess
import minimax

VERSION = "0.1"

sys.stdout.reconfigure(line_buffering=True)

board = chess.Board()   #new board, change later to accept board from GUI
depth = 10;             #Constant depth for testing, change later

#main loop to monitor for commands from GUI
while True:
    line = input()
    if not line:
        continue
    arg = line.split()

    #Respond to request for engine info
    if(arg[0] == "uci"):
        print("id name Salmon" + VERSION)
        print("id name Abhinav Brahmarouthu")
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
            board.reset()
            moves_start = 2
        elif(arg[1] == "fen"):
            board.set_fen(" ".join(arg[2:8]))
            moves_start = 8
        else:
            continue #bad command

        if len(arg) > moves_start and arg[moves_start] == "moves":
            for move in arg[moves_start + 1:]:
                board.push_uci(move)

    #Start calculating current position
    elif(arg[0] == "go"):
        #Parse the integer-valued go parameters the GUI sent
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

        max_depth = params.get("depth", depth)

        #Decide the time budget (seconds) from the GUI's instructions
        if "movetime" in params:
            movetime = params["movetime"] / 1000.0
        elif "wtime" in params or "btime" in params:
            remaining = params.get("wtime" if board.turn == chess.WHITE else "btime", 0)
            inc = params.get("winc" if board.turn == chess.WHITE else "binc", 0)
            movetime = max(0.05, (remaining / 20.0 + inc / 2.0) / 1000.0)
        elif "depth" in params:
            movetime = math.inf     #pure fixed-depth search
        else:
            movetime = minimax.default_movetime

        move = minimax.bestMove(board, depth=max_depth, time=movetime)
        print(f"bestmove {move}")
