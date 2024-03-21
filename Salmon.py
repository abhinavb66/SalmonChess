#Main python file for Salmon chess engine
#Designed to interface through Universal Chess Interface (UCI) with a chess GUI
#UCI protocol: https://www.wbec-ridderkerk.nl/html/UCIProtocol.html
#This class handles communication with the GUI through the UCI protocol
#Built by Abhinav Brahmarouthu

import chess, sys

VERSION = "0.1"

board = chess.board()   #new board, change later to accept board from GUI
depth = 10;             #Constant depth for testing, change later

#main loop to monitor for commands from GUI
while True:
    arg = input().split()

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
        return("readyok")
    
    #handle internal parameter change requests
    elif(arg[0] == "setoption"):
        if(arg[2] == "depth"):
            depth = int(arg[4])

    #Set up requested position
    elif(arg[0] == "position"):
        if(arg[1] == "startpos"):
            board.reset()
        elif(arg[1] == "fen"):
            board.set_fen(" ".join(arg[2:8]))
        else:
            break #bad command
        
        start = arg.index("moves")
        for move in arg[start+1:]:
            board.uci_push(move)

    #Start calculating current position
    elif(arg[0] == "go"):
        calc = 1

    #Stop calculating current postition and return best move
    elif(arg[0] == "stop"):
        move = 1
    
