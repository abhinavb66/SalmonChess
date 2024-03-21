#Main python file for Salmon chess engine
#Designed to interface through Universal Chess Interface (UCI) with a chess GUI
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

    
