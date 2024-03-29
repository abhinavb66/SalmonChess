#File containing base values for chess pieces and 
#scaling those values based on position and game state
#Inspiration for value scaling based on Tomasz Michniewski:
#https://www.chessprogramming.org/

import chess

#base value of pieces in centipawns
pieceVals = {
    chess.PAWN: 100,
    chess.KNIGHT: 325,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000 #Somewhat arbitrary value greater than all other pieces combined
}

# PIECE VAL TABLES
#Piece value tables represent the relative increase or decrease 
#in a piece's value depending on its position on the board

#Static pawn table, TODO consider adding endgame pawn table
pTableW = [
    0,  0,  0,  0,  0,  0,  0,  0,
    0, 10, 10, -20, -20, 10, 10,  0,
    0, -5, -10,  5,  5, -15, -5,  0,
    -5,  0,  0, 20, 20,  0,  0,  -5,
    0,  5, 10, 25, 25, 10,  5,  0,
    10, 10, 20, 30, 30, 20, 10, 10,
    75, 100, 75, 65, 65, 75, 100, 75,
    0, 0, 0, 0, 0, 0, 0, 0]
pTableB = reversed(pTableW)

nTableW = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 25, 35, 35, 25, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]
nTableB = reversed(nTableW)

bTableW = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]
btableB = reversed(bTableW)

rTableW = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]
rTableB = reversed(rTableW)

qTable = [
    -20, -10, -10, -5, -5, -10, -10, -20,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -10, 0, 5, 5, 5, 5, 0, -10,
    -5, 0, 5, 5, 5, 5, 0, -5,
    0, 0, 5, 5, 5, 5, 0, -5,
    -10, 5, 5, 5, 5, 5, 0, -10,
    -10, 0, 5, 0, 0, 0, 0, -10,
    -20, -10, -10, -5, -5, -10, -10, -20]

#TODO need to add endgame king table
kTableW = [
    20, 30, 10, 0, 0, 10, 30, 20,
    20, 20, 0, 0, 0, 0, 20, 20,
    -10, -20, -20, -20, -20, -20, -20, -10,
    20, -30, -30, -40, -40, -30, -30, -20,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30,
    -30, -40, -40, -50, -50, -40, -40, -30]
kTableB = reversed(kTableW)

#Evaluation functions

#Evaluates a position and returns a float. + value is good for white, - for black
def eval_board(board):
    total = 0

    return total

#Evaluates the change in position value from a move
def eval_move(bosrd, move):
    total = 0

    return total

#Determines if the position is an endgame or not
def endGame(board):
    val = False
    return val