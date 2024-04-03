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
pTableB = list(reversed(pTableW))

nTableW = [
    -50, -40, -30, -30, -30, -30, -40, -50,
    -40, -20, 0, 0, 0, 0, -20, -40,
    -30, 0, 10, 15, 15, 10, 0, -30,
    -30, 5, 15, 20, 20, 15, 5, -30,
    -30, 0, 15, 20, 20, 15, 0, -30,
    -30, 5, 25, 35, 35, 25, 5, -30,
    -40, -20, 0, 5, 5, 0, -20, -40,
    -50, -40, -30, -30, -30, -30, -40, -50]
nTableB = list(reversed(nTableW))

bTableW = [
    -20, -10, -10, -10, -10, -10, -10, -20,
    -10, 5, 0, 0, 0, 0, 5, -10,
    -10, 10, 10, 10, 10, 10, 10, -10,
    -10, 0, 10, 10, 10, 10, 0, -10,
    -10, 5, 5, 10, 10, 5, 5, -10,
    -10, 0, 5, 10, 10, 5, 0, -10,
    -10, 0, 0, 0, 0, 0, 0, -10,
    -20, -10, -10, -10, -10, -10, -10, -20]
bTableB = list(reversed(bTableW))

rTableW = [
    0, 0, 0, 5, 5, 0, 0, 0,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    -5, 0, 0, 0, 0, 0, 0, -5,
    5, 10, 10, 10, 10, 10, 10, 5,
    0, 0, 0, 0, 0, 0, 0, 0]
rTableB = list(reversed(rTableW))

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
kTableB = list(reversed(kTableW))

#Dictionaries contianing piece table values for each color
wTable = {
    chess.PAWN: pTableW,
    chess.KNIGHT: nTableW,
    chess.BISHOP: bTableW,
    chess.ROOK: rTableW,
    chess.QUEEN: qTable,
    chess.KING: kTableW
}

bTable = {
    chess.PAWN: pTableB,
    chess.KNIGHT: nTableB,
    chess.BISHOP: bTableB,
    chess.ROOK: rTableB,
    chess.QUEEN: qTable,
    chess.KING: kTableB
}

#Evaluation functions

#Evaluates a position and returns a float. + value is good for white, - for black
def eval_board(board):
    total = 0

    pieces = board.piece_map()
    for square in pieces:
        piece = board.piece_at(square)
        sign = 1 if piece.color == chess.WHITE else -1
        table = wTable[piece.piece_type] if sign == 1 else bTable[piece.piece_type]
        total += sign * (pieceVals[piece.piece_type] + table[square])
    
    return total

#Evaluates the change in position value from a move
#Assumes move is legal
def eval_move(bosrd, move):
    total = 0

    #TODO
    #Check for capture and calc capured piece value loss
    #Subtrace from piece value
    #Add to piece value
    #Do i need to consider promotions?


    return total

#Determines if the position is an endgame or not
#TODO
def endGame(board):
    val = False

    pieces = board.piece_map()

    return val