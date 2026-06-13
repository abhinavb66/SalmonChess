#File containing base values for chess pieces and
#scaling those values based on position and game state
#Inspiration for value scaling based on Tomasz Michniewski:
#https://www.chessprogramming.org/

import cozy_chess as cc

#base value of pieces in centipawns
pieceVals = {
    cc.Piece.Pawn: 100,
    cc.Piece.Knight: 325,
    cc.Piece.Bishop: 330,
    cc.Piece.Rook: 500,
    cc.Piece.Queen: 900,
    cc.Piece.King: 20000 #Somewhat arbitrary value greater than all other pieces combined
}

# PIECE VAL TABLES
#Piece value tables represent the relative increase or decrease
#in a piece's value depending on its position on the board.
#Indexed by square 0..63 (a1=0), matching cozy-chess int(square).

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

pEndTableW = [
    0,   0,   0,   0,   0,   0,   0,   0,
    10,  10,  10,  10,  10,  10,  10,  10,
    15,  15,  15,  15,  15,  15,  15,  15,
    20,  20,  20,  25,  25,  20,  20,  20,
    25,  25,  25,  30,  30,  25,  25,  25,
    30,  30,  35,  40,  40,  35,  30,  30,
    80,  80,  80,  80,  80,  80,  80,  80,
    0,   0,   0,   0,   0,   0,   0,   0]
pEndTableB = list(reversed(pEndTableW))

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

kEndTableW = [
    -50, -40, -30, -20, -20, -30, -40, -50,
    -30, -20, -10,   0,   0, -10, -20, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  30,  40,  40,  30, -10, -30,
    -30, -10,  20,  30,  30,  20, -10, -30,
    -30, -30,   0,   0,   0,   0, -30, -30,
    -50, -30, -30, -30, -30, -30, -30, -50]
kEndTableB = list(reversed(kEndTableW))

#Dictionaries containing piece table values for each color
wTable = {
    cc.Piece.Pawn: pTableW,
    cc.Piece.Knight: nTableW,
    cc.Piece.Bishop: bTableW,
    cc.Piece.Rook: rTableW,
    cc.Piece.Queen: qTable,
    cc.Piece.King: kTableW
}

bTable = {
    cc.Piece.Pawn: pTableB,
    cc.Piece.Knight: nTableB,
    cc.Piece.Bishop: bTableB,
    cc.Piece.Rook: rTableB,
    cc.Piece.Queen: qTable,
    cc.Piece.King: kTableB
}

wEndTable = {
    cc.Piece.Pawn: pEndTableW,
    cc.Piece.Knight: nTableW,
    cc.Piece.Bishop: bTableW,
    cc.Piece.Rook: rTableW,
    cc.Piece.Queen: qTable,
    cc.Piece.King: kEndTableW
}

bEndTable = {
    cc.Piece.Pawn: pEndTableB,
    cc.Piece.Knight: nTableB,
    cc.Piece.Bishop: bTableB,
    cc.Piece.Rook: rTableB,
    cc.Piece.Queen: qTable,
    cc.Piece.King: kEndTableB
}

_NONPAWN = (cc.Piece.Knight, cc.Piece.Bishop, cc.Piece.Rook, cc.Piece.Queen)

#Evaluation functions

#Total non-pawn, non-king material on the board (centipawns).
def nonpawn_material(board):
    return sum(pieceVals[p] * len(board.pieces(p)) for p in _NONPAWN)

#Determines if the position is an endgame or not.
#Endgame when total non-pawn, non-king material on the board is <= 1300 cp
#(roughly two minor pieces + one rook).
def endGame(board):
    return nonpawn_material(board) <= 1300

#Evaluates a position and returns a value. + value is good for white, - for black
def eval_board(board):
    isEndGame = nonpawn_material(board) <= 1300
    total = 0
    for piece in cc.Piece.ALL:
        val = pieceVals[piece]
        wt = wEndTable[piece] if isEndGame else wTable[piece]
        bt = bEndTable[piece] if isEndGame else bTable[piece]
        for sq in board.colored_pieces(cc.Color.White, piece):
            total += val + wt[int(sq)]
        for sq in board.colored_pieces(cc.Color.Black, piece):
            total -= val + bt[int(sq)]
    return total
