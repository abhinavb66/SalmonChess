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

wEndTable = {
    chess.PAWN: pEndTableW,
    chess.KNIGHT: nTableW,
    chess.BISHOP: bTableW,
    chess.ROOK: rTableW,
    chess.QUEEN: qTable,
    chess.KING: kEndTableW
}

bEndTable = {
    chess.PAWN: pEndTableB,
    chess.KNIGHT: nTableB,
    chess.BISHOP: bTableB,
    chess.ROOK: rTableB,
    chess.QUEEN: qTable,
    chess.KING: kEndTableB
}

#Evaluation functions

#Determines if the position is an endgame or not.
#Endgame when total non-pawn, non-king material on the board is <= 1300 cp
#(roughly two minor pieces + one rook).
def endGame(board):
    minorMajorTypes = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    materialTotal = sum(
        pieceVals[p.piece_type]
        for p in board.piece_map().values()
        if p.piece_type in minorMajorTypes
    )
    return materialTotal <= 1300

#Evaluates a position and returns a float. + value is good for white, - for black
def eval_board(board):
    total = 0
    isEndGame = endGame(board)

    pieces = board.piece_map()
    for square in pieces:
        piece = board.piece_at(square)
        sign = 1 if piece.color == chess.WHITE else -1
        if sign == 1:
            table = wEndTable[piece.piece_type] if isEndGame else wTable[piece.piece_type]
        else:
            table = bEndTable[piece.piece_type] if isEndGame else bTable[piece.piece_type]
        total += sign * (pieceVals[piece.piece_type] + table[square])

    return total

#Evaluates the change in position value from a move
#Assumes move is legal, board is position before move is made
def eval_move(board, move):
    delta = 0
    isEndGame = endGame(board)

    #Subtract value for piece leaving previous square
    piece = board.piece_at(move.from_square)
    sign = 1 if piece.color == chess.WHITE else -1
    if sign == 1:
        table = wEndTable[piece.piece_type] if isEndGame else wTable[piece.piece_type]
    else:
        table = bEndTable[piece.piece_type] if isEndGame else bTable[piece.piece_type]
    delta -= sign * table[move.from_square]

    #Additional steps for a promotion
    if move.promotion is not None:
        delta -= sign * pieceVals[chess.PAWN]
        promoPiece = move.promotion
        if sign == 1:
            promoTable = wEndTable[promoPiece.piece_type] if isEndGame else wTable[promoPiece.piece_type]
        else:
            promoTable = bEndTable[promoPiece.piece_type] if isEndGame else bTable[promoPiece.piece_type]
        delta += sign * (pieceVals[promoPiece] + promoTable[move.to_square])
    else: #Else add value for piece going to new square
        delta += sign * table[move.to_square]

    #Handle captures:
    if board.is_en_passant(move):
        captureSquare = (move.to_square - 8) if sign == 1 else (move.to_square + 8) #Square of captured pawn
        if isEndGame:
            captureTable = pEndTableB if sign == 1 else pEndTableW
        else:
            captureTable = pTableB if sign == 1 else pTableW
        delta += sign * (pieceVals[chess.PAWN] + captureTable[captureSquare])
    elif board.is_capture(move):
        capturePiece = board.piece_at(move.to_square)
        if sign == -1:
            captureTable = wEndTable[capturePiece.piece_type] if isEndGame else wTable[capturePiece.piece_type]
        else:
            captureTable = bEndTable[capturePiece.piece_type] if isEndGame else bTable[capturePiece.piece_type]
        delta += sign * (pieceVals[capturePiece.piece_type] + captureTable[move.to_square])

    return delta