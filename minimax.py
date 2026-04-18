#Minimax algorithm implementation for a basic chess engine

import random
import chess

default_movetime = 1    #Take 1 second per move unless specified

#Find the best move for the given position
#PLACEHOLDER: returns a random legal move until real minimax + alpha-beta lands.
#Keeps the UCI/UI pipeline exercisable end-to-end.
def bestMove(board, depth=0, time=default_movetime):
    legal = list(board.legal_moves)
    if not legal:
        return "0000"
    return random.choice(legal).uci()
