#Tests for the evaluation module.
#Runnable with pytest, or directly: `python test_eval.py`

import random

import chess

import eval


#Core invariant: the incremental delta from eval_move must equal the
#difference in full static evaluations before and after the move.
#This guards against bugs in special-move handling (captures, en passant,
#promotion, castling) where eval_move and eval_board can drift apart.
def test_eval_move_matches_board_delta():
    random.seed(0)
    mismatches = []
    for _ in range(300):
        board = chess.Board()
        for _ in range(80):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            before = eval.eval_board(board)
            delta = eval.eval_move(board, move)
            board.push(move)
            after = eval.eval_board(board)
            if delta != after - before:
                mismatches.append((board.fen(), move.uci(), delta, after - before))
    assert not mismatches, f"eval_move/eval_board disagreement: {mismatches[:5]}"


#Castling moves the rook as well as the king; eval_move must account for both.
def test_castling_consistency():
    start = "r3k2r/8/8/8/8/8/8/R3K2R {} KQkq - 0 1"
    cases = [
        ("w", "e1g1"),  # white kingside
        ("w", "e1c1"),  # white queenside
        ("b", "e8g8"),  # black kingside
        ("b", "e8c8"),  # black queenside
    ]
    for side, uci in cases:
        board = chess.Board(start.format(side))
        move = chess.Move.from_uci(uci)
        assert board.is_castling(move)
        before = eval.eval_board(board)
        delta = eval.eval_move(board, move)
        board.push(move)
        assert delta == eval.eval_board(board) - before, f"castling {uci} mismatch"


def test_promotion_consistency():
    #Plain promotion, promotion-with-capture, and underpromotion.
    cases = [
        ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q"),
        ("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1", "b7a8q"),
        ("4k3/8/8/8/8/8/6p1/4K3 b - - 0 1", "g2g1n"),
    ]
    for fen, uci in cases:
        board = chess.Board(fen)
        before = eval.eval_board(board)
        delta = eval.eval_move(board, chess.Move.from_uci(uci))
        board.push(chess.Move.from_uci(uci))
        assert delta == eval.eval_board(board) - before, f"promotion {uci} mismatch"


def test_eval_board_basics():
    assert eval.eval_board(chess.Board()) == 0  # start position is balanced
    #Mirrored positions should negate.
    a = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    b = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert eval.eval_board(a) == -eval.eval_board(b)
    assert eval.eval_board(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) > 800


def test_endgame_detection():
    assert eval.endGame(chess.Board()) is False
    assert eval.endGame(chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")) is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(1 if failures else 0)
