#Tests for the evaluation module (cozy-chess backend).
#Runnable with pytest, or directly: `python test_eval.py`

import cozy_chess as cc

import eval


def test_eval_board_basics():
    assert eval.eval_board(cc.Board()) == 0  # start position is balanced
    #Color-mirrored positions should negate (also guards square indexing).
    a = cc.Board.from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    b = cc.Board.from_fen("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert eval.eval_board(a) == -eval.eval_board(b)
    #Up a queen is a large advantage.
    assert eval.eval_board(cc.Board.from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1")) > 800


def test_nonpawn_material():
    #Two knights, two bishops, two rooks, a queen per side.
    assert eval.nonpawn_material(cc.Board()) == 2 * (2 * 325 + 2 * 330 + 2 * 500 + 900)


def test_endgame_detection():
    assert eval.endGame(cc.Board()) is False
    assert eval.endGame(cc.Board.from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")) is True


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
