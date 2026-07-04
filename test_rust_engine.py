#Tests for the Rust engine extension (salmon_nnue_rs). They validate that the
#Rust hot loop mirrors the pure-Python engine: hash/UCI parity now, and
#inference + search parity as those land. Skips cleanly when the extension is
#not built, so the suite runs on a pure-Python checkout.
#Runnable with pytest, or directly: `python test_rust_engine.py`

import random

import cozy_chess as cc

import common

try:
    import salmon_nnue_rs
    AVAILABLE = True
except ImportError:
    AVAILABLE = False


def _skip():
    print("  (salmon_nnue_rs not built - skipped)")


#Rust board.hash() must equal cozy_chess_py's over random games — this is what
#lets game-history repetition hashes pass straight through to the Rust search.
def test_hash_parity():
    if not AVAILABLE:
        return _skip()
    random.seed(7)
    for _ in range(8):
        board = cc.Board()
        for _ in range(40):
            assert salmon_nnue_rs.zobrist_hash(board.fen()) == board.hash(), \
                f"hash mismatch at {board.fen()}"
            moves = list(board.generate_moves())
            if not moves:
                break
            board.play(random.choice(moves))


#Rust legal_moves_uci must match common.to_uci over generate_moves as ORDERED
#lists (search tie-breaks depend on identical move order), covering castling
#and promotion UCI normalization.
def test_uci_parity():
    if not AVAILABLE:
        return _skip()
    random.seed(11)
    fixtures = [
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",      # castling both sides
        "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",            # promotion
        "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",  # en passant
    ]
    boards = [cc.Board.from_fen(f) for f in fixtures]
    for _ in range(6):                               # plus random-game positions
        b = cc.Board()
        for _ in range(30):
            boards.append(cc.Board.from_fen(b.fen()))
            ms = list(b.generate_moves())
            if not ms:
                break
            b.play(random.choice(ms))
    for board in boards:
        py = [common.to_uci(board, m) for m in board.generate_moves()]
        rs = salmon_nnue_rs.legal_moves_uci(board.fen())
        assert py == rs, f"uci order mismatch at {board.fen()}:\n py={py}\n rs={rs}"


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
