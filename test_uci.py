#UCI smoke tests: drive each engine entry point as a subprocess the way a real
#GUI (or UI/uci_client.py) would, and check the minimal UCI contract.
#Runnable with pytest, or directly: `python test_uci.py`

import os
import subprocess
import sys
import time

import cozy_chess as cc

import common

HERE = os.path.dirname(os.path.abspath(__file__))

#Entry points under test: (script, expected id-name prefix).
ENGINES = [
    ("Salmon.py", "Salmon"),
]


class _Engine:
    def __init__(self, script):
        env = dict(os.environ)
        self.proc = subprocess.Popen(
            [sys.executable, "-u", os.path.join(HERE, script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            bufsize=1, env=env)

    def send(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix, timeout=15.0):
        end = time.monotonic() + timeout
        lines = []
        while time.monotonic() < end:
            line = self.proc.stdout.readline().strip()
            lines.append(line)
            if line.startswith(prefix):
                return line, lines
        raise AssertionError(f"no '{prefix}' within {timeout}s; got {lines[-5:]}")

    def quit(self):
        try:
            self.send("quit")
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def _smoke(script, name_prefix):
    eng = _Engine(script)
    try:
        eng.send("uci")
        line, lines = eng.read_until("uciok")
        assert any(l.startswith("id name " + name_prefix) for l in lines), lines

        eng.send("isready")
        eng.read_until("readyok")

        #startpos with moves -> legal bestmove in the resulting position
        eng.send("position startpos moves e2e4 e7e5")
        eng.send("go movetime 300")
        line, _ = eng.read_until("bestmove")
        move = line.split()[1]
        board = cc.Board()
        board.play(common.from_uci(board, "e2e4"))
        board.play(common.from_uci(board, "e7e5"))
        legal = {common.to_uci(board, m) for m in board.generate_moves()}
        assert move in legal, f"{script}: bestmove {move} not legal"

        #fen position -> legal bestmove
        fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 0 1"
        eng.send(f"position fen {fen}")
        eng.send("go movetime 300")
        line, _ = eng.read_until("bestmove")
        move = line.split()[1]
        board = cc.Board.from_fen(fen)
        legal = {common.to_uci(board, m) for m in board.generate_moves()}
        assert move in legal, f"{script}: bestmove {move} not legal in fen pos"
    finally:
        eng.quit()
    assert eng.proc.poll() is not None, f"{script}: did not exit on quit"


def test_uci_smoke():
    for script, name_prefix in ENGINES:
        _smoke(script, name_prefix)


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
