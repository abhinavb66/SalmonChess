"""Thin UCI subprocess client for driving Salmon.py from the web UI."""

import os
import subprocess
import sys
import threading


ENGINE_PATH = os.path.join(os.path.dirname(__file__), "..", "Salmon.py")


class UCIClient:
    def __init__(self, cmd=None, read_timeout=10.0):
        if cmd is None:
            cmd = [sys.executable, "-u", os.path.abspath(ENGINE_PATH)]
        self.read_timeout = read_timeout
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until_prefix("uciok")
        self._send("isready")
        self._read_until_prefix("readyok")

    def _send(self, line):
        if self.proc.poll() is not None:
            raise RuntimeError("engine process has exited")
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _read_until_prefix(self, prefix):
        result = {"line": None}

        def reader():
            for raw in self.proc.stdout:
                stripped = raw.strip()
                if stripped.startswith(prefix):
                    result["line"] = stripped
                    return

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(self.read_timeout)
        if result["line"] is None:
            raise TimeoutError(f"engine did not respond with '{prefix}' within {self.read_timeout}s")
        return result["line"]

    def set_option(self, name, value):
        self._send(f"setoption name {name} value {value}")

    def bestmove(self, board, movetime_ms=1000):
        self._send(f"position fen {board.fen()}")
        self._send(f"go movetime {movetime_ms}")
        line = self._read_until_prefix("bestmove ")
        parts = line.split()
        return parts[1] if len(parts) >= 2 else None

    def quit(self):
        try:
            if self.proc.poll() is None:
                self._send("quit")
                self.proc.wait(timeout=2.0)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
