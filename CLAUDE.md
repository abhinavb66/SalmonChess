# CLAUDE.md

Guidance for working on SalmonChess, a chess engine written in Python.

## Overview

SalmonChess is a UCI chess engine built on the `python-chess` library. It
communicates with chess GUIs over the Universal Chess Interface (UCI) protocol
and ships with a small local web UI for playing against it.

## Architecture

- `Salmon.py` — UCI front-end and main loop. Reads UCI commands on stdin
  (`uci`, `isready`, `setoption`, `position`, `go`, `quit`) and drives the
  search. The `go` handler parses time controls (`movetime`, `depth`,
  `wtime`/`btime`/`winc`/`binc`) into a time budget.
- `minimax.py` — the search: negamax with fail-soft alpha-beta pruning,
  iterative deepening within a time budget, and a quiescence search (with
  delta pruning) at the leaves. `bestMove(board, depth, time)` is the entry
  point; pass `time=math.inf` for a pure fixed-depth search.
- `eval.py` — static evaluation. Centipawn piece values plus piece-square
  tables (with separate middlegame/endgame tables for pawns and kings).
  `eval_board(board)` returns an absolute score (+ favors White, - favors
  Black); `eval_move(board, move)` returns the incremental delta of a move.
- `UI/` — a Flask + chessboard.js web app. It drives the engine over UCI as a
  subprocess (`UI/uci_client.py`), exercising the same interface a real GUI
  would.

## Conventions

- Evaluation is White-relative (absolute). The search converts to the
  side-to-move (negamax) convention by multiplying by `_sign(board)`.
- Search pruning comes in two kinds: alpha-beta and quiescence cutoffs are
  *exact* (value-preserving); delta pruning is *heuristic* (lossy). Search
  functions take a `delta` flag — pass `delta=False` for an exact,
  value-preserving search.
- Comments use `#comment` (no leading space) to match the existing style.

## Running and testing

Install dependencies (see `requirements.txt`):

```
pip install python-chess flask
```

Run the engine directly (speaks UCI on stdin/stdout):

```
python3 Salmon.py
```

Run the web UI:

```
python3 UI/server.py
```

Run the test suites (plain Python, also pytest-compatible):

```
python3 test_eval.py
python3 test_minimax.py
```

### Testing notes

- The strongest correctness check is the property that pruned search equals an
  un-pruned reference. `test_minimax.py` verifies this for both negamax
  alpha-beta and quiescence, running them in exact mode (`delta=False`).
- Keep the test suites fast (a few seconds). The un-pruned reference searches
  are exponential, so cap depth per position and prefer sparse positions for
  the deeper cases.
