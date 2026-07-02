# CLAUDE.md

Guidance for working on SalmonChess, a chess engine written in Python.

## Overview

SalmonChess is a UCI chess engine. The engine core uses `cozy-chess` (a fast
native move generator) for board representation, while `python-chess` is kept
for the tests (as a correctness oracle) and the web UI. It communicates with
chess GUIs over the Universal Chess Interface (UCI) protocol and ships with a
small local web UI for playing against it.

## Architecture

- `Salmon.py` — UCI front-end and main loop. Reads UCI commands on stdin
  (`uci`, `isready`, `setoption`, `position`, `go`, `quit`) and drives the
  search. The `go` handler parses time controls (`movetime`, `depth`,
  `wtime`/`btime`/`winc`/`binc`) into a time budget.
- `minimax.py` — the search over cozy-chess boards: negamax with fail-soft
  alpha-beta pruning, iterative deepening within a time budget, a quiescence
  search (with delta pruning) at the leaves, a transposition table (keyed by
  cozy's native `board.hash()`), and killer/history move ordering. Boards are
  advanced by `copy.copy` + `play` (cozy has no unmake). `bestMove(board,
  depth, time, history)` is the entry point; pass `time=math.inf` for a pure
  fixed-depth search. `to_uci`/`from_uci` translate cozy's king-takes-rook
  castling encoding to/from standard UCI at the boundary.
- `eval.py` — static evaluation over cozy boards. Centipawn piece values plus
  piece-square tables (separate middlegame/endgame tables for pawns and kings).
  `eval_board(board)` returns an absolute score (+ favors White, - favors
  Black).
- `UI/` — a Flask + chessboard.js web app. It drives the engine over UCI as a
  subprocess (`UI/uci_client.py`), exercising the same interface a real GUI
  would. The UI keeps a `python-chess` board and converts via FEN when calling
  `eval_board`.

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
pip install cozy-chess-py python-chess flask
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
python3 test_perft.py     # move-gen correctness (perft + vs python-chess)
python3 test_eval.py
python3 test_minimax.py
```

### Testing notes

- `test_perft.py` validates the cozy move generator against published perft
  counts and against `python-chess` (the oracle) over random games — run it
  first if you touch anything move-related.
- The strongest search correctness check is the property that pruned search
  equals an un-pruned reference. `test_minimax.py` verifies this for both
  negamax alpha-beta and quiescence, running them in exact mode (`delta=False`),
  plus that the transposition table preserves the exact value.
- Keep the test suites fast (a few seconds). The un-pruned reference searches
  are exponential, so cap depth per position and prefer sparse positions for
  the deeper cases.

## Roadmap / TODO

Deferred follow-ups, roughly by impact per effort:

### Performance (the Python search loop is now the bottleneck)
- **Read terminals off the move list** instead of calling `status()` per node.
  cozy's `status()` runs its own move generation to detect checkmate/stalemate,
  roughly doubling movegen per node. Generate moves once; an empty list plus
  `checkers()` distinguishes mate from stalemate. Likely the biggest easy win.
- **Re-add incremental evaluation** on the cozy board. The migration dropped the
  incremental eval/`eval_move` threading, so `eval_board` is recomputed at every
  leaf.
- **Compiled full-search rewrite** (Cython, or Rust + PyO3 writing the search on
  top of cozy/shakmaty) — the order-of-magnitude option; only if leaving the
  Python loop entirely is worth it.

### Strength measurement
- **Run the full gauntlet** (`gauntlet.py`) for a measured Elo + confidence
  interval, replacing the current estimate.

### Evaluation & search features (biggest lever on playing strength)
- Richer evaluation: king safety, pawn structure, mobility, passed pawns.
- Principal-variation / aspiration windows.
- Opening book; endgame tablebases.

### Correctness refinements (minor)
- `_insufficient_material` is simplified: it does not catch same-colored-bishop
  (KBvKB) draws or KNN nuances that python-chess handles.
- Repetition is a 2-fold-in-search heuristic (seeded from game history) rather
  than strict threefold.

### Housekeeping
- `bench_speed.py`'s old PyPy angle is stale — cozy-chess is a Rust extension, so
  the CPython-vs-PyPy comparison may not apply (PyPy wheel availability unverified).
