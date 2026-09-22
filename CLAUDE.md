# CLAUDE.md

Guidance for working on SalmonChess, a chess project with two engines written
in Python.

## Overview

SalmonChess ships two UCI engines over cozy-chess boards (`cozy-chess` is a
fast native move generator; `python-chess` is kept for the tests as a
correctness oracle, for the web UI, and for the training pipeline):

- **Salmon** (`Salmon.py`) — classical alpha-beta search + hand-written
  evaluation. ~2000 Elo measured vs Stockfish UCI_Elo anchors.
- **SalmonNNUE** (`SalmonNNUE.py`) — NNUE-style value+policy network driving
  a Monte Carlo Tree Search (an Lc0-shaped search with an NNUE-style net).
  Ships a Rust hot loop (`rust/`) that the engine prefers when built, with a
  pure-Python fallback so the repo works without a Rust toolchain.

Both speak UCI on stdin/stdout through the shared front end in `common.py`,
and both are playable (and pit-able against each other) in the local web UI.

## Architecture

- `common.py` — shared engine plumbing: cozy<->UCI move translation
  (`to_uci`/`from_uci`, handling cozy's king-takes-rook castling encoding),
  `terminal_result` (mate/stalemate/fifty-move/insufficient material),
  `parse_go` (time controls: `movetime`, `depth`, `wtime`/`btime`/`winc`/
  `binc`), and `run_uci_loop(name, author, bestmove_fn)` — the whole UCI
  stdin loop, parameterized by `bestmove_fn(board, depth, time, history)`.
- `Salmon.py` — thin entry point binding `run_uci_loop` to `minimax.bestMove`.
- `minimax.py` — the classical search: negamax with fail-soft alpha-beta,
  iterative deepening within a time budget, quiescence (with delta pruning),
  a transposition table keyed by cozy's native `board.hash()`, and
  killer/history move ordering. Boards advance by `copy.copy` + `play`.
- `eval.py` — classical static evaluation: centipawn piece values plus
  piece-square tables (separate middlegame/endgame tables for pawns/kings).
  `eval_board(board)` is White-relative (+ favors White).
- `SalmonNNUE.py` — thin entry point; loads the net from `$SALMON_NNUE_NET`,
  else the current default version (`_DEFAULT_NET`), else fixed-seed random
  weights (bootable before training). Trained nets are versioned artifacts
  (`nets/salmon_nnue_vN.npz`, history in `nets/README.md`); bumping the
  default is part of committing a new net.
- `nnue/network.py` — 768-feature per-perspective encoding (piece x color x
  square, stm-mirrored), `[768,256]` shared-weight accumulator, 512-concat ->
  L1[512,32] -> sigmoid value head, plus a [512,4096] from-to policy head.
  NumPy float32 inference; float16 `.npz` weights; `Accumulator` updates
  incrementally on make-move (castling falls back to full recompute).
- `nnue/mcts.py` — PUCT tree search: policy priors (`PRIOR_MODE`
  policy/heuristic/blend), value-head leaf evaluation, terminal caching,
  2-fold path repetition seeded from game history, time- or sims-capped.
- `rust/` — a Rust re-implementation of the NNUE inference + MCTS as a PyO3
  extension (`salmon_nnue_rs`), mirroring `nnue/mcts.py` and
  `nnue/network.py` semantics exactly (parity-tested). ~30x faster than the
  Python search on this machine. `nnue/rust_engine.py` adapts it to the
  `bestMove` contract; `SalmonNNUE.py` uses it when importable, else the
  Python MCTS (`SALMON_NNUE_ENGINE=python` forces the fallback).
- `nnue/training/gen_data.py` — parallel, resumable Stockfish labeling of
  sampled positions into `data/shard_*.npz` (gitignored).
- `nnue/training/train.py` — PyTorch training (torch is training-only;
  play-time inference is NumPy) exporting to the `.npz` the engine loads.
- `UI/` — Flask + chessboard.js web app driving the engines over UCI as
  subprocesses (`UI/uci_client.py`). Per-side engine dropdowns (White/Black)
  let Salmon and SalmonNNUE play each other. The UI keeps a `python-chess`
  board and converts via FEN when calling `eval_board`.

## Conventions

- Evaluation is White-relative (absolute). The searches convert to the
  side-to-move convention (negamax `_sign`, MCTS value flip) internally.
- Classical search pruning comes in two kinds: alpha-beta and quiescence
  cutoffs are *exact* (value-preserving); delta pruning is *heuristic*
  (lossy). Search functions take a `delta` flag — pass `delta=False` for an
  exact, value-preserving search.
- MCTS values are in [-1,1] from the perspective of the player to move at
  the parent; the net's sigmoid value maps via `q = 2v - 1`.
- Comments use `#comment` (no leading space) to match the existing style.

## Running and testing

Install play-time dependencies (see `requirements.txt`):

```
pip install cozy-chess-py python-chess flask numpy
```

Run an engine directly (speaks UCI on stdin/stdout):

```
python3 Salmon.py
python3 SalmonNNUE.py
```

Run the web UI (per-side engine selection lives in the sidebar):

```
python3 UI/server.py
```

Run the test suites (plain Python, also pytest-compatible):

```
python3 test_perft.py     # move-gen correctness (perft + vs python-chess)
python3 test_eval.py
python3 test_minimax.py
python3 test_nnue.py      # feature/accumulator/inference guards
python3 test_mcts.py
python3 test_uci.py       # UCI smoke test of both entry points
python3 test_rust_engine.py  # Rust<->Python parity (skips if not built)
```

Benchmarks: `bench_speed.py` (alpha-beta nps), `bench_mcts.py` (MCTS sims/s,
Python vs Rust when built).

### Building the Rust engine (optional, speeds up SalmonNNUE ~30x)

```
pip install maturin
cd rust && maturin build --release
pip install target/wheels/salmon_nnue_rs-*.whl
```

(`maturin develop` needs a virtualenv; `build` + `pip install` works without
one.) Binaries are gitignored — the engine falls back to pure Python when the
extension is absent, so building is never required. The default release build
is portable; the dot products are hand-vectorized (8 accumulator lanes) so no
`target-cpu=native` is needed.

### Training the NNUE net

Training needs torch (CPU wheel is fine):

```
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cpu
python3 -m nnue.training.gen_data     # resumable; GEN_SHARDS/GEN_SHARD_SIZE/GEN_SF_DEPTH env
python3 -m nnue.training.train --smoke     # quick sanity run
python3 -m nnue.training.train --epochs 20     # exports nets/salmon_nnue_dev.npz
```

Strength: `GAUNTLET_ENGINE=SalmonNNUE.py python3 gauntlet.py` (vs Stockfish
UCI_Elo anchors) and `python3 gauntlet.py h2h 20` (Salmon vs SalmonNNUE).
Measure a candidate via `SALMON_NNUE_NET=nets/salmon_nnue_dev.npz`; commit it
only if it beats the current default, as the next `nets/salmon_nnue_vN.npz`
(update `_DEFAULT_NET` in `SalmonNNUE.py` and `nets/README.md` with it).

### Testing notes

- `test_perft.py` validates the cozy move generator against published perft
  counts and against `python-chess` (the oracle) over random games — run it
  first if you touch anything move-related.
- The strongest classical-search check is that pruned search equals an
  un-pruned reference (`test_minimax.py`, exact mode `delta=False`), plus
  that the transposition table preserves the exact value.
- The strongest NNUE checks are that the incremental accumulator equals a
  full recompute across random games (`test_nnue.py`) and that NumPy
  inference matches the torch training module (skips cleanly without torch).
- Tests use `make_random_net(seed=0)` — never the trained artifact.
- Keep the test suites fast (a few seconds). The un-pruned reference searches
  are exponential, so cap depth per position and prefer sparse positions for
  the deeper cases.

## Roadmap / TODO

Deferred follow-ups, roughly by impact per effort:

### Classical engine performance (the Python search loop is the bottleneck)
- **Read terminals off the move list** instead of calling `status()` per node.
  cozy's `status()` runs its own move generation to detect checkmate/stalemate,
  roughly doubling movegen per node. Generate moves once; an empty list plus
  `checkers()` distinguishes mate from stalemate. Likely the biggest easy win.
- **Re-add incremental evaluation** on the cozy board (`eval_board` is
  recomputed at every leaf).
- **Compiled full-search rewrite** (Cython, or Rust + PyO3 on top of
  cozy/shakmaty) — the order-of-magnitude option.

### SalmonNNUE
- **int8/SIMD quantized inference** in the Rust engine (currently float32);
  the net eval is the remaining per-simulation bottleneck.
- **Leaf batching + virtual loss** in MCTS (amortize net calls), and **tree
  reuse across moves** (the Rust arena is rebuilt per search).
- **Bigger training run** (more positions, higher label depth) and net-size
  experiments; measure each net with the gauntlet before committing it.
- **Underpromotion policy cells** (promotions currently share one from-to
  cell) and Dirichlet noise if self-play training is ever added.

### Evaluation & search features (classical engine strength)
- Richer evaluation: king safety, pawn structure, mobility, passed pawns.
- Principal-variation / aspiration windows.
- Opening book; endgame tablebases.

### Correctness refinements (minor)
- `insufficient_material` (common.py) is simplified: it does not catch
  same-colored-bishop (KBvKB) draws or KNN nuances python-chess handles.
- Repetition is a 2-fold-in-search heuristic (seeded from game history) in
  both engines rather than strict threefold.

### New engines
- **A drawing engine that out-draws Drawfish** — a third engine tuned to
  steer games toward draws (for example, a contempt that rewards draws,
  and eval/search biased toward simplification, fortresses and repetition),
  measured by draw rate against Drawfish.

### Housekeeping
- `bench_speed.py`'s old PyPy angle is stale — cozy-chess is a Rust extension,
  so the CPython-vs-PyPy comparison may not apply.
