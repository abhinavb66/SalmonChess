# Trained nets

Versioned, immutable artifacts: a new net is committed as the next
`salmon_nnue_vN.npz` alongside its predecessors, never by overwriting one.
The engine's default is `_DEFAULT_NET` in `SalmonNNUE.py` (currently **v3**);
bumping it is part of committing a new net. Any version can be played
without code changes:

```
SALMON_NNUE_NET=nets/salmon_nnue_v2.npz python3 SalmonNNUE.py
```

All versions share the same architecture (768-feature per-perspective
encoding -> [768,256] accumulator -> 512-concat -> L1[512,32] -> sigmoid
value head, plus a [512,4096] from-to policy head, float16 `.npz`), so they
are interchangeable at load time — they differ only in training data and
schedule. A net is only committed if it measures stronger than the current
default (gauntlet anchor sweep + h2h, single-worker protocol — see
`gauntlet.py`); training candidates live in the gitignored
`salmon_nnue_dev.npz` until they pass that gate.

## Version history

| Net | Training data | Schedule | Measured strength (1s/move) |
|---|---|---|---|
| `salmon_nnue_v2.npz` | 4M Stockfish depth-10 positions | 20 epochs, loss 2.39→1.29 | ~1711 Elo (Python backend, ~3.4k sims/s) / ~2191 (Rust backend, ~56k sims/s); h2h vs Salmon 4.0/10 (PRs #8, #9) |
| `salmon_nnue_v3.npz` | 8M Stockfish depth-10 positions (v2's 4M + 80 new shards) | 30 epochs, loss 2.01→1.259 | ~2336 Elo (Rust backend); beat v2 9.5/10 h2h; **first net to beat classical Salmon** (8.0/10) (PR #10) |

v2 vs v3 in one line: **v3 is v2's recipe with double the data and 1.5× the
epochs** — same architecture, same labeler (Stockfish depth 10), same
inference cost, ~+500 Elo h2h.

The v1 net (600k depth-8 positions, crossover ~1400 vs Stockfish anchors)
predates this scheme and exists only in git history
(`git show 6d690d7:nets/salmon_nnue.npz`, pre-#8 `main`).

Strength numbers are ballpark: 8 games/anchor and 10-game h2h on one idle
16-core machine; both engines are deterministic, so h2h games can repeat.
Elo is relative to Stockfish's `UCI_Elo` scale. Classical Salmon measures
~2000 on the same scale.
