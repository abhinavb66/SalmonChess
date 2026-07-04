# Measurement protocol and baselines

Strength claims in this project are only trusted when produced under this
protocol. Everything below was learned the hard way; don't improvise.

## The two gotchas (both have produced false results already)

1. **CPU contention corrupts time-based measurement.** SalmonNNUE's strength
   at `time = X s/move` is a function of simulations completed; concurrent
   engine games starve each other. A 2-worker gauntlet once measured **0/18**
   where the true single-worker score was 88%. Therefore:
   **always `GAUNTLET_WORKERS=1`** for SalmonNNUE time-based runs, and never
   run training (or anything CPU-heavy) while measuring.
2. **The gauntlet is resumable via `gauntlet_results_SalmonNNUE.jsonl`** —
   stale rows from a previous net/backend get counted as done. `rm -f` the
   results file before any sweep that changes the engine under test.

Small samples: 4–8 games per anchor gives ballpark Elo, not tight CIs. The
v1 net measured 88/12/0 (@1320/1500/1700) in one run and 81/19/6 in another
— same engine. Compare configurations, don't over-read single percentages.

## Protocol

All at 1s/move unless stated. Run from the repo root, venv active.

```bash
# Head-to-head vs the classical Salmon (~2000 Elo), alternating colors:
GAUNTLET_SALMON_TIME=1.0 H2H_A=SalmonNNUE.py H2H_B=Salmon.py python3 gauntlet.py h2h 10

# Stockfish UCI_Elo anchor sweep (SINGLE WORKER — see gotcha 1):
rm -f gauntlet_results_SalmonNNUE.jsonl
GAUNTLET_ENGINE=SalmonNNUE.py GAUNTLET_ANCHORS="1500,1700,1900,2000" GAUNTLET_GAMES=8 \
  GAUNTLET_SALMON_TIME=1.0 GAUNTLET_SF_TIME=0.3 GAUNTLET_WORKERS=1 python3 gauntlet.py

# A/B the Python backend under identical conditions:
SALMON_NNUE_ENGINE=python <same command>

# A/B an alternative net:
SALMON_NNUE_NET=nets/<other>.npz <same command>
```

Wall-clock budget: a 1s/move game runs ~100–180 plies ≈ 2–4 min, so an
8-game × 4-anchor sweep is ~1.5–2.5 h and an h2h 10 is ~30–60 min. Plan
sweeps accordingly (they resume if interrupted — but see gotcha 2).

## Baselines (all 1s/move, single worker)

Stockfish UCI_Elo anchor sweeps:

| Config | sims/s (bench_mcts, 4-core box) | @1320 | @1500 | @1700 | Est. Elo |
|---|---|---|---|---|---|
| v1 net, Python (PR #7) | ~2.5k | 88% | 12% | 0% | crossover ~1400 |
| v1 net, Python (PR #9 rerun) | ~2.3k | 81% | 19% | 6% | ~1350 |
| v1 net, **Rust** (PR #9) | ~59k | 55% | 25% | 15% | ~1354 |
| **v2 net, Python** (PR #8) | ~2.3k | 100% | 88% | 44% | **~1871** (crossover ~1700) |
| **v2 net, Rust** | ? | — | — | — | **← the missing measurement (Task 2)** |

Head-to-head, SalmonNNUE vs Salmon (10 games):

| Config | Score |
|---|---|
| v1 net, Python | 1.5/10 (one run measured 0.0/10) |
| v2 net, Python | 2.5/10 |
| v2 net, Rust | ? — measure |

Reference points: Salmon ≈ 2000 on the same anchor scale; your machine's
sims/s will differ — record `bench_mcts.py` output alongside any numbers
you report.

## The saturation finding (why v1+Rust gained nothing)

With the v1 net, 26× more simulations bought **zero measurable Elo**
(~1350 → ~1354): the weak value/policy net converges on its move well under
~2.5k sims, so sims 2.5k–59k changed almost nothing except making losses
longer (~40–95 plies → ~125–190 plies vs Salmon). Interpretation that
guided the roadmap: **the ceiling is the net, not the search** — throughput
is the enabler for training iteration and bigger/quantized nets, and pays
Elo only once the net is strong enough to exploit depth. Whether v2 is
already strong enough is exactly the Task 2 question.

## Environment-variable quick reference

| Var | Meaning |
|---|---|
| `GAUNTLET_ENGINE` | engine under test (`Salmon.py` / `SalmonNNUE.py`) |
| `GAUNTLET_ANCHORS` / `GAUNTLET_GAMES` | UCI_Elo anchors / games per anchor |
| `GAUNTLET_SALMON_TIME` / `GAUNTLET_SF_TIME` | s/move for our engine / Stockfish |
| `GAUNTLET_WORKERS` | parallel games — **1** for SalmonNNUE time-based runs |
| `H2H_A` / `H2H_B` | engines for `gauntlet.py h2h N` |
| `SALMON_NNUE_NET` | net path override (default `nets/salmon_nnue.npz`) |
| `SALMON_NNUE_ENGINE` | `python` forces the pure-Python MCTS (else Rust when built) |
| `STOCKFISH_PATH` | Stockfish binary (default `/usr/games/stockfish`) |
| `GEN_SHARDS` / `GEN_SHARD_SIZE` / `GEN_SF_DEPTH` | data-gen size and label depth |
| `PRIOR_MODE` / `CPUCT` / `FPU_REDUCTION` | search knobs in `nnue/mcts.py` (mirrored in `rust/src/mcts.rs`) |
