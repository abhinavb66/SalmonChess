# HANDOFF — SalmonChess: land the Rust engine, then chase strength

You are a local agent picking up SalmonChess on a real machine (more cores
than the 4-core cloud container this was built in, and able to install a
Rust toolchain). Read `CLAUDE.md` first for project conventions; this folder
is the mission briefing.

- `README.md` (this file) — state of the world, mission, setup.
- `TASKS.md` — step-by-step instructions per task, in priority order.
- `MEASUREMENTS.md` — every baseline number, the measurement protocol, and
  the gotchas that have already burned us once.

## State of the world (2026-07-04)

Two UCI engines over cozy-chess boards, both playable in the web UI:

- **Salmon** — classical alpha-beta. **~2000 Elo** vs Stockfish UCI_Elo
  anchors. Stable; not the focus.
- **SalmonNNUE** — NNUE-style value+policy net driving a PUCT MCTS.

### What's merged (on `main` @ `de4c655`)

- The whole SalmonNNUE stack: net + MCTS + training pipeline + gauntlet +
  per-side UI (PR #7, merged).
- The **v2 net** (`nets/salmon_nnue.npz`): 4M Stockfish depth-10 positions,
  loss 2.39 → 1.29. Measured **~1871 Elo** (crossover ~1700) with the pure
  Python engine at ~2.3k sims/s (PR #8, merged).

### What's in flight (draft PR #9 — your first job)

Branch `claude/salmon-nnue-rust` rewrites the hot loop (NNUE inference +
PUCT MCTS) in **Rust via PyO3**, mirroring the Python engine exactly:
**~2,300 → ~59,000 sims/s (~26×)**. Pure-Python fallback kept; repo works
without a Rust toolchain. 45 tests across 7 suites, including
value/prior/hash/behavior parity between the two backends. The first commit
on the branch is a Python-side `nnue/mcts.py` bug fix (interior-node visits
double-counted) that only exists there — it lands with the PR.

Two things are stale about PR #9, and they are the mission:

1. **It's stacked on pre-merge history.** The base was retargeted to `main`
   on 2026-07-04, but the branch itself still needs a rebase onto `main`.
   Already trial-rebased: **zero conflicts**, and the rebased tree is
   exactly `main` + the Rust work (16 files, +1262). See `TASKS.md` Task 1.
2. **Its strength numbers used the v1 net** (the branch predates PR #8).
   With v1, the 26× throughput bought **no Elo** — the weak net saturates
   MCTS below ~2.5k sims (see `MEASUREMENTS.md`). The open question you get
   to answer: **does Rust throughput + the v2 net buy real Elo?**

## Mission (priority order)

1. **Land PR #9** — rebase onto `main`, rebuild, all tests green, push.
2. **Remeasure with the v2 net** — Rust vs Python backend, gauntlet + h2h;
   put the numbers in PR #9's description and undraft it. (Never merge —
   the human reviews.)
3. **v3 net** — the throughput exists to iterate on training now; bigger
   data / deeper labels / bigger net, each gated by measurement.
4. **Roadmap extras** if time remains (`TASKS.md` Task 4, `CLAUDE.md`
   roadmap): int8/SIMD quantized inference in Rust, MCTS tree reuse,
   classical-engine eval features.

## Setup

```bash
git clone <repo-url> && cd SalmonChess
git checkout claude/salmon-nnue-rust        # the work branch (PR #9)

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cpu
# torch is training-only; play-time inference is NumPy (or Rust).

# Stockfish (labeler + gauntlet opponent): apt install stockfish / brew install stockfish.
# If not at /usr/games/stockfish: export STOCKFISH_PATH=/path/to/stockfish

# Rust backend (optional for the repo, required for this mission):
# install rustup if needed, then:
pip install maturin
(cd rust && maturin build --release) && pip install rust/target/wheels/salmon_nnue_rs-*.whl
python3 -c "from nnue import rust_engine; print(rust_engine.AVAILABLE)"   # must print True

# Verify before doing anything (7 suites, all must pass):
for t in test_perft test_eval test_minimax test_nnue test_mcts test_rust_engine test_uci; do python3 $t.py || break; done
python3 bench_mcts.py    # note this machine's Python vs Rust sims/s
```

`SALMON_NNUE_ENGINE=python` forces the pure-Python fallback (that's how you
A/B the backends under identical conditions).

## Ground rules

- **Branches**: PR #9 work goes on `claude/salmon-nnue-rust` only
  (force-with-lease is fine for the rebase). v3-net work starts a **new
  branch off the landed result** — don't pile it into PR #9. Never push to
  `main`. Never merge PRs.
- **This `handoff/` folder** lives only on branch
  `claude/chess-engines-history-i7lqic` — it's a briefing, not product code.
  Don't merge it anywhere; it dies with its branch when the mission is done.
- **Commit gates**: all test suites green before any commit; nets are
  committed only with measured improvement, numbers in the commit message
  (see `MEASUREMENTS.md` for the protocol); never commit `data/`, result
  `.jsonl` files, or backup net copies.
- **Measurement discipline**: `GAUNTLET_WORKERS=1` for any time-based
  SalmonNNUE run, and never train while measuring. This has produced a 0/18
  false result once already — details in `MEASUREMENTS.md`.
