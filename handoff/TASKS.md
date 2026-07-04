# Tasks, in priority order

Work top to bottom. Each task is gated by the one before it.

## Task 1 — Rebase and land branch `claude/salmon-nnue-rust` (PR #9)

The branch is stacked on the pre-merge history of PR #7. PR #9's **base**
was already retargeted to `main` (2026-07-04); the **branch** still needs
the rebase. This exact command was trial-run on 2026-07-04 and completed
with **zero conflicts**:

```bash
git fetch origin main claude/salmon-nnue-rust
git checkout -B claude/salmon-nnue-rust origin/claude/salmon-nnue-rust
git rebase --onto origin/main 13b800e425875a948dba79785462b4c7f21375d6
```

(`13b800e` is the old base tip — the last pre-merge commit of
`claude/chess-engines-history-i7lqic`. The rebase carries the 7 Rust-branch
commits, `814786e..6765b6f`, onto `main`.)

Sanity-check the rebased tree (all three were verified in the trial):

- `HANDOFF.md` must NOT exist (main deleted it; it was the old v2 briefing).
- `nets/salmon_nnue.npz` must be byte-identical to `main`'s (the v2 net).
- `git diff --stat origin/main HEAD` ≈ 16 files, +1262 — Rust work only.

Then rebuild, test, push:

```bash
(cd rust && maturin build --release) && pip install --force-reinstall rust/target/wheels/salmon_nnue_rs-*.whl
for t in test_perft test_eval test_minimax test_nnue test_mcts test_rust_engine test_uci; do python3 $t.py || break; done
git push -u origin claude/salmon-nnue-rust --force-with-lease
```

## Task 2 — Remeasure strength with the v2 net, update PR #9, undraft

PR #9's strength table was measured with the **v1** net. The rebase gives
the Rust engine the v2 net for free. Run the full protocol in
`MEASUREMENTS.md` (gauntlet sweep + h2h, Rust and Python backends). Since
v2+Python already sits at ~1871, include higher anchors:
`GAUNTLET_ANCHORS="1500,1700,1900,2000"`.

Then edit PR #9's description:

- Replace the strength table with the v2-net numbers (keep the v1 rows for
  history if you like).
- Remove/rewrite the stale "Stacked on `claude/chess-engines-history-i7lqic`"
  blockquote and tick the "Retarget to `main`" checkbox (base retarget done
  2026-07-04; your rebase completes it).
- If the extra sims now buy Elo, say how much; if they still don't, say so
  honestly and keep the throughput framing (it's still what unblocks
  training iteration and quantization).

Undraft the PR. **Do not merge it** — the human reviews.

## Task 3 — v3 net (start only after Task 2's numbers are in)

New branch off the landed Rust result. The recipe that produced v2
(`nnue/training/`, resumable, `GEN_*` env — see `CLAUDE.md`):

```bash
GEN_SHARDS=80 GEN_SHARD_SIZE=50000 GEN_SF_DEPTH=10 python3 -m nnue.training.gen_data
python3 -m nnue.training.train --epochs 20 --out nets/salmon_nnue.npz
```

Levers for v3, roughly by expected value:

1. **More data / deeper labels** — scale `GEN_SHARDS` (v2 = 80 × 50k = 4M
   @ depth 10; try 8–10M and/or depth 12). Expect very roughly 300–500
   labels/s aggregate on 16 cores at depth 10; budget wall-time first.
2. **More epochs** — v2 stopped at 20 with loss still falling; try 30
   (`train.py` auto-selects CUDA if you have it; export is device-independent).
3. **Bigger net** — accumulator 256 → 384/512 in `nnue/network.py` +
   `nnue/training/train.py` + `rust/src/net.rs` (all three must agree; the
   `.npz` carries shapes). Costs sims/s — re-run `bench_mcts.py` and let the
   gauntlet arbitrate the tradeoff.

Gate: v3 is committed **only** if it beats v2 under the `MEASUREMENTS.md`
protocol (h2h + anchor sweep, numbers in the commit message). Keep a v2
backup copy locally for A/B via `SALMON_NNUE_NET=...` — don't commit the
backup.

Judgement call from last time: if a new policy head seems to hurt, flip
`PRIOR_MODE` in `nnue/mcts.py` (mirrored by a constant in `rust/src/mcts.rs`)
to `"heuristic"` and re-measure — no retraining needed.

## Task 4 — If time remains (see also `CLAUDE.md` roadmap)

- **int8/SIMD quantized inference** in `rust/src/net.rs` — the accumulator
  matmul dominates; quantization is the next order of magnitude and is now
  cleanly the bottleneck.
- **MCTS tree reuse across moves** (both backends, or Rust-only with the
  Python fallback unchanged) — gated by `test_mcts.py`/`test_rust_engine.py`
  parity plus a measured h2h improvement.
- **Underpromotion policy cells** (promotions currently share one from-to
  cell) — needs net + training + Rust changes together.
- Classical Salmon: king safety / pawn structure / mobility eval terms;
  read terminals off the move list (biggest cheap speedup, see roadmap).

## Housekeeping when the mission ends

- Branch `claude/salmon-nnue-v2` is merged (PR #8) — the human can delete it.
- This `handoff/` folder lives only on `claude/chess-engines-history-i7lqic`;
  it should never be merged. Leave a summary of what you measured/changed as
  PR comments or commit messages, not here.
