#Gauntlet harness: estimate SalmonChess's playing strength by playing it against
#Stockfish pinned to fixed UCI_Elo anchors over UCI, then computing a performance
#rating.
#
#Games run in parallel across worker processes (default: one per CPU core). Each
#game result is appended (under a lock) to a single JSONL file, so the run is
#resumable across restarts: rerun `python3 gauntlet.py` and it continues with
#only the not-yet-played games. Run `python3 gauntlet.py report` at any time to
#summarize results so far.
#
#Salmon uses a FIXED DEPTH by default so its strength is independent of CPU
#contention when many workers saturate the cores (a time budget would search
#fewer nodes under load and bias the estimate low). Set GAUNTLET_SALMON_TIME to
#use a time control instead.
#
#Config via env vars:
#  GAUNTLET_ANCHORS="1320,1500,1700,1900"  GAUNTLET_GAMES=25
#  GAUNTLET_WORKERS=<cpu count>            GAUNTLET_MAX_PLIES=200
#  GAUNTLET_SALMON_DEPTH=6                 GAUNTLET_SALMON_TIME=<seconds, overrides depth>
#  GAUNTLET_SF_TIME=0.3

import json
import math
import os
import queue
import sys
import multiprocessing as mp

import chess
import chess.engine

HERE = os.path.dirname(os.path.abspath(__file__))
SALMON = [sys.executable, os.path.join(HERE, "Salmon.py")]
STOCKFISH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
RESULTS = os.path.join(HERE, "gauntlet_results.jsonl")

ANCHORS = [int(x) for x in os.environ.get("GAUNTLET_ANCHORS", "1320,1500,1700,1900").split(",")]
GAMES_PER_ANCHOR = int(os.environ.get("GAUNTLET_GAMES", "25"))
WORKERS = int(os.environ.get("GAUNTLET_WORKERS", str(os.cpu_count() or 1)))
MAX_PLIES = int(os.environ.get("GAUNTLET_MAX_PLIES", "200"))
SF_TIME = float(os.environ.get("GAUNTLET_SF_TIME", "0.3"))
SALMON_DEPTH = int(os.environ.get("GAUNTLET_SALMON_DEPTH", "6"))
SALMON_TIME = os.environ.get("GAUNTLET_SALMON_TIME")        #if set, overrides depth

_MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9}


def _salmon_limit():
    if SALMON_TIME is not None:
        return chess.engine.Limit(time=float(SALMON_TIME))
    return chess.engine.Limit(depth=SALMON_DEPTH)


#Adjudicate a game that hit the ply cap by material balance (>= 4 pawns wins).
def _adjudicate(board):
    bal = 0
    for piece in board.piece_map().values():
        v = _MATERIAL.get(piece.piece_type, 0)
        bal += v if piece.color == chess.WHITE else -v
    if bal >= 4:
        return chess.WHITE
    if bal <= -4:
        return chess.BLACK
    return None


#Play one game; return (salmon_score in {0,0.5,1}, plies, termination_str).
def play_game(salmon, stockfish, salmon_white, salmon_limit, sf_limit):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        engine = salmon if (board.turn == chess.WHITE) == salmon_white else stockfish
        limit = salmon_limit if engine is salmon else sf_limit
        result = engine.play(board, limit)
        if result.move is None:
            break
        board.push(result.move)

    if board.is_game_over(claim_draw=True):
        winner = board.outcome(claim_draw=True).winner
        term = "rules"
    else:
        winner = _adjudicate(board)
        term = "adjudicated"

    if winner is None:
        score = 0.5
    else:
        score = 1.0 if (winner == chess.WHITE) == salmon_white else 0.0
    return score, board.ply(), term


#Worker process: drain tasks from the queue, append each result under the lock.
def _worker(task_q, lock):
    salmon = chess.engine.SimpleEngine.popen_uci(SALMON)
    stockfish = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    salmon_limit = _salmon_limit()
    sf_limit = chess.engine.Limit(time=SF_TIME)
    try:
        while True:
            try:
                anchor, gidx = task_q.get_nowait()
            except queue.Empty:
                break
            stockfish.configure({"UCI_LimitStrength": True, "UCI_Elo": anchor})
            salmon_white = (gidx % 2 == 0)
            score, plies, term = play_game(salmon, stockfish, salmon_white,
                                           salmon_limit, sf_limit)
            line = json.dumps({"anchor": anchor, "salmon_white": salmon_white,
                               "score": score, "plies": plies, "term": term})
            with lock:
                with open(RESULTS, "a") as f:
                    f.write(line + "\n")
            print(f"  vs {anchor}  game {gidx + 1}/{GAMES_PER_ANCHOR}  "
                  f"salmon={'W' if salmon_white else 'B'}  score={score}  "
                  f"{plies}p {term}", flush=True)
    finally:
        salmon.quit()
        stockfish.quit()


def _completed_counts():
    counts = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as f:
            for line in f:
                line = line.strip()
                if line:
                    a = json.loads(line)["anchor"]
                    counts[a] = counts.get(a, 0) + 1
    return counts


def run():
    control = f"time {SALMON_TIME}s/move" if SALMON_TIME is not None else f"depth {SALMON_DEPTH}"
    done = _completed_counts()
    tasks = []
    for anchor in ANCHORS:
        for gidx in range(done.get(anchor, 0), GAMES_PER_ANCHOR):
            tasks.append((anchor, gidx))

    n_workers = max(1, min(WORKERS, len(tasks)))
    print(f"Salmon ({control}) vs Stockfish @ UCI_Elo {ANCHORS}, "
          f"{GAMES_PER_ANCHOR} games/anchor (SF {SF_TIME}s/move, cap {MAX_PLIES} plies)")
    print(f"{len(tasks)} games to play across {n_workers} workers "
          f"({sum(done.values())} already done)")
    if not tasks:
        report()
        return

    task_q = mp.Queue()
    for t in tasks:
        task_q.put(t)
    lock = mp.Lock()
    procs = [mp.Process(target=_worker, args=(task_q, lock)) for _ in range(n_workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    report()


#Performance rating from a score rate p against opponents rated R:
#  perf = R + 400 * log10(p / (1 - p)), clamped for sweeps of 0% / 100%.
def _perf(p, r):
    p = min(max(p, 0.01), 0.99)
    return r + 400 * math.log10(p / (1 - p))


def report():
    if not os.path.exists(RESULTS):
        print("No results yet.")
        return
    by_anchor = {}
    for line in open(RESULTS):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_anchor.setdefault(rec["anchor"], []).append(rec["score"])

    print("\n=== Gauntlet results ===")
    perfs = []
    total_games = 0
    for anchor in sorted(by_anchor):
        scores = by_anchor[anchor]
        n = len(scores)
        total_games += n
        s = sum(scores)
        p = s / n
        perf = _perf(p, anchor)
        se = math.sqrt(max(p * (1 - p), 1e-9) / n)
        lo, hi = _perf(p - 1.96 * se, anchor), _perf(p + 1.96 * se, anchor)
        perfs.append(perf)
        print(f"vs Stockfish {anchor}:  {s:.1f}/{n}  ({100 * p:.0f}%)  "
              f"-> perf ~{perf:.0f} Elo  (95% CI {lo:.0f}..{hi:.0f})")

    if perfs:
        print(f"\nOverall estimate (mean of per-anchor performance, {total_games} games): "
              f"~{sum(perfs) / len(perfs):.0f} Elo")
        print("Caveat: Stockfish UCI_Elo anchors are approximate and time-control "
              "dependent; treat this as a ballpark.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        report()
    else:
        run()
