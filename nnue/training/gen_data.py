#Training-data generation for SalmonNNUE: sample diverse positions, label each
#with a Stockfish evaluation (value target) and best move (policy target), and
#write compact npz shards under data/.
#
#Parallel and resumable: each worker claims a shard by exclusively creating
#data/shard_NNNNN.tmp, writes it, then renames to .npz — rerunning skips
#finished shards (the same pattern gauntlet.py uses for results).
#
#Shard arrays (N = positions per shard):
#  sq    uint8[N,32]  piece squares, 255-padded
#  pc    uint8[N,32]  piece codes piece_idx*2+color (P,N,B,R,Q,K x white,black), 255-padded
#  stm   uint8[N]     side to move (0 white, 1 black)
#  score int16[N]     Stockfish eval, centipawns from stm's view, clamped +-3000
#  bf,bt uint8[N]     Stockfish best move from/to square (standard UCI squares)
#
#Config via env vars:
#  GEN_SHARDS=10  GEN_SHARD_SIZE=100000  GEN_WORKERS=<cpus>  GEN_SF_DEPTH=8
#  STOCKFISH_PATH=/usr/games/stockfish
#
#Run: python3 -m nnue.training.gen_data   (from the repo root)

import multiprocessing as mp
import os
import random
import sys

import numpy as np
import chess
import chess.engine
import chess.polyglot

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIR = os.path.join(REPO, "data")

STOCKFISH = os.environ.get("STOCKFISH_PATH", "/usr/games/stockfish")
SHARDS = int(os.environ.get("GEN_SHARDS", "10"))
SHARD_SIZE = int(os.environ.get("GEN_SHARD_SIZE", "100000"))
WORKERS = int(os.environ.get("GEN_WORKERS", str(os.cpu_count() or 1)))
SF_DEPTH = int(os.environ.get("GEN_SF_DEPTH", "8"))

SCORE_CLAMP = 3000
_PIECE_IDX = {chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
              chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5}


#Yield positions to label: a mix of short-random-opening + engine play-through
#(with random-move injection for diversity) and pure random walks.
def _positions(rng, engine):
    while True:
        if rng.random() < 0.6:
            #Random opening, then fast engine self-play with 10% random moves.
            board = chess.Board()
            for _ in range(rng.randint(6, 12)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            while not board.is_game_over() and board.ply() <= 80:
                if rng.random() < 0.10:
                    board.push(rng.choice(list(board.legal_moves)))
                else:
                    result = engine.play(board, chess.engine.Limit(time=0.01))
                    if result.move is None:
                        break
                    board.push(result.move)
                if board.ply() % 2 == 0 and not board.is_game_over():
                    yield board.copy(stack=False)
        else:
            #Random walk of 10-40 plies.
            board = chess.Board()
            for _ in range(rng.randint(10, 40)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(rng.choice(moves))
            if not board.is_game_over():
                yield board.copy(stack=False)


def _encode_position(board):
    sq = np.full(32, 255, dtype=np.uint8)
    pc = np.full(32, 255, dtype=np.uint8)
    for i, (square, piece) in enumerate(board.piece_map().items()):
        sq[i] = square
        pc[i] = _PIECE_IDX[piece.piece_type] * 2 + (0 if piece.color else 1)
    return sq, pc


def _make_shard(shard_idx, tmp_path):
    rng = random.Random(1000 + shard_idx)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
    engine.configure({"Threads": 1})
    n = SHARD_SIZE
    sq = np.empty((n, 32), dtype=np.uint8)
    pc = np.empty((n, 32), dtype=np.uint8)
    stm = np.empty(n, dtype=np.uint8)
    score = np.empty(n, dtype=np.int16)
    bf = np.empty(n, dtype=np.uint8)
    bt = np.empty(n, dtype=np.uint8)
    seen = set()
    count = 0
    try:
        for board in _positions(rng, engine):
            if board.is_check():        #in-check evals are noisy; skip
                continue
            key = chess.polyglot.zobrist_hash(board)
            if key in seen:
                continue
            seen.add(key)

            info = engine.analyse(board, chess.engine.Limit(depth=SF_DEPTH))
            pov = info["score"].pov(board.turn)
            if pov.is_mate():
                cp = SCORE_CLAMP if pov.mate() > 0 else -SCORE_CLAMP
            else:
                cp = max(-SCORE_CLAMP, min(SCORE_CLAMP, pov.score()))
            pv = info.get("pv")
            if not pv:
                continue

            sq[count], pc[count] = _encode_position(board)
            stm[count] = 0 if board.turn else 1
            score[count] = cp
            bf[count] = pv[0].from_square
            bt[count] = pv[0].to_square
            count += 1
            if count % 5000 == 0:
                print(f"  shard {shard_idx}: {count}/{n}", flush=True)
            if count >= n:
                break
    finally:
        engine.quit()

    np.savez_compressed(tmp_path, sq=sq[:count], pc=pc[:count], stm=stm[:count],
                        score=score[:count], bf=bf[:count], bt=bt[:count])


def _worker():
    for shard_idx in range(SHARDS):
        final = os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.npz")
        tmp = os.path.join(DATA_DIR, f"shard_{shard_idx:05d}.tmp")
        if os.path.exists(final):
            continue
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
        except FileExistsError:
            continue                    #another worker owns this shard
        print(f"worker {os.getpid()}: shard {shard_idx}", flush=True)
        tmp_npz = tmp + ".npz"          #np.savez appends .npz itself otherwise
        _make_shard(shard_idx, tmp_npz)
        os.replace(tmp_npz, final)
        os.remove(tmp)
        print(f"worker {os.getpid()}: shard {shard_idx} done", flush=True)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    #Clear stale claim files from a previous crashed run.
    for f in os.listdir(DATA_DIR):
        if f.endswith(".tmp") or f.endswith(".tmp.npz"):
            os.remove(os.path.join(DATA_DIR, f))
    done = sum(1 for f in os.listdir(DATA_DIR) if f.endswith(".npz"))
    print(f"{SHARDS} shards x {SHARD_SIZE} positions, depth {SF_DEPTH}, "
          f"{WORKERS} workers ({done} shards already done)")
    procs = [mp.Process(target=_worker) for _ in range(max(1, WORKERS))]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("data generation complete")


if __name__ == "__main__":
    main()
