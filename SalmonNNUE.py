#UCI entry point for SalmonNNUE: an NNUE-style value+policy network driving a
#Monte Carlo Tree Search (see nnue/network.py and nnue/mcts.py).
#
#The network is loaded from $SALMON_NNUE_NET, else the current default version
#below (see nets/README.md for the version history), else a fixed-seed random
#net (so the engine boots before a trained net exists).

import functools
import os
import sys

import common
from nnue import mcts, network, rust_engine

VERSION = "0.1"

_DEFAULT_NET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "nets", "salmon_nnue_v3.npz")


def _load_net():
    path = os.environ.get("SALMON_NNUE_NET") or _DEFAULT_NET
    if os.path.exists(path):
        return network.Net.load(path)
    sys.stdout.reconfigure(line_buffering=True)
    print(f"info string no net at {path}, using random weights")
    return network.make_random_net(0)


if __name__ == "__main__":
    net = _load_net()
    #Prefer the Rust engine when built; SALMON_NNUE_ENGINE=python forces the
    #pure-Python MCTS (used to keep the fallback path tested).
    if rust_engine.AVAILABLE and os.environ.get("SALMON_NNUE_ENGINE") != "python":
        bestmove_fn = functools.partial(rust_engine.bestMove,
                                        engine=rust_engine.make_engine(net))
    else:
        bestmove_fn = functools.partial(mcts.bestMove, net=net)
    common.run_uci_loop("SalmonNNUE" + VERSION, "Abhinav Brahmarouthu", bestmove_fn)
