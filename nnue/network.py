#NNUE-style evaluation network for SalmonNNUE: feature encoding, NumPy
#inference, and the efficiently-updatable accumulator.
#
#Feature encoding (768 per perspective): for perspective color c, a piece of
#color pc and type p on square sq (0..63, a1=0) activates
#    idx = side*384 + piece*64 + rel_sq
#where rel_sq mirrors vertically (sq ^ 56) for the Black perspective, side is
#0 for own pieces / 1 for enemy pieces, and piece order is P,N,B,R,Q,K.
#
#Architecture (float32 at inference):
#    accumulator: [768,256] shared weights, one accumulator per perspective
#    concat stm-first -> 512, clipped ReLU to [0,1]
#    L1 [512,32] clipped ReLU -> value head [32,1] (sigmoid)
#    policy head [512,4096] from the concat (stm-relative from*64+to logits)

import copy

import numpy as np
import cozy_chess as cc

_PIECE_IDX = {cc.Piece.Pawn: 0, cc.Piece.Knight: 1, cc.Piece.Bishop: 2,
              cc.Piece.Rook: 3, cc.Piece.Queen: 4, cc.Piece.King: 5}

FEATURES = 768
ACC_SIZE = 256
L1_SIZE = 32
POLICY_SIZE = 64 * 64

_WHITE, _BLACK = 0, 1           #accumulator row per perspective


def _pidx(color):
    return _WHITE if color == cc.Color.White else _BLACK


def feature_index(perspective, piece_color, piece, square):
    rel_sq = square if perspective == cc.Color.White else square ^ 56
    side = 0 if piece_color == perspective else 1
    return side * 384 + _PIECE_IDX[piece] * 64 + rel_sq


def active_features(board, perspective):
    feats = []
    for color in (cc.Color.White, cc.Color.Black):
        for piece in cc.Piece.ALL:
            for sq in board.colored_pieces(color, piece):
                feats.append(feature_index(perspective, color, piece, int(sq)))
    return feats


#Policy index of a move, relative to the side to move: from*64 + to on the
#stm-mirrored board, with cozy's king-takes-rook castling normalized to the
#standard g-/c-file king destination (matching common.to_uci). All promotions
#of a pawn share the same from-to cell.
def policy_index(board, move):
    fs, ts = int(move.from_square), int(move.to_square)
    if (board.piece_on(move.from_square) == cc.Piece.King
            and board.piece_on(move.to_square) == cc.Piece.Rook
            and board.color_on(move.to_square) == board.color_on(move.from_square)):
        rank = fs - (fs % 8)
        ts = rank + (6 if (ts % 8) == 7 else 2)
    if board.side_to_move() == cc.Color.Black:
        fs ^= 56
        ts ^= 56
    return fs * 64 + ts


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Net:
    """NumPy weight container + forward pass. Weights stored float32."""

    KEYS = ("ft.weight", "ft.bias", "l1.weight", "l1.bias",
            "value.weight", "value.bias", "policy.weight", "policy.bias")

    def __init__(self, ft_w, ft_b, l1_w, l1_b, v_w, v_b, p_w, p_b):
        self.ft_w = np.ascontiguousarray(ft_w, dtype=np.float32)   #[768,256]
        self.ft_b = np.ascontiguousarray(ft_b, dtype=np.float32)   #[256]
        self.l1_w = np.ascontiguousarray(l1_w, dtype=np.float32)   #[512,32]
        self.l1_b = np.ascontiguousarray(l1_b, dtype=np.float32)   #[32]
        self.v_w = np.ascontiguousarray(v_w, dtype=np.float32)     #[32]
        self.v_b = float(v_b)
        self.p_w = np.ascontiguousarray(p_w, dtype=np.float32)     #[512,4096]
        self.p_b = np.ascontiguousarray(p_b, dtype=np.float32)     #[4096]

    #Forward pass from an accumulator: returns (value, policy_logits) where
    #value is in (0,1) from the side to move's perspective.
    def value_policy(self, acc, stm):
        me = _pidx(stm)
        x = np.concatenate([acc.vals[me], acc.vals[1 - me]])
        np.clip(x, 0.0, 1.0, out=x)
        h = x @ self.l1_w + self.l1_b
        np.clip(h, 0.0, 1.0, out=h)
        value = _sigmoid(float(h @ self.v_w) + self.v_b)
        logits = x @ self.p_w + self.p_b
        return value, logits

    def save(self, path):
        np.savez_compressed(
            path,
            **{"ft.weight": self.ft_w.astype(np.float16),
               "ft.bias": self.ft_b.astype(np.float16),
               "l1.weight": self.l1_w.astype(np.float16),
               "l1.bias": self.l1_b.astype(np.float16),
               "value.weight": self.v_w.astype(np.float16),
               "value.bias": np.array([self.v_b], dtype=np.float16),
               "policy.weight": self.p_w.astype(np.float16),
               "policy.bias": self.p_b.astype(np.float16),
               "meta.version": np.array([1])})

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            version = int(z["meta.version"][0])
            if version != 1:
                raise ValueError(f"unsupported net version: {version}")
            return cls(z["ft.weight"], z["ft.bias"], z["l1.weight"], z["l1.bias"],
                       z["value.weight"], float(z["value.bias"][0]),
                       z["policy.weight"], z["policy.bias"])


#Small random net with a fixed seed, for tests and as a bootable fallback
#before a trained net exists.
def make_random_net(seed=0):
    rng = np.random.default_rng(seed)
    return Net(
        rng.normal(0, 0.05, (FEATURES, ACC_SIZE)),
        rng.normal(0, 0.05, ACC_SIZE),
        rng.normal(0, 0.1, (2 * ACC_SIZE, L1_SIZE)),
        rng.normal(0, 0.1, L1_SIZE),
        rng.normal(0, 0.1, L1_SIZE),
        0.0,
        rng.normal(0, 0.02, (2 * ACC_SIZE, POLICY_SIZE)),
        np.zeros(POLICY_SIZE))


class Accumulator:
    """Per-perspective first-layer activations, updated incrementally.

    vals is float32 [2,256] indexed by perspective (White=0, Black=1). The
    incremental path handles normal moves, captures, en passant, and
    promotions; castling (cozy's king-takes-rook, two pieces move) falls back
    to a full recompute of the child position.
    """

    __slots__ = ("vals",)

    def __init__(self, vals):
        self.vals = vals

    @classmethod
    def from_board(cls, net, board):
        vals = np.empty((2, ACC_SIZE), dtype=np.float32)
        for perspective in (cc.Color.White, cc.Color.Black):
            row = net.ft_b.copy()
            for f in active_features(board, perspective):
                row += net.ft_w[f]
            vals[_pidx(perspective)] = row
        return cls(vals)

    #Accumulator of the position after `move`, given the pre-move board.
    #board_after (the already-played child board) is required only for the
    #castling fallback; passing it avoids a redundant board copy.
    def play(self, net, board_before, move, board_after=None):
        piece = board_before.piece_on(move.from_square)
        mover = board_before.color_on(move.from_square)

        #Castling moves two pieces of one color; recompute rather than delta.
        victim_on_to = board_before.piece_on(move.to_square)
        if piece == cc.Piece.King and victim_on_to == cc.Piece.Rook \
                and board_before.color_on(move.to_square) == mover:
            if board_after is None:
                board_after = copy.copy(board_before)
                board_after.play(move)
            return Accumulator.from_board(net, board_after)

        fs, ts = int(move.from_square), int(move.to_square)
        final_piece = move.promotion if move.promotion is not None else piece
        is_ep = (piece == cc.Piece.Pawn and victim_on_to is None
                 and (fs % 8) != (ts % 8))

        vals = self.vals.copy()
        for perspective in (cc.Color.White, cc.Color.Black):
            row = vals[_pidx(perspective)]
            row -= net.ft_w[feature_index(perspective, mover, piece, fs)]
            row += net.ft_w[feature_index(perspective, mover, final_piece, ts)]
            if victim_on_to is not None:
                victim_color = board_before.color_on(move.to_square)
                row -= net.ft_w[feature_index(perspective, victim_color,
                                              victim_on_to, ts)]
            elif is_ep:
                cap_sq = ts - 8 if mover == cc.Color.White else ts + 8
                enemy = cc.Color.Black if mover == cc.Color.White else cc.Color.White
                row -= net.ft_w[feature_index(perspective, enemy,
                                              cc.Piece.Pawn, cap_sq)]
        return Accumulator(vals)
