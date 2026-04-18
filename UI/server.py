"""Flask server for the SalmonChess local web UI."""

import os
import sys
import threading

import chess
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import eval as salmon_eval  # noqa: E402

from uci_client import UCIClient  # noqa: E402


app = Flask(__name__)
_lock = threading.Lock()

state = {
    "board": chess.Board(),
    "mode": "human_vs_engine",   # human_vs_engine | human_vs_human | engine_vs_engine
    "human_color": chess.WHITE,
    "history": [],               # UCI move strings
    "engine_white": None,        # UCIClient or None
    "engine_black": None,        # UCIClient or None
    "movetime_ms": 1000,
}


def _shutdown_engines():
    for key in ("engine_white", "engine_black"):
        client = state.get(key)
        if client is not None:
            try:
                client.quit()
            except Exception:
                pass
            state[key] = None


def _start_engines():
    """Spawn whichever engine subprocesses the current mode needs."""
    _shutdown_engines()
    mode = state["mode"]
    if mode == "human_vs_engine":
        engine_color = chess.BLACK if state["human_color"] == chess.WHITE else chess.WHITE
        if engine_color == chess.WHITE:
            state["engine_white"] = UCIClient()
        else:
            state["engine_black"] = UCIClient()
    elif mode == "engine_vs_engine":
        state["engine_white"] = UCIClient()
        state["engine_black"] = UCIClient()


def _engine_for_turn():
    board = state["board"]
    if board.turn == chess.WHITE:
        return state["engine_white"]
    return state["engine_black"]


def _is_human_turn():
    mode = state["mode"]
    board = state["board"]
    if mode == "human_vs_human":
        return True
    if mode == "engine_vs_engine":
        return False
    return board.turn == state["human_color"]


def _result_string(board):
    if not board.is_game_over():
        return None
    outcome = board.outcome()
    if outcome is None:
        return None
    return outcome.result()


def _state_payload():
    board = state["board"]
    last_move = state["history"][-1] if state["history"] else None
    try:
        eval_cp = salmon_eval.eval_board(board)
    except Exception:
        eval_cp = 0
    return {
        "fen": board.fen(),
        "turn": "w" if board.turn == chess.WHITE else "b",
        "mode": state["mode"],
        "human_color": "w" if state["human_color"] == chess.WHITE else "b",
        "is_check": board.is_check(),
        "is_game_over": board.is_game_over(),
        "result": _result_string(board),
        "last_move": last_move,
        "history": list(state["history"]),
        "eval_cp": eval_cp,
        "is_human_turn": _is_human_turn(),
        "legal_moves": [m.uci() for m in board.legal_moves],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state", methods=["GET"])
def api_state():
    with _lock:
        return jsonify(_state_payload())


@app.route("/api/new_game", methods=["POST"])
def api_new_game():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "human_vs_engine")
    human_color_str = data.get("human_color", "w")
    fen = data.get("fen")
    movetime_ms = int(data.get("movetime_ms", 1000))

    if mode not in ("human_vs_engine", "human_vs_human", "engine_vs_engine"):
        return jsonify({"error": f"unknown mode: {mode}"}), 400

    with _lock:
        if fen:
            try:
                state["board"] = chess.Board(fen)
            except ValueError as e:
                return jsonify({"error": f"invalid FEN: {e}"}), 400
        else:
            state["board"] = chess.Board()
        state["mode"] = mode
        state["human_color"] = chess.WHITE if human_color_str == "w" else chess.BLACK
        state["history"] = []
        state["movetime_ms"] = movetime_ms
        try:
            _start_engines()
        except Exception as e:
            return jsonify({"error": f"failed to start engine: {e}"}), 500
        return jsonify(_state_payload())


@app.route("/api/move", methods=["POST"])
def api_move():
    data = request.get_json(silent=True) or {}
    uci = data.get("uci")
    if not uci:
        return jsonify({"error": "missing 'uci'"}), 400

    with _lock:
        board = state["board"]
        if state["mode"] == "engine_vs_engine":
            return jsonify({"error": "human moves not allowed in engine_vs_engine mode"}), 400
        if state["mode"] == "human_vs_engine" and board.turn != state["human_color"]:
            return jsonify({"error": "not your turn"}), 400

        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return jsonify({"error": f"invalid UCI: {uci}"}), 400

        # Auto-promote to queen if UI sent a pawn-to-last-rank move without promotion
        if move.promotion is None:
            piece = board.piece_at(move.from_square)
            if piece and piece.piece_type == chess.PAWN:
                to_rank = chess.square_rank(move.to_square)
                if to_rank == 0 or to_rank == 7:
                    move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

        if move not in board.legal_moves:
            return jsonify({"error": f"illegal move: {uci}"}), 400

        board.push(move)
        state["history"].append(move.uci())
        return jsonify(_state_payload())


@app.route("/api/engine_move", methods=["POST"])
def api_engine_move():
    with _lock:
        board = state["board"]
        if board.is_game_over():
            return jsonify(_state_payload())
        engine = _engine_for_turn()
        if engine is None:
            return jsonify({"error": "no engine configured for this side"}), 400
        try:
            uci = engine.bestmove(board, movetime_ms=state["movetime_ms"])
        except Exception as e:
            return jsonify({"error": f"engine error: {e}"}), 500
        if not uci or uci == "0000":
            return jsonify({"error": "engine returned no move"}), 500
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return jsonify({"error": f"engine returned invalid UCI: {uci}"}), 500
        if move not in board.legal_moves:
            return jsonify({"error": f"engine returned illegal move: {uci}"}), 500
        board.push(move)
        state["history"].append(move.uci())
        return jsonify(_state_payload())


@app.route("/api/undo", methods=["POST"])
def api_undo():
    with _lock:
        board = state["board"]
        if not state["history"]:
            return jsonify(_state_payload())
        # In HvE, pop twice so the human's previous move is back on the board for them.
        pops = 2 if state["mode"] == "human_vs_engine" and len(state["history"]) >= 2 else 1
        for _ in range(pops):
            if state["history"]:
                board.pop()
                state["history"].pop()
        return jsonify(_state_payload())


@app.route("/api/legal", methods=["GET"])
def api_legal():
    square_str = request.args.get("square", "")
    try:
        square = chess.parse_square(square_str)
    except ValueError:
        return jsonify({"error": f"bad square: {square_str}"}), 400
    with _lock:
        board = state["board"]
        targets = sorted({
            chess.square_name(m.to_square)
            for m in board.legal_moves
            if m.from_square == square
        })
        return jsonify({"from": square_str, "targets": targets})


if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    finally:
        _shutdown_engines()
