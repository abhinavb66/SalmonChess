let board = null;
let currentState = null;
let engineLoopRunning = false;

const $status = $('#status');
const $evalNumber = $('#eval-number');
const $evalFill = $('#eval-fill');
const $moveList = $('#move-list');

function api(path, opts = {}) {
  return fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  }).then(async r => {
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `HTTP ${r.status}`);
    return data;
  });
}

function removeHighlights() {
  $('#board .square-55d63').removeClass('highlight-square highlight-target');
}

function highlightSquare(sq, cls) {
  $('#board .square-' + sq).addClass(cls);
}

function renderState(s) {
  currentState = s;
  if (board) board.position(s.fen, false);

  // Eval bar: clamp to ±1000 cp → 0..100%
  const clamped = Math.max(-1000, Math.min(1000, s.eval_cp));
  const pct = 50 + (clamped / 1000) * 50;
  $evalFill.css('width', pct + '%');
  const pawns = (s.eval_cp / 100).toFixed(2);
  $evalNumber.text((s.eval_cp >= 0 ? '+' : '') + pawns);

  // Move list (pair into full moves)
  $moveList.empty();
  for (let i = 0; i < s.history.length; i += 2) {
    const w = s.history[i];
    const b = s.history[i + 1] || '';
    $moveList.append(`<li>${w} ${b}</li>`);
  }

  // Status line
  let msg = '';
  if (s.is_game_over) {
    msg = `Game over: ${s.result || '?'}`;
  } else if (s.is_check) {
    msg = `${s.turn === 'w' ? 'White' : 'Black'} to move — check!`;
  } else {
    msg = `${s.turn === 'w' ? 'White' : 'Black'} to move`;
  }
  $status.text(msg);

  // Orient board based on human side in HvE
  if (s.mode === 'human_vs_engine') {
    const want = s.human_color === 'w' ? 'white' : 'black';
    if (board && board.orientation() !== want) board.orientation(want);
  }
}

async function refreshState() {
  const s = await api('/api/state');
  renderState(s);
  return s;
}

async function triggerEngineIfNeeded() {
  if (!currentState || currentState.is_game_over) return;
  const mode = currentState.mode;
  if (mode === 'human_vs_human') return;
  if (mode === 'human_vs_engine' && currentState.is_human_turn) return;

  // For engine_vs_engine, keep looping until game over
  if (engineLoopRunning) return;
  engineLoopRunning = true;
  try {
    while (currentState && !currentState.is_game_over) {
      if (mode === 'human_vs_engine' && currentState.is_human_turn) break;
      const s = await api('/api/engine_move', { method: 'POST' });
      renderState(s);
      if (mode !== 'engine_vs_engine') break;
      await new Promise(r => setTimeout(r, 350));
    }
  } catch (e) {
    $status.text('Engine error: ' + e.message);
  } finally {
    engineLoopRunning = false;
  }
}

function onDragStart(source, piece) {
  if (!currentState || currentState.is_game_over) return false;
  if (currentState.mode === 'engine_vs_engine') return false;
  if (currentState.mode === 'human_vs_engine' && !currentState.is_human_turn) return false;
  const isWhite = piece.startsWith('w');
  if (isWhite && currentState.turn !== 'w') return false;
  if (!isWhite && currentState.turn !== 'b') return false;

  // Highlight legal destinations
  api('/api/legal?square=' + source).then(data => {
    removeHighlights();
    highlightSquare(source, 'highlight-square');
    (data.targets || []).forEach(t => highlightSquare(t, 'highlight-target'));
  }).catch(() => {});
}

function onDrop(source, target) {
  removeHighlights();
  if (source === target) return 'snapback';
  const uci = source + target;
  api('/api/move', {
    method: 'POST',
    body: JSON.stringify({ uci }),
  }).then(s => {
    renderState(s);
    triggerEngineIfNeeded();
  }).catch(err => {
    $status.text('Invalid: ' + err.message);
    refreshState();
  });
  // Optimistically snapback; server reply will drive the next render.
  return 'snapback';
}

function onSnapEnd() {
  if (board && currentState) board.position(currentState.fen, false);
}

async function newGame() {
  const mode = $('#mode').val();
  const human_color = $('#human-color').val();
  const movetime_ms = parseInt($('#movetime').val(), 10) || 1000;
  const s = await api('/api/new_game', {
    method: 'POST',
    body: JSON.stringify({ mode, human_color, movetime_ms }),
  });
  renderState(s);
  triggerEngineIfNeeded();
}

async function loadFen() {
  const fen = $('#fen-input').val().trim();
  if (!fen) return;
  const mode = $('#mode').val();
  const human_color = $('#human-color').val();
  const movetime_ms = parseInt($('#movetime').val(), 10) || 1000;
  try {
    const s = await api('/api/new_game', {
      method: 'POST',
      body: JSON.stringify({ mode, human_color, movetime_ms, fen }),
    });
    renderState(s);
    triggerEngineIfNeeded();
  } catch (e) {
    $status.text('FEN error: ' + e.message);
  }
}

$(function () {
  board = Chessboard('board', {
    draggable: true,
    position: 'start',
    pieceTheme: 'https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/website/img/chesspieces/wikipedia/{piece}.png',
    onDragStart,
    onDrop,
    onSnapEnd,
  });
  $('#new-game').on('click', () => newGame().catch(e => $status.text(e.message)));
  $('#load-fen').on('click', () => loadFen());
  refreshState().catch(e => $status.text(e.message));
});
