//Rust hot loop for SalmonNNUE: NNUE inference + PUCT MCTS, exposed to Python
//as the `salmon_nnue_rs` extension module. Mirrors nnue/mcts.py and
//nnue/network.py semantics exactly so the Python behavioral tests transfer.

mod mcts;
mod net;
mod uci;

use std::sync::Mutex;

use cozy_chess::Board;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mcts::TreeCache;
use net::{Accumulator, Net};

fn parse_board(fen: &str) -> PyResult<Board> {
    Board::from_fen(fen, false).map_err(|e| PyValueError::new_err(format!("bad fen: {:?}", e)))
}

//Zobrist hash of a position (cozy's board.hash()), for history-parity checks.
#[pyfunction]
fn zobrist_hash(fen: &str) -> PyResult<u64> {
    Ok(parse_board(fen)?.hash())
}

//Legal moves as standard-UCI strings, in cozy's generation order.
#[pyfunction]
fn legal_moves_uci(fen: &str) -> PyResult<Vec<String>> {
    let board = parse_board(fen)?;
    Ok(uci::legal_moves(&board)
        .into_iter()
        .map(|mv| uci::to_uci(&board, mv))
        .collect())
}

#[pyclass(frozen)]
struct RustEngine {
    net: Net,
    //Tree carried across searches when `tree_reuse` (promoted when the new
    //position is a played-move + one-reply continuation of the last search).
    tree_reuse: bool,
    cache: Mutex<Option<TreeCache>>,
}

#[pymethods]
impl RustEngine {
    //Weights are little-endian f32 byte buffers; v_b is an f64 scalar. `quant`
    //enables int8 inference for the L1/policy matmuls (approximate — see
    //net.rs); the default false keeps exact f32 parity with the NumPy path.
    #[new]
    #[pyo3(signature = (ft_w, ft_b, l1_w, l1_b, v_w, v_b, p_w, p_b, quant=false, tree_reuse=false))]
    fn new(
        ft_w: Vec<u8>, ft_b: Vec<u8>, l1_w: Vec<u8>, l1_b: Vec<u8>, v_w: Vec<u8>, v_b: f64,
        p_w: Vec<u8>, p_b: Vec<u8>, quant: bool, tree_reuse: bool,
    ) -> PyResult<Self> {
        let net = Net::from_bytes(&ft_w, &ft_b, &l1_w, &l1_b, &v_w, v_b, &p_w, &p_b, quant)
            .map_err(PyValueError::new_err)?;
        Ok(RustEngine { net, tree_reuse, cache: Mutex::new(None) })
    }

    //Best move by MCTS. Returns (uci, simulations run this call). GIL released
    //for the whole search (RustEngine is frozen; the cache is behind a Mutex).
    fn search(
        &self, py: Python<'_>, fen: &str, history: Vec<u64>, movetime_s: f64,
        max_sims: Option<u64>,
    ) -> PyResult<(String, u64)> {
        py.detach(|| {
            let mut cache = self.cache.lock().unwrap();
            mcts::search(&self.net, fen, &history, movetime_s, max_sims,
                         &mut cache, self.tree_reuse)
        })
        .map(|r| (r.best_uci, r.sims))
        .map_err(PyValueError::new_err)
    }

    //Test/debug hook: (arena_len, root_visits) of the tree cached by the last
    //search, or None (reuse disabled / no search yet).
    fn cache_stats(&self) -> Option<(usize, u32)> {
        self.cache.lock().unwrap().as_ref().map(TreeCache::stats)
    }

    //Repetition-test hook: root children (uci, visits, value_sum) after `sims`
    //simulations.
    fn root_stats(
        &self, py: Python<'_>, fen: &str, history: Vec<u64>, sims: u64,
    ) -> PyResult<Vec<(String, u32, f64)>> {
        py.detach(|| mcts::root_child_stats(&self.net, fen, &history, sims))
            .map_err(PyValueError::new_err)
    }

    //Parity/debug: play the given UCI moves advancing the accumulator
    //INCREMENTALLY (exercising Accumulator::play incl. the castling fallback),
    //then return the value. Compared against eval_debug (full recompute) in
    //tests to catch incremental-update bugs.
    fn eval_after_moves(&self, fen: &str, moves: Vec<String>) -> PyResult<f64> {
        let mut board = parse_board(fen)?;
        let mut acc = Accumulator::from_board(&self.net, &board);
        for uci_str in moves {
            let mv = uci::legal_moves(&board)
                .into_iter()
                .find(|&m| uci::to_uci(&board, m) == uci_str)
                .ok_or_else(|| PyValueError::new_err(format!("illegal move {}", uci_str)))?;
            let mut after = board.clone();
            after.play_unchecked(mv);
            acc = acc.play(&self.net, &board, mv, &after);
            board = after;
        }
        let (value, _) = self.net.value_and_x(&acc, board.side_to_move());
        Ok(value)
    }

    //Parity/debug: the (0,1) value and per-move blended priors exactly as
    //mcts._expand computes them, keyed by standard-UCI move string.
    fn eval_debug(&self, fen: &str) -> PyResult<(f64, Vec<(String, f64)>)> {
        let board = parse_board(fen)?;
        let acc = Accumulator::from_board(&self.net, &board);
        let (value, x) = self.net.value_and_x(&acc, board.side_to_move());
        let moves = uci::legal_moves(&board);
        let priors = self.net.move_priors(&board, &moves, &x);
        let out = moves
            .iter()
            .zip(priors.iter())
            .map(|(&m, &p)| (uci::to_uci(&board, m), p))
            .collect();
        Ok((value, out))
    }
}

#[pymodule]
fn salmon_nnue_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust NNUE inference + MCTS for SalmonNNUE")?;
    m.add_function(wrap_pyfunction!(zobrist_hash, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves_uci, m)?)?;
    m.add_class::<RustEngine>()?;
    Ok(())
}
