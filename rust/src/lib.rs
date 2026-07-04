//Rust hot loop for SalmonNNUE: NNUE inference + PUCT MCTS, exposed to Python
//as the `salmon_nnue_rs` extension module. Mirrors nnue/mcts.py and
//nnue/network.py semantics exactly so the Python behavioral tests transfer.

mod uci;

use cozy_chess::Board;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

#[pymodule]
fn salmon_nnue_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust NNUE inference + MCTS for SalmonNNUE")?;
    m.add_function(wrap_pyfunction!(zobrist_hash, m)?)?;
    m.add_function(wrap_pyfunction!(legal_moves_uci, m)?)?;
    Ok(())
}
