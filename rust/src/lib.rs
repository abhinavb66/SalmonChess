//Rust hot loop for SalmonNNUE: NNUE inference + PUCT MCTS, exposed to Python
//as the `salmon_nnue_rs` extension module. Mirrors nnue/mcts.py and
//nnue/network.py semantics exactly so the Python behavioral tests transfer.

use pyo3::prelude::*;

#[pymodule]
fn salmon_nnue_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust NNUE inference + MCTS for SalmonNNUE")?;
    Ok(())
}
