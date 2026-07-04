//PUCT MCTS, a faithful transcription of nnue/mcts.py (post visit-fix). Uses an
//index-based arena so the recursive `simulate` never holds a node borrow across
//the recursive call. Node boards/accumulators are materialized lazily on first
//visit, exactly like the Python search.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::{Duration, Instant};

use cozy_chess::{Board, Move};

//The `seen` keys are already zobrist hashes; the default SipHash is wasted work
//on them. A cheap multiplicative mix spreads the bits well enough for a set
//that holds at most one search path.
#[derive(Default)]
struct U64Hasher(u64);
impl Hasher for U64Hasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, _: &[u8]) {
        unreachable!("only u64 keys")
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}
type SeenSet = HashSet<u64, BuildHasherDefault<U64Hasher>>;

use crate::net::{Accumulator, Net};
use crate::uci::{self, Terminal};

const CPUCT: f64 = 1.5;
const FPU_REDUCTION: f64 = 0.3;

struct NodeState {
    board: Board,
    acc: Accumulator,
}

struct Node {
    mv: Option<Move>,
    prior: f64,
    visits: u32,
    value_sum: f64,
    children: Option<(usize, usize)>, // (arena start, count); None = unexpanded
    state: Option<Box<NodeState>>,    // materialized lazily
    terminal: Option<Terminal>,
}

impl Node {
    fn stub(mv: Option<Move>, prior: f64) -> Node {
        Node { mv, prior, visits: 0, value_sum: 0.0, children: None, state: None, terminal: None }
    }
    fn q(&self) -> f64 {
        if self.visits > 0 {
            self.value_sum / self.visits as f64
        } else {
            0.0
        }
    }
}

struct Tree {
    arena: Vec<Node>,
}

fn terminal_value(t: Terminal) -> f64 {
    match t {
        Terminal::Loss => -1.0,
        Terminal::Draw => 0.0,
    }
}

//Evaluate a leaf: attach child stubs with blended priors, return leaf value in
//[-1,1] from the node's side-to-move perspective. Mirrors mcts._expand.
fn expand(tree: &mut Tree, idx: usize, net: &Net) -> f64 {
    let (value, moves, priors) = {
        let state = tree.arena[idx].state.as_ref().unwrap();
        let (value, x) = net.value_and_x(&state.acc, state.board.side_to_move());
        let moves = uci::legal_moves(&state.board);
        let priors = net.move_priors(&state.board, &moves, &x);
        (value, moves, priors)
    };
    let start = tree.arena.len();
    for (m, p) in moves.iter().zip(priors.iter()) {
        tree.arena.push(Node::stub(Some(*m), *p));
    }
    tree.arena[idx].children = Some((start, moves.len()));
    2.0 * value - 1.0
}

//One simulation from node `idx`, returning its value in [-1,1]. `seen` holds
//the position hashes on the path from the root (plus game history).
fn simulate(tree: &mut Tree, idx: usize, net: &Net, seen: &mut SeenSet) -> f64 {
    if let Some(t) = tree.arena[idx].terminal {
        return terminal_value(t);
    }
    if tree.arena[idx].children.is_none() {
        let term = uci::terminal_result(&tree.arena[idx].state.as_ref().unwrap().board);
        if let Some(t) = term {
            tree.arena[idx].terminal = Some(t);
            return terminal_value(t);
        }
        return expand(tree, idx, net);
    }

    //Select the child maximizing PUCT (first max on ties, strict >).
    let (cs, cl) = tree.arena[idx].children.unwrap();
    let sqrt_visits = ((tree.arena[idx].visits + 1) as f64).sqrt();
    let fpu = (tree.arena[idx].q() - FPU_REDUCTION).max(-1.0);
    let mut best = cs;
    let mut best_score = f64::NEG_INFINITY;
    for ci in 0..cl {
        let c = &tree.arena[cs + ci];
        let q = if c.visits > 0 { c.q() } else { fpu };
        let score = q + CPUCT * c.prior * sqrt_visits / (1.0 + c.visits as f64);
        if score > best_score {
            best_score = score;
            best = cs + ci;
        }
    }

    //Materialize the chosen child's board/accumulator on first visit.
    if tree.arena[best].state.is_none() {
        let mv = tree.arena[best].mv.unwrap();
        let (child_board, child_acc) = {
            let parent = tree.arena[idx].state.as_ref().unwrap();
            let mut after = parent.board.clone();
            after.play_unchecked(mv);
            let acc = parent.acc.play(net, &parent.board, mv, &after);
            (after, acc)
        };
        tree.arena[best].state = Some(Box::new(NodeState { board: child_board, acc: child_acc }));
    }

    let h = tree.arena[best].state.as_ref().unwrap().board.hash();
    let v_child = if seen.contains(&h) {
        0.0 //repetition on the path -> draw
    } else {
        seen.insert(h);
        let v = simulate(tree, best, net, seen);
        seen.remove(&h);
        v
    };

    let v = -v_child;
    tree.arena[best].visits += 1;
    tree.arena[best].value_sum += v;
    v
}

//Build the root and run simulations until the cap or deadline. Returns the tree
//(root at index 0, expanded after >=1 simulation).
fn run(net: &Net, board: &Board, history: &[u64], deadline: Option<Instant>, cap: u64) -> Tree {
    let acc = Accumulator::from_board(net, board);
    let mut tree = Tree { arena: Vec::with_capacity(1024) };
    tree.arena.push(Node {
        mv: None,
        prior: 1.0,
        visits: 0,
        value_sum: 0.0,
        children: None,
        state: Some(Box::new(NodeState { board: board.clone(), acc })),
        terminal: None,
    });

    let mut seen: SeenSet = SeenSet::default();
    seen.extend(history.iter().copied());
    seen.insert(board.hash());

    let mut sims: u64 = 0;
    loop {
        simulate(&mut tree, 0, net, &mut seen);
        tree.arena[0].visits += 1; //the root's own visit (see bestMove in mcts.py)
        sims += 1;
        if sims >= cap {
            break;
        }
        if let Some(d) = deadline {
            if Instant::now() >= d {
                break;
            }
        }
    }
    tree
}

fn deadline_for(movetime_s: f64) -> Option<Instant> {
    if movetime_s.is_finite() {
        Some(Instant::now() + Duration::from_secs_f64(movetime_s.max(0.0)))
    } else {
        None
    }
}

pub struct SearchResult {
    pub best_uci: String,
    pub sims: u64,
}

//Best move by MCTS within the budget, mirroring mcts.bestMove.
pub fn search(
    net: &Net, fen: &str, history: &[u64], movetime_s: f64, max_sims: Option<u64>,
) -> Result<SearchResult, String> {
    let board = Board::from_fen(fen, false).map_err(|e| format!("bad fen: {:?}", e))?;
    let moves = uci::legal_moves(&board);
    if moves.is_empty() {
        return Ok(SearchResult { best_uci: "0000".to_string(), sims: 0 });
    }
    if moves.len() == 1 {
        return Ok(SearchResult { best_uci: uci::to_uci(&board, moves[0]), sims: 0 });
    }

    let cap = max_sims.unwrap_or(u64::MAX);
    let tree = run(net, &board, history, deadline_for(movetime_s), cap);

    let (cs, cl) = match tree.arena[0].children {
        Some(c) => c,
        None => return Ok(SearchResult { best_uci: uci::to_uci(&board, moves[0]), sims: 0 }),
    };
    //Best root move by (visits, q), first max on ties (matches Python's max()).
    let mut best = cs;
    let mut best_key = (0u32, f64::NEG_INFINITY);
    for ci in 0..cl {
        let c = &tree.arena[cs + ci];
        let key = (c.visits, c.q());
        if key.0 > best_key.0 || (key.0 == best_key.0 && key.1 > best_key.1) {
            best_key = key;
            best = cs + ci;
        }
    }
    let mv = tree.arena[best].mv.unwrap();
    let sims = tree.arena[0].visits as u64;
    Ok(SearchResult { best_uci: uci::to_uci(&board, mv), sims })
}

//Root children (uci, visits, value_sum) after exactly `sims` simulations, for
//the repetition test.
pub fn root_child_stats(
    net: &Net, fen: &str, history: &[u64], sims: u64,
) -> Result<Vec<(String, u32, f64)>, String> {
    let board = Board::from_fen(fen, false).map_err(|e| format!("bad fen: {:?}", e))?;
    let tree = run(net, &board, history, None, sims);
    let (cs, cl) = match tree.arena[0].children {
        Some(c) => c,
        None => return Ok(vec![]),
    };
    Ok((0..cl)
        .map(|ci| {
            let c = &tree.arena[cs + ci];
            (uci::to_uci(&board, c.mv.unwrap()), c.visits, c.value_sum)
        })
        .collect())
}
