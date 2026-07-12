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

//Retention policy for the between-move subtree. The PUCT head start lives in
//visit counts, so relocate keeps a node's children block only if the node has
//MIN_KEEP_VISITS — a barely-visited node re-expands on its next visit for one
//net eval, same as cold, while its own visits/value_sum are always retained.
//REUSE_MAX_NODES is a hard backstop (~56 bytes/node once states are stripped:
//1M nodes ~ 56MB): past it, further blocks are truncated the same way. Reuse
//must be a head start, never a memory liability (an unbounded cache once grew
//until a gauntlet engine stopped answering within the move budget).
const MIN_KEEP_VISITS: u32 = 4;
const REUSE_MAX_NODES: usize = 1_000_000;

//Search tree kept alive between moves for reuse: the cached tree's root is
//the child whose move we returned (its stats/subtree intact, states stripped),
//and `board` is the position after that move. If the next search's position
//matches one of the root's children (i.e. the opponent replied), that subtree
//is promoted to the new root instead of starting cold.
pub struct TreeCache {
    tree: Tree,
    board: Board,
}

impl TreeCache {
    //Test/debug hook: (arena length, root visits) of the cached tree. After a
    //search that promoted a subtree, the cached root (the child we played)
    //keeps visits from previous moves — far more than that call's simulations
    //alone could give it, which is what proves reuse happened.
    pub fn stats(&self) -> (usize, u32) {
        (self.tree.arena.len(), self.tree.arena[0].visits)
    }
}

//Move the subtree rooted at `root` into a fresh arena, applying the retention
//policy: every relocated node's state is dropped (boards/accumulators, ~2KB
//per visited node, rematerialize lazily on the next search exactly as they
//would in a cold tree), and a node's children block is dropped — children set
//back to None, so it re-expands on visit — unless the node is well-visited
//and the arena is under the backstop. The root's block is always kept: it is
//what try_promote matches the opponent's reply against. Old blocks are
//disjoint and each reachable node is visited once, so a single forward pass
//relocates every kept block.
fn relocate(mut old: Vec<Node>, root: usize) -> Tree {
    let mut arena = Vec::with_capacity(1024);
    arena.push(std::mem::replace(&mut old[root], Node::stub(None, 0.0)));
    let mut i = 0;
    while i < arena.len() {
        if let Some((os, cnt)) = arena[i].children {
            let keep = i == 0
                || (arena[i].visits >= MIN_KEEP_VISITS
                    && arena.len() + cnt <= REUSE_MAX_NODES);
            if keep {
                let ns = arena.len();
                for j in 0..cnt {
                    arena.push(std::mem::replace(&mut old[os + j], Node::stub(None, 0.0)));
                }
                arena[i].children = Some((ns, cnt));
            } else {
                arena[i].children = None;
            }
        }
        arena[i].state = None;
        i += 1;
    }
    Tree { arena }
}

//Promote the cached subtree matching `board` (one opponent reply after our
//played move), if any. The cached root's children enumerate every legal
//reply, so a legal continuation always has a candidate; anything else (new
//game, analysis jump) misses and the caller starts a fresh tree.
fn try_promote(cache: TreeCache, board: &Board) -> Option<Tree> {
    let (cs, cl) = cache.tree.arena[0].children?;
    let mut found = None;
    for ci in 0..cl {
        let mv = cache.tree.arena[cs + ci].mv.unwrap();
        let mut b = cache.board.clone();
        b.play_unchecked(mv);
        if b.hash() == board.hash() && b.same_position(board) {
            found = Some(cs + ci);
            break;
        }
    }
    Some(relocate(cache.tree.arena, found?))
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

//Build (or adopt) the root and run simulations until the cap or deadline.
//Returns the tree (root at index 0, expanded after >=1 simulation) and the
//number of simulations run in THIS call (a promoted root keeps old visits).
fn run(
    net: &Net, board: &Board, history: &[u64], deadline: Option<Instant>, cap: u64,
    promoted: Option<Tree>,
) -> (Tree, u64) {
    let mut tree = promoted.unwrap_or_else(|| {
        let mut t = Tree { arena: Vec::with_capacity(1024) };
        t.arena.push(Node::stub(None, 1.0));
        t
    });
    if tree.arena[0].state.is_none() {
        let acc = Accumulator::from_board(net, board);
        tree.arena[0].state = Some(Box::new(NodeState { board: board.clone(), acc }));
    }

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
    (tree, sims)
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

//Best move by MCTS within the budget, mirroring mcts.bestMove. `cache` (when
//tree reuse is enabled) carries the previous move's tree in and this move's
//tree out; pass a `&mut None` throwaway to search statelessly.
pub fn search(
    net: &Net, fen: &str, history: &[u64], movetime_s: f64, max_sims: Option<u64>,
    cache: &mut Option<TreeCache>, reuse: bool,
) -> Result<SearchResult, String> {
    let board = Board::from_fen(fen, false).map_err(|e| format!("bad fen: {:?}", e))?;
    //Taken unconditionally: a stale tree from an unrelated position would just
    //miss in try_promote, and holding it across early returns wastes memory.
    let promoted = cache.take().filter(|_| reuse).and_then(|c| try_promote(c, &board));

    let moves = uci::legal_moves(&board);
    if moves.is_empty() {
        return Ok(SearchResult { best_uci: "0000".to_string(), sims: 0 });
    }
    if moves.len() == 1 {
        return Ok(SearchResult { best_uci: uci::to_uci(&board, moves[0]), sims: 0 });
    }

    let cap = max_sims.unwrap_or(u64::MAX);
    let (tree, sims) = run(net, &board, history, deadline_for(movetime_s), cap, promoted);

    let (cs, cl) = match tree.arena[0].children {
        Some(c) => c,
        None => return Ok(SearchResult { best_uci: uci::to_uci(&board, moves[0]), sims }),
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
    let best_uci = uci::to_uci(&board, mv);
    if reuse {
        //Cache only the played child's subtree (stripped, truncated) — the
        //rest of the tree is dead the moment we commit to this move.
        let t = relocate(tree.arena, best);
        let mut after = board.clone();
        after.play_unchecked(mv);
        *cache = Some(TreeCache { tree: t, board: after });
    }
    Ok(SearchResult { best_uci, sims })
}

//Root children (uci, visits, value_sum) after exactly `sims` simulations, for
//the repetition test.
pub fn root_child_stats(
    net: &Net, fen: &str, history: &[u64], sims: u64,
) -> Result<Vec<(String, u32, f64)>, String> {
    let board = Board::from_fen(fen, false).map_err(|e| format!("bad fen: {:?}", e))?;
    let (tree, _) = run(net, &board, history, None, sims, None);
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
