//NNUE inference, mirroring nnue/network.py. Float32 like the NumPy path so
//results are parity-testable. The policy weights are transposed at load so
//each move's logit is one contiguous 512-dot-product — the search only ever
//needs the legal moves' logits, never all 4096.

use cozy_chess::{Board, Color, Move, Piece};

use crate::uci;

pub const FEATURES: usize = 768;
pub const ACC: usize = 256;
pub const CONCAT: usize = 2 * ACC; // 512
pub const L1: usize = 32;
pub const POLICY: usize = 64 * 64; // 4096
const BLEND_POLICY_WEIGHT: f64 = 0.75;

pub struct Net {
    ft_w: Vec<f32>,   // [768*256] row-major
    ft_b: Vec<f32>,   // [256]
    l1_w_t: Vec<f32>, // [32*512] transposed from [512,32]
    l1_b: Vec<f32>,   // [32]
    v_w: Vec<f32>,    // [32]
    v_b: f64,
    p_w_t: Vec<f32>,  // [4096*512] transposed from [512,4096]
    p_b: Vec<f32>,    // [4096]
}

fn decode(bytes: &[u8], n: usize, name: &str) -> Result<Vec<f32>, String> {
    if bytes.len() != n * 4 {
        return Err(format!("{}: expected {} f32 ({} bytes), got {}", name, n, n * 4, bytes.len()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl Net {
    //Build from little-endian f32 byte buffers (as np.ascontiguousarray(...,
    //np.float32).tobytes() produces). Transposes l1 and policy weights.
    pub fn from_bytes(
        ft_w: &[u8], ft_b: &[u8], l1_w: &[u8], l1_b: &[u8], v_w: &[u8], v_b: f64,
        p_w: &[u8], p_b: &[u8],
    ) -> Result<Net, String> {
        let ft_w = decode(ft_w, FEATURES * ACC, "ft.weight")?;
        let ft_b = decode(ft_b, ACC, "ft.bias")?;
        let l1_w_flat = decode(l1_w, CONCAT * L1, "l1.weight")?;
        let l1_b = decode(l1_b, L1, "l1.bias")?;
        let v_w = decode(v_w, L1, "value.weight")?;
        let p_w_flat = decode(p_w, CONCAT * POLICY, "policy.weight")?;
        let p_b = decode(p_b, POLICY, "policy.bias")?;

        let mut l1_w_t = vec![0f32; L1 * CONCAT]; // [32][512]
        for i in 0..CONCAT {
            for j in 0..L1 {
                l1_w_t[j * CONCAT + i] = l1_w_flat[i * L1 + j];
            }
        }
        let mut p_w_t = vec![0f32; POLICY * CONCAT]; // [4096][512]
        for i in 0..CONCAT {
            for k in 0..POLICY {
                p_w_t[k * CONCAT + i] = p_w_flat[i * POLICY + k];
            }
        }
        Ok(Net { ft_w, ft_b, l1_w_t, l1_b, v_w, v_b, p_w_t, p_b })
    }

    fn ft_col(&self, feature: usize) -> &[f32] {
        &self.ft_w[feature * ACC..feature * ACC + ACC]
    }

    //Value in (0,1) from stm's view + the clipped 512 concat (stm-first).
    pub fn value_and_x(&self, acc: &Accumulator, stm: Color) -> (f64, [f32; CONCAT]) {
        let me = stm as usize;
        let mut x = [0f32; CONCAT];
        for i in 0..ACC {
            x[i] = acc.vals[me][i].clamp(0.0, 1.0);
            x[ACC + i] = acc.vals[1 - me][i].clamp(0.0, 1.0);
        }
        let mut dot = 0f32;
        for j in 0..L1 {
            let row = &self.l1_w_t[j * CONCAT..j * CONCAT + CONCAT];
            let mut s = self.l1_b[j];
            for i in 0..CONCAT {
                s += x[i] * row[i];
            }
            dot += s.clamp(0.0, 1.0) * self.v_w[j];
        }
        let value = 1.0 / (1.0 + (-(dot as f64 + self.v_b)).exp());
        (value, x)
    }

    pub fn logit(&self, x: &[f32; CONCAT], policy_idx: usize) -> f32 {
        let row = &self.p_w_t[policy_idx * CONCAT..policy_idx * CONCAT + CONCAT];
        let mut s = self.p_b[policy_idx];
        for i in 0..CONCAT {
            s += x[i] * row[i];
        }
        s
    }

    //Blended move priors, mirroring mcts._priors (PRIOR_MODE "blend"): softmax
    //over legal-move policy logits, blended 0.75/0.25 with a uniform +
    //capture/promotion-boost heuristic.
    pub fn move_priors(&self, board: &Board, moves: &[Move], x: &[f32; CONCAT]) -> Vec<f64> {
        let logits: Vec<f64> = moves
            .iter()
            .map(|&m| self.logit(x, uci::policy_index(board, m)) as f64)
            .collect();
        let maxl = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - maxl).exp()).collect();
        let psum: f64 = exps.iter().sum();

        let heur: Vec<f64> = moves
            .iter()
            .map(|&m| {
                let mut h = 1.0;
                if uci::is_capture(board, m) {
                    h += 2.0;
                }
                if m.promotion.is_some() {
                    h += 2.0;
                }
                h
            })
            .collect();
        let hsum: f64 = heur.iter().sum();

        exps.iter()
            .zip(heur.iter())
            .map(|(e, h)| BLEND_POLICY_WEIGHT * (e / psum) + (1.0 - BLEND_POLICY_WEIGHT) * (h / hsum))
            .collect()
    }
}

pub fn feature_index(perspective: Color, piece_color: Color, piece: Piece, sq: usize) -> usize {
    let rel = if perspective == Color::White { sq } else { sq ^ 56 };
    let side = if piece_color == perspective { 0 } else { 1 };
    side * 384 + (piece as usize) * 64 + rel
}

//Per-perspective first-layer activations; vals indexed [White, Black].
pub struct Accumulator {
    pub vals: [[f32; ACC]; 2],
}

impl Accumulator {
    pub fn from_board(net: &Net, board: &Board) -> Accumulator {
        let mut vals = [[0f32; ACC]; 2];
        for &persp in &[Color::White, Color::Black] {
            let row = &mut vals[persp as usize];
            row.copy_from_slice(&net.ft_b);
            for &color in &[Color::White, Color::Black] {
                for &piece in &Piece::ALL {
                    for sq in board.colored_pieces(color, piece) {
                        let f = feature_index(persp, color, piece, sq as usize);
                        let col = net.ft_col(f);
                        for i in 0..ACC {
                            row[i] += col[i];
                        }
                    }
                }
            }
        }
        Accumulator { vals }
    }

    //Accumulator after `mv`, from the pre-move board. Castling (king onto own
    //rook, two pieces move) falls back to a full recompute of `after`.
    pub fn play(&self, net: &Net, before: &Board, mv: Move, after: &Board) -> Accumulator {
        let piece = before.piece_on(mv.from).unwrap();
        let mover = before.color_on(mv.from).unwrap();
        let victim = before.piece_on(mv.to);

        if piece == Piece::King && victim == Some(Piece::Rook)
            && before.color_on(mv.to) == Some(mover)
        {
            return Accumulator::from_board(net, after);
        }

        let fs = mv.from as usize;
        let ts = mv.to as usize;
        let final_piece = mv.promotion.unwrap_or(piece);
        let is_ep = piece == Piece::Pawn && victim.is_none() && (fs % 8) != (ts % 8);

        let mut vals = self.vals;
        for &persp in &[Color::White, Color::Black] {
            let row = &mut vals[persp as usize];
            let leave = net.ft_col(feature_index(persp, mover, piece, fs));
            let arrive = net.ft_col(feature_index(persp, mover, final_piece, ts));
            for i in 0..ACC {
                row[i] += arrive[i] - leave[i];
            }
            if let Some(vp) = victim {
                let vc = before.color_on(mv.to).unwrap();
                let col = net.ft_col(feature_index(persp, vc, vp, ts));
                for i in 0..ACC {
                    row[i] -= col[i];
                }
            } else if is_ep {
                let cap_sq = if mover == Color::White { ts - 8 } else { ts + 8 };
                let enemy = if mover == Color::White { Color::Black } else { Color::White };
                let col = net.ft_col(feature_index(persp, enemy, Piece::Pawn, cap_sq));
                for i in 0..ACC {
                    row[i] -= col[i];
                }
            }
        }
        Accumulator { vals }
    }
}
