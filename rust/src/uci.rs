//Board/UCI translation and terminal detection, replicating common.py and
//nnue/network.py exactly so the Python behavioral tests transfer.

use cozy_chess::{Board, Color, GameStatus, Move, Piece, Square};

//Flatten cozy's callback move generation into a Vec, preserving iteration
//order (tie-breaks in the search depend on it matching Python's).
pub fn legal_moves(board: &Board) -> Vec<Move> {
    let mut moves = Vec::with_capacity(48);
    board.generate_moves(|piece_moves| {
        for mv in piece_moves {
            moves.push(mv);
        }
        false
    });
    moves
}

fn square_name(sq: usize) -> String {
    let file = (b'a' + (sq % 8) as u8) as char;
    let rank = (b'1' + (sq / 8) as u8) as char;
    format!("{}{}", file, rank)
}

fn promo_char(piece: Piece) -> char {
    match piece {
        Piece::Knight => 'n',
        Piece::Bishop => 'b',
        Piece::Rook => 'r',
        Piece::Queen => 'q',
        _ => 'q',
    }
}

//True when the move is castling in cozy's king-takes-own-rook encoding.
fn is_castling(board: &Board, mv: Move) -> bool {
    board.piece_on(mv.from) == Some(Piece::King)
        && board.piece_on(mv.to) == Some(Piece::Rook)
        && board.color_on(mv.to) == board.color_on(mv.from)
}

//Standard-UCI string for a move (common.to_uci): castling normalized to the
//king's g-/c-file destination; promotion char appended.
pub fn to_uci(board: &Board, mv: Move) -> String {
    let from = mv.from as usize;
    let mut to = mv.to as usize;
    if is_castling(board, mv) {
        let rank = from - (from % 8);
        to = rank + if to % 8 == 7 { 6 } else { 2 };
    }
    let mut s = square_name(from);
    s.push_str(&square_name(to));
    if let Some(p) = mv.promotion {
        s.push(promo_char(p));
    }
    s
}

//Policy index (network.policy_index): from*64+to on the stm-mirrored board,
//castling normalized like to_uci; promotions collapse onto the from-to cell.
pub fn policy_index(board: &Board, mv: Move) -> usize {
    let mut from = mv.from as usize;
    let mut to = mv.to as usize;
    if is_castling(board, mv) {
        let rank = from - (from % 8);
        to = rank + if to % 8 == 7 { 6 } else { 2 };
    }
    if board.side_to_move() == Color::Black {
        from ^= 56;
        to ^= 56;
    }
    from * 64 + to
}

//mcts._is_capture: destination occupied (this counts castling as a capture,
//since the rook sits on the destination) or an en-passant pawn push.
pub fn is_capture(board: &Board, mv: Move) -> bool {
    if board.piece_on(mv.to).is_some() {
        return true;
    }
    board.piece_on(mv.from) == Some(Piece::Pawn) && (mv.from as usize % 8) != (mv.to as usize % 8)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Terminal {
    Loss,
    Draw,
}

//common.insufficient_material: no pawns/rooks/queens and at most one minor.
fn insufficient_material(board: &Board) -> bool {
    if board.pieces(Piece::Pawn).len() != 0
        || board.pieces(Piece::Rook).len() != 0
        || board.pieces(Piece::Queen).len() != 0
    {
        return false;
    }
    let minors = board.pieces(Piece::Knight).len() + board.pieces(Piece::Bishop).len();
    minors <= 1
}

//common.terminal_result: side-to-move loss (checkmate), draw (stalemate /
//fifty-move / insufficient material), or None (ongoing).
pub fn terminal_result(board: &Board) -> Option<Terminal> {
    match board.status() {
        GameStatus::Won => Some(Terminal::Loss),
        GameStatus::Drawn => Some(Terminal::Draw),
        GameStatus::Ongoing => {
            if insufficient_material(board) {
                Some(Terminal::Draw)
            } else {
                None
            }
        }
    }
}
