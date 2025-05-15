from board import Board, LRUCache, get_tt_entry

from board import Board, LRUCache, get_tt_entry



def evaluate(board):
    # Nếu trạng thái thắng hiện tại, trả về điểm tuyệt đối
    if board.winning_board_state():
        return board.get_score()

    # Các hệ số đánh giá
    setScoreFor2 = 1   # Điểm cho 2 quân liên tiếp
    setScoreFor3 = 5    # Điểm cho 3 quân liên tiếp
    setNegWeight = 1.5  # Trọng số giảm cho điểm đối thủ
    

    score = 0
    current_player = board.get_current_player()
    opponent = board.get_opponent()
    
    # Xác định ký hiệu của người chơi và đối thủ theo bitboard __repr__
    # Ở đây, giả sử: người chơi 0 dùng 'x', người chơi 1 dùng 'o'
    if current_player == 0:
        mark = 'x'
        opponent_mark = 'o'
    else:
        mark = 'o'
        opponent_mark = 'x'
    
    # Hàm trợ giúp để lấy ký hiệu tại một ô từ bit index
    def get_cell(pos):
        if board.board_state[0] & (1 << pos):
            return 'x'
        elif board.board_state[1] & (1 << pos):
            return 'o'
        else:
            return '.'

    # Danh sách các đường thắng (mỗi đường gồm 4 ô) được tạo động theo kích thước bàn cờ.
    lines = []
    cols = board.w
    rows = board.h
    stride = board.h + 1  # vì mỗi cột cách nhau (h+1) bit

    # --- Hàng ngang ---
    for r in range(rows):
        for c in range(cols - 3):
            line = [stride * (c + i) + r for i in range(4)]
            lines.append(line)

    # --- Hàng dọc ---
    for c in range(cols):
        for r in range(rows - 3):
            line = [stride * c + (r + i) for i in range(4)]
            lines.append(line)

    # --- Đường chéo dương (xuống) --- 
    # Ví dụ: bắt đầu từ hàng trên (r nhỏ) và tăng dần theo cột
    for c in range(cols - 3):
        for r in range(rows - 3):
            line = [stride * (c + i) + (r + i) for i in range(4)]
            lines.append(line)

    # --- Đường chéo âm (lên) ---
    for c in range(cols - 3):
        for r in range(3, rows):
            line = [stride * (c + i) + (r - i) for i in range(4)]
            lines.append(line)

    # Duyệt qua tất cả các đường thắng và cộng/trừ điểm
    for line in lines:
        # Lấy giá trị của các ô trong đường này
        cells = [get_cell(pos) for pos in line]

        # Nếu đường này không chứa quân đối thủ, tính điểm cho người chơi
        if opponent_mark not in cells:
            count = cells.count(mark)
            if count == 2:
                score += setScoreFor2
            elif count == 3:
                score += setScoreFor3

        # Nếu đường này không chứa quân của người chơi, tính điểm cho đối thủ (trừ điểm với trọng số)
        if mark not in cells:
            count = cells.count(opponent_mark)
            if count == 2:
                score -= setScoreFor2 * setNegWeight
            elif count == 3:
                score -= setScoreFor3 * setNegWeight

    return score



def solve(board, max_depth=8):

    TT = LRUCache(4531985219092)

    def recurse(alpha, beta, depth=0):
        alpha_original = alpha
        best_move = None
        key = board.get_key()
        
        if key in TT:
            entry = TT[key]
            if entry['LB']:
                alpha = max(alpha, entry['value'])
            elif entry['UB']:
                beta = min(beta, entry['value'])
            else:
                return entry['value'], None
            if alpha >= beta:
                return entry['value'], None

        if board.winning_board_state():
            return board.get_score(), None
        elif board.moves == board.w * board.h:
            return 0, None  # Hòa

        # Giới hạn độ sâu tìm kiếm, sử dụng hàm đánh giá tạm thời
        if depth >= max_depth:
            return evaluate(board), None



        value = -board.w * board.h
        moves = list(board.get_search_order())
        if not moves:
            return value, None

        for col in moves:
            board.play(col)
            score, _ = recurse(-beta, -alpha, depth+1)
            board.backtrack()

            if -score > value:
                value = -score
                best_move = col

            alpha = max(alpha, value)
            if alpha >= beta:
                break  # cắt nhánh Alpha

        if best_move is None:
            center = board.w // 2
            best_move = center if board.can_play(center) else moves[0]

        if value <= alpha_original:
            TT[key] = get_tt_entry(value, UB=True)
        elif value >= beta:
            TT[key] = get_tt_entry(value, LB=True)
        else:
            TT[key] = get_tt_entry(value)

        return value, best_move

    _, best_move = recurse(-1e9, 1e9)
    if best_move is None:
        center = board.w // 2
        best_move = center if board.can_play(center) else 0
    return best_move
