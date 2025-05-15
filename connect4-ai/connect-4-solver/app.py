from fastapi import FastAPI, HTTPException
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from board import Board  
from solver import solve

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]  # 0 = empty, 1 = X, 2 = O
    current_player: int     # 1 or 2
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int


def from_input(state: GameState) -> Board:
    """
    Chuyển GameState thành Board:
      - state.board: ma trận rows x cols
      - state.current_player: 1 hoặc 2
    """
    rows = len(state.board)
    cols = len(state.board[0]) if rows > 0 else 0
    b = Board(width=cols, height=rows)

    # Thiết lập bitboard và đếm số quân mỗi cột
    col_counts = [0] * cols
    for r, row in enumerate(state.board):
        for c, v in enumerate(row):
            if v not in (0, 1, 2):
                raise ValueError(f"Invalid cell ({r},{c}): {v}")
            if v:
                player_idx = v - 1  # 1→0, 2→1
                bit_row = (rows - 1) - r
                bit_index = (rows + 1) * c + bit_row
                b.board_state[player_idx] |= (1 << bit_index)
                col_counts[c] += 1

    # Cập nhật độ cao của mỗi cột
    for c in range(cols):
        b.col_heights[c] = (rows + 1) * c + col_counts[c]

    # Số nước đi hiện tại
    b.moves = sum(col_counts)

    # Đồng bộ current_player
    desired = state.current_player - 1
    if b.get_current_player() != desired:
        b.moves -= 1

    # Reset history nếu cần
    b.history = []
    return b

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}


@app.post("/api/connect4-move", response_model=AIResponse)
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("No valid moves")

        # Chuyển GameState → Board và tìm nước đi tốt nhất
        board = from_input(game_state)
        move = solve(board)
        return AIResponse(move=move)
    except Exception as e:
        # Fallback: trả về nước đi đầu tiên
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
