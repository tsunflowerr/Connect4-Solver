from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

import c4a0_rust
from c4a0.training import TrainingGen

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

# Load model callback at startup, giống như play()
gen = TrainingGen.load_latest("training")
nn = gen.get_model("training")
model_callback = lambda model_id, x: nn.forward_numpy(x)

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
        move = c4a0_rust.predict_move(
            game_state.board,
            game_state.current_player,
            800,    # n_mcts_iterations, có thể điều chỉnh
            5.0,    # c_exploration
            0.01,   # c_ply_penalty
            model_callback
        )
        if move not in game_state.valid_moves:
            move = game_state.valid_moves[0]
        return AIResponse(move=move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)