from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from c4a0.main import MCTS
from c4a0.pos import Pos

from c4a0.training import TrainingGen

app = FastAPI()

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}



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
    is_new_game: bool

class AIResponse(BaseModel):
    move: int


@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    base_dir = os.path.join(os.path.dirname(__file__), "training")
    gen = TrainingGen.load_latest(base_dir)
    nn = gen.get_model(base_dir)


    try:
        initial_pos = Pos.from_2d_board(game_state.board)
        mcts = MCTS(initial_pos, nn, c_exploration=1.5, c_ply_penalty=0.01)
        mcts.run(1300)
        policy = mcts.get_root_policy()
        move = np.argmax(policy)
        if move not in game_state.valid_moves:
            move = game_state.valid_moves[0]
        return AIResponse(move=move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
