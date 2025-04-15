from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import random
import torch
import numpy as np

from c4a0.training import TrainingGen
from c4a0.utils import get_torch_device

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = "training"
DEVICE = get_torch_device()

try:
    latest_gen = TrainingGen.load_latest(BASE_DIR)
    model = latest_gen.get_model(BASE_DIR)
    model.to(DEVICE)
    model.eval()
except:
    model = None

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

@app.post("/api/connect4-move", response_model=AIResponse)
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not model:
            raise ValueError("Model not loaded")

        if not game_state.valid_moves:
            raise ValueError("No valid moves")

        raw_board = np.array(game_state.board)
        board_tensor = np.zeros((2, 6, 7), dtype=np.float32)

        for r in range(6):
            for c in range(7):
                if raw_board[r][c] == game_state.current_player:
                    board_tensor[0][r][c] = 1
                elif raw_board[r][c] != 0:
                    board_tensor[1][r][c] = 1

        board_tensor = board_tensor[np.newaxis, ...]

        with torch.no_grad():
            policy, _, _ = model.forward_numpy(board_tensor)
            policy = policy[0]

        valid_probs = [(i, policy[i]) for i in game_state.valid_moves]
        valid_probs.sort(key=lambda x: x[1], reverse=True)
        selected_move = valid_probs[0][0]

        return AIResponse(move=selected_move)

    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
