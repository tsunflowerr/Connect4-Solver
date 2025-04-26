import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from c4a0.main import MCTS
from c4a0.pos import Pos
from c4a0.training import TrainingGen

# ==== Load model ngay khi khởi động server ====
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "training"))
print(f"👉 Looking for model in: {base_dir}")

if not os.path.exists(base_dir):
    print("❌ ERROR: 'training/' folder not found! Server will fail to load model.")
else:
    files = os.listdir(base_dir)
    print(f"📂 training/ contents: {files}")

try:
    gen = TrainingGen.load_latest(base_dir)
    print(f"✅ Loaded generation: {gen}")
    nn = gen.get_model(base_dir)
    print(f"✅ Model loaded successfully: {type(nn)}")
except Exception as e:
    print(f"❌ ERROR loading model: {str(e)}")
    raise e

# ==== Setup FastAPI app ====
app = FastAPI()

@app.get("/api/test")
async def health_check():
    return {"status": "ok", "message": "Server is running"}

# Optional: kiểm tra model từ API
@app.get("/api/check-model")
async def check_model():
    if not os.path.exists(base_dir):
        raise HTTPException(status_code=500, detail="No training folder found")
    files = os.listdir(base_dir)
    if not files:
        raise HTTPException(status_code=500, detail="No model checkpoint found")
    return {"status": "ok", "files": files}

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==== Define models ====
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]
    is_new_game: bool

class AIResponse(BaseModel):
    move: int

# ==== Main API Endpoint ====
@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # Create game position
        initial_pos = Pos.from_2d_board(game_state.board)

        # Setup MCTS
        mcts = MCTS(initial_pos, nn, c_exploration=1.5, c_ply_penalty=0.01)

        # Run MCTS and measure time
        start_time = time.time()
        mcts.run(10)
        policy = mcts.get_root_policy()
        move = np.argmax(policy)
        elapsed_time = time.time() - start_time
        print(f"⏱️ MCTS move selection took {elapsed_time:.2f} seconds")

        # Validate move
        if move not in game_state.valid_moves:
            print(f"⚠️ Suggested move {move} is invalid. Picking first valid move instead.")
            move = game_state.valid_moves[0]

        return AIResponse(move=move)

    except Exception as e:
        print(f"❌ Error during move selection: {str(e)}")
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

# ==== Run server locally (for dev) ====
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
