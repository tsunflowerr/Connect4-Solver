import sys
import os
from pathlib import Path
import logging
from typing import List, Optional, Dict
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Add project root to PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

from c4a0.training import TrainingGen
from c4a0.utils import get_torch_device
from c4a0.nn import ConnectFourNet

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
BASE_DIR = "C:/Users/MSi/Desktop/python/Connect4-Solver-main_promax/Connect4-Solver-main/src/training"
DEVICE = get_torch_device()
MCTS_ITERS = 2000  # Number of MCTS iterations
C_EXPLORATION = 5.0  # Exploration constant for UCB
N_COLS = 7
N_ROWS = 6

# Load model
model = None
latest_gen = None
try:
    if not os.path.exists(BASE_DIR):
        logger.warning(f"Training directory '{BASE_DIR}' not found, creating it")
        os.makedirs(BASE_DIR)
    latest_gen = TrainingGen.load_latest(BASE_DIR)
    model = latest_gen.get_model(BASE_DIR)
    model.to(DEVICE)
    model.eval()
    logger.info(f"Loaded model from generation {latest_gen.gen_n}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    logger.info("API will return random valid move if no model is loaded")

# Pydantic models
class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int
    confidence: Optional[float] = None

# MCTS Node
class MCTSNode:
    def __init__(self, state: np.ndarray, player: int, parent=None, move: Optional[int] = None, prior: float = 0.0):
        self.state = state  # Board state (2 x 6 x 7)
        self.player = player
        self.parent = parent
        self.move = move
        self.prior = prior  # Prior probability from NN
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visits = 0
        self.value_sum = 0.0
        self.valid_moves = self.get_valid_moves()

    def get_valid_moves(self) -> List[int]:
        """Get valid moves based on top row of the board."""
        valid = []
        for c in range(N_COLS):
            if self.state[0, 0, c] == 0 and self.state[1, 0, c] == 0:  # Top row empty
                valid.append(c)
        return valid

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """Check if the game is over (simplified check)."""
        return not self.valid_moves or self.check_win()

    def check_win(self) -> bool:
        """Check if the current state has a winner (simplified)."""
        board = self.state[0] - self.state[1]  # Player 1: 1, Player 2: -1
        for r in range(N_ROWS):
            for c in range(N_COLS - 3):
                if abs(sum(board[r, c:c+4])) == 4:
                    return True
        for c in range(N_COLS):
            for r in range(N_ROWS - 3):
                if abs(sum(board[r:r+4, c])) == 4:
                    return True
        for r in range(N_ROWS - 3):
            for c in range(N_COLS - 3):
                if abs(sum([board[r+i, c+i] for i in range(4)])) == 4:
                    return True
                if abs(sum([board[r+i, c+3-i] for i in range(4)])) == 4:
                    return True
        return False

    def ucb_score(self, c_exploration: float) -> float:
        """Calculate UCB score for node selection."""
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        return (self.value_sum / self.visits) + c_exploration * self.prior * (parent_visits ** 0.5) / (1 + self.visits)

# MCTS Search
class MCTS:
    def __init__(self, model: Optional[ConnectFourNet], mcts_iters: int, c_exploration: float):
        self.model = model
        self.mcts_iters = mcts_iters
        self.c_exploration = c_exploration

    def make_move(self, state: np.ndarray, player: int) -> np.ndarray:
        """Convert board to NN input format."""
        state_tensor = np.zeros((2, N_ROWS, N_COLS), dtype=np.float32)
        board = state[0] - state[1]  # Player 1: 1, Player 2: -1
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if board[r, c] == 1:
                    state_tensor[0, r, c] = 1
                elif board[r, c] == -1:
                    state_tensor[1, r, c] = 1
        return state_tensor

    def apply_move(self, state: np.ndarray, move: int, player: int) -> np.ndarray:
        """Apply a move to the board."""
        new_state = state.copy()
        for r in range(N_ROWS - 1, -1, -1):
            if new_state[0, r, move] == 0 and new_state[1, r, move] == 0:
                new_state[0 if player == 1 else 1, r, move] = 1
                break
        return new_state

    def search(self, root_state: np.ndarray, player: int) -> tuple[int, float]:
        """Run MCTS to select the best move."""
        root = MCTSNode(root_state, player)
        
        for _ in range(self.mcts_iters):
            node = root
            state = root_state.copy()

            # Selection
            while not node.is_leaf() and not node.is_terminal():
                move, node = max(node.children.items(), key=lambda x: x[1].ucb_score(self.c_exploration))
                state = self.apply_move(state, move, node.player)

            # Expansion
            if not node.is_terminal():
                state_tensor = self.make_move(state, node.player)
                state_tensor = state_tensor[np.newaxis, ...]
                
                if self.model:
                    with torch.no_grad():
                        policy_logits, q_penalty, _ = self.model.forward_numpy(state_tensor)
                        policy = np.exp(policy_logits[0])
                        value = q_penalty[0]
                else:
                    policy = np.ones(N_COLS) / N_COLS
                    value = 0.0

                for move in node.valid_moves:
                    if move not in node.children:
                        prior = policy[move]
                        new_state = self.apply_move(state, move, node.player)
                        next_player = 3 - node.player  # Switch player (1 -> 2, 2 -> 1)
                        node.children[move] = MCTSNode(new_state, next_player, node, move, prior)

                # Backpropagation
                value = float(value) if self.model else 0.0
                while node:
                    node.visits += 1
                    node.value_sum += value if node.player == player else -value
                    node = node.parent

        # Select move based on visit counts
        visits = [(move, child.visits) for move, child in root.children.items()]
        move, _ = max(visits, key=lambda x: x[1])
        confidence = root.children[move].visits / sum(child.visits for child in root.children.values())
        return move, confidence

# API Endpoints
@app.post("/api/connect4-move", response_model=AIResponse)
async def make_move(game_state: GameState) -> AIResponse:
    try:
        if not game_state.valid_moves:
            raise ValueError("No valid moves provided")

        # Convert board to numpy array
        raw_board = np.array(game_state.board)
        board = np.zeros((2, N_ROWS, N_COLS), dtype=np.float32)
        for r in range(N_ROWS):
            for c in range(N_COLS):
                if raw_board[r, c] == game_state.current_player:
                    board[0, r, c] = 1
                elif raw_board[r, c] != 0:
                    board[1, r, c] = 1

        # Run MCTS
        mcts = MCTS(model, MCTS_ITERS, C_EXPLORATION)
        selected_move, confidence = mcts.search(board, game_state.current_player)
        
        if selected_move not in game_state.valid_moves:
            logger.warning(f"MCTS selected invalid move {selected_move}, falling back to first valid move")
            selected_move = game_state.valid_moves[0]
            confidence = 0.0

        logger.info(f"Selected move {selected_move} with confidence {confidence:.4f}")
        return AIResponse(move=selected_move, confidence=confidence)

    except Exception as e:
        logger.error(f"Error computing move: {e}")
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Check API status."""
    if model:
        return {"status": "ok", "model_generation": getattr(latest_gen, "gen_n", "unknown")}
    return {"status": "degraded", "reason": "model not loaded"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Connect Four AI API...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
