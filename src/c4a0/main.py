#!/usr/bin/env python

from pathlib import Path
import sys

from typing import List, Optional, Dict
import warnings
import numpy as np
from loguru import logger
import optuna
import torch
import typer
import math
import random

from typing import List, Optional

# Ensure that the parent directory of this file exists on Python path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from c4a0.pos import Pos
from c4a0.nn import ModelConfig  # noqa: E402
from c4a0.sweep import perform_hparam_sweep  # noqa: E402
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer  # noqa: E402
from c4a0.training import (  # noqa: E402
    SolverConfig,
    TrainingGen,
    parse_lr_schedule,
    training_loop,
)
from c4a0.utils import get_torch_device  # noqa: E402

import c4a0_rust  # noqa: E402

app = typer.Typer()


@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    # These parameters were chosen based on the results of the nn_sweep and mcts_sweep
    n_self_play_games: int = 1700,
    n_mcts_iterations: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3, 10, 8e-4],
    l2_reg: float = 4e-4,
    max_gens: Optional[int] = None,
    solver_path: Optional[str] = None,
    book_path: Optional[str] = None,
    solutions_path: str = "./solutions.db",
):
    """Trains a model via self-play."""

    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    if solver_path and book_path:
        logger.info("Using solver")
        solver_config = SolverConfig(
            solver_path=solver_path,
            book_path=book_path,
            solutions_path=solutions_path,
        )
    else:
        logger.info("Solver not provided, skipping solutions")
        solver_config = None

    training_loop(
        base_dir=base_dir,
        device=torch.device(device),
        n_self_play_games=n_self_play_games,
        n_mcts_iterations=n_mcts_iterations,
        c_exploration=c_exploration,
        c_ply_penalty=c_ply_penalty,
        self_play_batch_size=self_play_batch_size,
        training_batch_size=training_batch_size,
        model_config=model_config,
        max_gens=max_gens,
        solver_config=solver_config,
    )

class Node:
    EPS = 1e-8

    def __init__(self, pos, parent=None, initial_policy_value=0):
        """Khởi tạo nút với trạng thái Pos, nút cha, và giá trị chính sách ban đầu."""
        self.pos = pos
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.q_sum_penalty = 0.0  # Tổng giá trị Q có phạt ply
        self.q_sum_no_penalty = 0.0  # Tổng giá trị Q không phạt ply
        self.initial_policy_value = initial_policy_value

    def q_with_penalty(self):
        """Tính giá trị Q trung bình có phạt ply."""
        return self.q_sum_penalty / (self.visit_count + self.EPS)

    def q_no_penalty(self):
        """Tính giá trị Q trung bình không phạt ply."""
        return self.q_sum_no_penalty / (self.visit_count + self.EPS)

    def exploration_value(self):
        """Tính giá trị khám phá dựa trên UCT."""
        parent_visits = self.parent.visit_count if self.parent else self.visit_count
        exploration = math.sqrt(math.log(parent_visits + 1) / (self.visit_count + self.EPS))
        return exploration * (self.initial_policy_value + self.EPS)

    def uct_value(self, c_exploration):
        """Tính giá trị UCT để chọn nút con tốt nhất."""
        return -self.q_with_penalty() + c_exploration * self.exploration_value()

    def is_terminal(self):
        """Kiểm tra xem nút có phải là trạng thái kết thúc không."""
        return self.pos.is_terminal_state() is not None

    def policy(self):
        """Trả về chính sách dựa trên số lần thăm các nút con."""
        if self.is_terminal():
            return [0.0] * 7
        if not self.children:
            return [1.0 / 7] * 7  # Chính sách đồng đều nếu chưa mở rộng
        child_visits = [self.children.get(a, Node(self.pos)).visit_count for a in range(7)]
        total_visits = sum(child_visits)
        if total_visits == 0:
            return [1.0 / 7] * 7
        return [v / total_visits for v in child_visits]

class MCTS:
    def __init__(self, root_pos, model, c_exploration=1.5, c_ply_penalty=0.01):
        """Khởi tạo MCTS với trạng thái gốc, mô hình, và các hằng số."""
        self.root = Node(root_pos)
        self.leaf = self.root
        self.model = model
        self.c_exploration = c_exploration
        self.c_ply_penalty = c_ply_penalty

    def select_leaf(self):
        """Chọn nút lá để mở rộng hoặc đánh giá."""
        current = self.root
        while current.children and not current.is_terminal():
            current = max(current.children.values(), key=lambda child: child.uct_value(self.c_exploration))
        self.leaf = current

    def expand_leaf(self, policy_probs):
        """Mở rộng nút lá với các nước đi hợp lệ."""
        if self.leaf.is_terminal():
            return
        legal_actions = [a for a, is_legal in enumerate(self.leaf.pos.legal_moves()) if is_legal]
        for action in legal_actions:
            new_pos = self.leaf.pos.make_move(action)
            if new_pos is not None:
                child = Node(new_pos, parent=self.leaf, initial_policy_value=policy_probs[action])
                self.leaf.children[action] = child

    def backpropagate(self, q_penalty, q_no_penalty):
        """Lan truyền ngược giá trị Q từ lá lên gốc."""
        current = self.leaf
        while current:
            current.visit_count += 1
            current.q_sum_penalty += q_penalty
            current.q_sum_no_penalty += q_no_penalty
            q_penalty = -q_penalty  # Đảo dấu cho người chơi đối thủ
            q_no_penalty = -q_no_penalty
            current = current.parent

    def on_received_policy(self, policy_logits, q_penalty, q_no_penalty):
        """Xử lý kết quả từ mô hình và cập nhật cây."""
        if self.leaf.is_terminal():
            terminal_values = self.leaf.pos.terminal_value_with_ply_penalty(self.c_ply_penalty)
            if terminal_values is not None:
                q_penalty, q_no_penalty = terminal_values
                self.backpropagate(q_penalty, q_no_penalty)
        else:
            policy_probs = self.softmax(policy_logits)
            self.expand_leaf(policy_probs)
            self.backpropagate(q_penalty, q_no_penalty)

    def run(self, num_iterations):
        """Chạy MCTS với số vòng lặp chỉ định."""
        for _ in range(num_iterations):
            self.select_leaf()
            input_tensor = self.leaf.pos.write_numpy_buffer()
            policy_logits, q_penalty, q_no_penalty = self.model.forward_numpy(input_tensor)
            policy_logits = policy_logits[0]
            self.on_received_policy(policy_logits, q_penalty, q_no_penalty)

    def get_root_policy(self):
        """Trả về chính sách của nút gốc."""
        return self.root.policy()
    @staticmethod
    def softmax(logits):
        """Chuyển logits thành xác suất bằng softmax."""
        logits = logits - np.max(logits)  # Tránh tràn số
        e = np.exp(logits)
        return e / e.sum()


@app.command()
def debug2api(model: str = "best", base_dir: str = "training"):
    gen = TrainingGen.load_latest(base_dir)
    print("gen", gen)
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    board4 = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0],
        [0, 2, 0, 1, 0, 0, 0],
        [0, 2, 0, 1, 0, 0, 1],
    ]
    print("Start")
    initial_pos = Pos.from_2d_board(board4)
    mcts = MCTS(initial_pos, nn, c_exploration=1.5, c_ply_penalty=0.01)
    mcts.run(1300)
    # Lấy chính sách và dự đoán nước đi tốt nhất
    policy = mcts.get_root_policy()
    print("Policy:", policy)
    best_move = np.argmax(policy)
    print("BestMove", best_move)
    
# uv run src/c4a0/main.py debug --model=best


@app.command()
def play(
    base_dir: str = "training",
    max_mcts_iters: int = 1400,
    c_exploration: float = 6.6,
    c_ply_penalty: float = 0.01,
    model: str = "best",
):
    """Play interactive games"""
    gen = TrainingGen.load_latest(base_dir)
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    c4a0_rust.run_tui(  # type: ignore
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
    )


@app.command()
def nn_sweep(base_dir: str = "training"):
    """
    Performs a hyperparameter sweep to determine best nn model params based on existing training
    data.
    """
    perform_hparam_sweep(base_dir)


@app.command()
def mcts_sweep(
    device: str = str(get_torch_device()),
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    # These NN parameters were chosen based on the results of the nn_sweep
    n_residual_blocks: int = 1,
    conv_filter_size: int = 32,
    n_policy_layers: int = 4,
    n_value_layers: int = 2,
    lr_schedule: List[float] = [0, 2e-3],
    l2_reg: float = 4e-4,
    # End NN parameters
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    solutions_path: str = "./solutions.db",
):
    """
    Performs sweep of MCTS hyperparameters (e.g. n_self_play_games, n_mcts_iterations,
    c_exploration) to determine optimal values by performing `n_trials` independent training
    runs, each with `max_gens_per_trial` generations, seeking to maximize the solver score.
    """
    base_path = Path(base_training_dir)
    base_path.mkdir(exist_ok=True)

    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    def objective(trial: optuna.Trial):
        trial_path = base_path / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        gen = training_loop(
            base_dir=str(trial_path),
            device=torch.device(device),
            n_self_play_games=trial.suggest_int("n_self_play_games", 1000, 5000),
            n_mcts_iterations=trial.suggest_int("n_mcts_iterations", 100, 1500),
            c_exploration=trial.suggest_float("c_exploration", 0.5, 12.0),
            c_ply_penalty=c_ply_penalty,
            self_play_batch_size=self_play_batch_size,
            training_batch_size=training_batch_size,
            model_config=model_config,
            max_gens=max_gens_per_trial,
            solver_config=SolverConfig(
                solver_path=solver_path,
                book_path=book_path,
                solutions_path=solutions_path,
            ),
        )
        logger.info(
            "Trial {} completed. Solver score: {}", trial.number, gen.solver_score
        )
        score = gen.solver_score
        assert score is not None
        return score

    storage_name = f"sqlite:///{optuna_db_path}"
    study = optuna.create_study(
        study_name="mcts_sweep",
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)


@app.command()
def score(
    solver_path: str,
    book_path: str,
    base_dir: str = "training",
    solutions_path: str = "./solutions.db",
):
    """Scores the training generations using the given solver."""
    gens = TrainingGen.load_all(base_dir)
    for gen in gens:
        logger.info("Getting games for: {}", gen.gen_n)
        games = gen.get_games(base_dir)  # type: ignore
        if not games:
            continue
        if gen.solver_score is not None:
            logger.info(f"Gen already has score: {gen.solver_score}")
            continue
        score = games.score_policies(solver_path, book_path, solutions_path)  # type: ignore
        gen.solver_score = score
        gen.save_metadata(base_dir)
        logger.info("Gen {} has score: {}", gen.gen_n, score)


if __name__ == "__main__":
    # Disable unnecessary pytorch warnings
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    app()
