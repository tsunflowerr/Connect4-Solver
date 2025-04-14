#!/usr/bin/env python

from pathlib import Path
import sys
from typing import List, Optional
import warnings
from loguru import logger
import optuna
import torch
import typer

parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from c4a0.nn import ModelConfig
from c4a0.sweep import perform_hparam_sweep
from c4a0.tournament import ModelID, RandomPlayer, UniformPlayer
from c4a0.training import SolverConfig, TrainingGen, parse_lr_schedule, training_loop
from c4a0.utils import get_torch_device

import c4a0_rust  # type: ignore

app = typer.Typer()

@app.command()
def train(
    base_dir: str = "training",
    device: str = str(get_torch_device()),
    n_self_play_games: int = 3000,  # Increased from 1700
    n_mcts_iterations: int = 2000,  # Increased from 1400
    c_exploration: float = 5.0,     # Adjusted from 6.6
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 3,     # Increased from 1
    conv_filter_size: int = 64,     # Increased from 32
    n_policy_layers: int = 3,       # Adjusted from 4
    n_value_layers: int = 3,        # Increased from 2
    lr_schedule: List[float] = [0, 1e-3, 20, 5e-4, 40, 1e-4],  # More gradual decay
    l2_reg: float = 1e-4,           # Reduced from 4e-4
    label_smoothing: float = 0.1,   # New parameter
    max_gens: Optional[int] = None,
    solver_path: Optional[str] = None,
    book_path: Optional[str] = None,
    solutions_path: str = "./solutions.db",
):
    """Trains a model via self-play with improved defaults"""
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
        label_smoothing=label_smoothing,
    )

    solver_config = None
    if solver_path and book_path:
        logger.info("Using solver")
        solver_config = SolverConfig(
            solver_path=solver_path,
            book_path=book_path,
            solutions_path=solutions_path,
        )

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

@app.command()
def play(
    base_dir: str = "training",
    max_mcts_iters: int = 2000,  # Increased from 1400
    c_exploration: float = 5.0,  # Adjusted from 6.6
    c_ply_penalty: float = 0.01,
    model: str = "best",
):
    """Play interactive games with updated defaults"""
    gen = TrainingGen.load_latest(base_dir)
    if model == "best":
        nn = gen.get_model(base_dir)
    elif model == "random":
        nn = RandomPlayer(ModelID(0))
    elif model == "uniform":
        nn = UniformPlayer(ModelID(0))
    else:
        raise ValueError(f"unrecognized model: {model}")

    c4a0_rust.run_tui(
        lambda model_id, x: nn.forward_numpy(x),
        max_mcts_iters,
        c_exploration,
        c_ply_penalty,
    )

@app.command()
def nn_sweep(base_dir: str = "training"):
    """Performs neural network hyperparameter sweep"""
    perform_hparam_sweep(base_dir)

@app.command()
def mcts_sweep(
    device: str = str(get_torch_device()),
    c_ply_penalty: float = 0.01,
    self_play_batch_size: int = 2000,
    training_batch_size: int = 2000,
    n_residual_blocks: int = 3,      # Updated from 1
    conv_filter_size: int = 64,      # Updated from 32
    n_policy_layers: int = 3,        # Updated from 4
    n_value_layers: int = 3,         # Updated from 2
    lr_schedule: List[float] = [0, 1e-3, 20, 5e-4],
    l2_reg: float = 1e-4,            # Updated from 4e-4
    base_training_dir: str = "training-sweeps",
    optuna_db_path: str = "optuna.db",
    n_trials: int = 100,
    max_gens_per_trial: int = 10,
    solver_path: str = "/home/advait/connect4/c4solver",
    book_path: str = "/home/advait/connect4/7x6.book",
    solutions_path: str = "./solutions.db",
):
    """Performs MCTS hyperparameter sweep with updated defaults"""
    model_config = ModelConfig(
        n_residual_blocks=n_residual_blocks,
        conv_filter_size=conv_filter_size,
        n_policy_layers=n_policy_layers,
        n_value_layers=n_value_layers,
        lr_schedule=parse_lr_schedule(lr_schedule),
        l2_reg=l2_reg,
    )

    def objective(trial: optuna.Trial):
        trial_path = Path(base_training_dir) / f"trial_{trial.number}"
        trial_path.mkdir(exist_ok=False)
        
        return training_loop(
            base_dir=str(trial_path),
            device=torch.device(device),
            n_self_play_games=trial.suggest_int("n_self_play_games", 2000, 5000),
            n_mcts_iterations=trial.suggest_int("n_mcts_iterations", 1500, 2500),
            c_exploration=trial.suggest_float("c_exploration", 3.0, 8.0),
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
        ).solver_score

    study = optuna.create_study(
        storage=f"sqlite:///{optuna_db_path}",
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
    """Scores training generations using solver"""
    gens = TrainingGen.load_all(base_dir)
    for gen in gens:
        if games := gen.get_games(base_dir):
            if gen.solver_score is None:
                gen.solver_score = games.score_policies(solver_path, book_path, solutions_path)
                gen.save_metadata(base_dir)
                logger.info("Gen {} scored: {}", gen.gen_n, gen.solver_score)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    app()