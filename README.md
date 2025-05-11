# Connect4-AlphaZero

An implementation of AlphaZero for Connect4 using Python and Rust.

## Overview

This project implements the AlphaZero algorithm to learn and play Connect4 through self-play reinforcement learning. It combines deep neural networks with Monte Carlo Tree Search (MCTS) to develop a strong Connect4 agent without human knowledge.

## Features

- **Self-play training**: The agent improves by playing against itself
- **Monte Carlo Tree Search**: Efficient game tree exploration
- **Neural Network**: Policy and value networks to evaluate positions
- **Hybrid Architecture**: Python for ML components and Rust for performance-critical game logic
- **Hyperparameter Optimization**: Optuna-based tuning for MCTS and neural network parameters
- **Tournament Evaluation**: Compare different model versions and players
- **Optional Connect4 Solver Integration**: Perfect play evaluation using an external solver

## Requirements

- Python 3.11+
- Rust (for the performance-critical components)
- PyTorch
- CUDA-compatible GPU (recommended for faster training)

## Installation

1. Clone this repository
2. Install clang
   ```
   # Instructions for Ubuntu/Debian (other OSs may vary)
   sudo apt install clang
   ```
3. Install uv for python dep/env management
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
4. Install deps and create virtual env:
   ```
   uv sync
   ```
5. Compile rust code
   ```
   uv run maturin develop --release
   ```
6. (Optional) Download a connect four solver to objectively measure training progress:
   ```
   ugit clone https://github.com/PascalPons/connect4.git solver
   cd solver
   make
    # Download opening book to speed up solutions
   wget https://github.com/PascalPons/connect4/releases/download/book/7x6.book
   ```

## Usage

### Training a Model

```bash
   uv run src/c4a0/main.py train --max-gens=10
```

### Playing Against a Trained Model

```bash
 uv run src/c4a0/main.py play --model=best
```

### Test for api which can be used to compare different model versions and players in tourament web

```bash
uv run src/c4a0/main.py debug2api --model=best
```

### Hyperparameter Optimization

```bash
# Neural network parameter sweep
python -m c4a0.main nn_sweep --base-dir training

# MCTS parameter sweep
python -m c4a0.main mcts_sweep --base-dir training-sweeps
```

### Evaluating Model Strength using connect4 solver

```bash
   uv run python src/c4a0/main.py score solver/c4solver solver/7x6.book
```

## Project Structure

- `src/c4a0/`: Python source code
  - `main.py`: CLI entry points and MCTS implementation
  - `nn.py`: Neural network architecture
  - `pos.py`: Game position representation
  - `training.py`: Self-play training loop
  - `tournament.py`: Model evaluation through tournaments
  - `sweep.py`: Hyperparameter optimization
- `rust/`: Rust implementation for performance-critical components

## License

See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by DeepMind's AlphaZero algorithm as described in their paper "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm".
