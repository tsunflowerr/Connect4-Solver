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
2. Install Python dependencies:
   ```
   pip install -e .
   ```
3. Build the Rust components:
   ```
   cd rust
   cargo build --release
   ```

## Usage

### Training a Model

```bash
python -m c4a0.main train --base-dir training --n-self-play-games 1700 --n-mcts-iterations 1400
```

### Playing Against a Trained Model

```bash
python -m c4a0.main play --base-dir training --model best
```

### Hyperparameter Optimization

```bash
# Neural network parameter sweep
python -m c4a0.main nn_sweep --base-dir training

# MCTS parameter sweep
python -m c4a0.main mcts_sweep --base-dir training-sweeps
```

### Evaluating Model Strength

```bash
python -m c4a0.main score --solver-path /path/to/solver --book-path /path/to/book
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
