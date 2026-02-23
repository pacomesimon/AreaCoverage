# Logic Module
This directory contains the core business logic for the coverage optimization application.

## Contents
- `engine.py`: The dispatcher that runs the selected optimization algorithm.
- `dqn_logic.py`: Deep Q-Learning implementation using an MLP to optimize sensor placement sequentially.
- `shapes.py`: Mathematical masks for different sensor radiation patterns (Antennas, FOVs, etc.).
- `__init__.py`: Makes this directory a Python package.

## Optimization Methods
1. **Monte Carlo**: Randomly places sensors and keeps the best configuration. Good for broad explorations.
2. **Deep Q-Learning (DQN)**: Uses a Multi-Layer Perceptron (MLP) that takes the current positions/angles as state and learns to predict incremental shifts (actions) to maximize total area coverage.
