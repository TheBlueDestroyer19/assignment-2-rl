# Assignment 2: Monte Carlo, TD Prediction and Value Function Approximation

A minimal implementation of all three assignment parts using only NumPy and Matplotlib (plus Python's standard library).

## Setup

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Run all experiments from the project root:

```bash
python -m unittest tests.test_gridworld
python -m experiments.part1_prediction
python -m experiments.part2_control
python -m experiments.part3_approximation
```

## Environment assumptions

The assignment asks for a GridWorld with suitable rewards but does not specify its exact dimensions or reward values. This implementation uses:
- 4x4 deterministic grid
- Start: state 0 (top-left)
- Goal: state 15 (bottom-right)
- Actions: 0=up, 1=right, 2=down, 3=left
- Reward -1 for each non-terminal transition, +10 for entering the goal
- Discount factor gamma=0.9
- Episodes end on reaching the goal

The policy for prediction in Part 1 is uniform random. For control, Sarsa and Q-learning use epsilon-greedy exploration with epsilon=0.1. Experiments use 50 seeds and plot the first 500 episodes.

## Parts

### Part 1
Runs:
- First-visit MC with constant alpha
- First-visit MC with incremental mean
- Every-visit MC with constant alpha
- Every-visit MC with incremental mean
- TD(0)

The reference value function is calculated by iterative policy evaluation for the uniform random policy. RMS error is averaged across 50 independent runs.

### Part 2
Runs Sarsa and Q-learning with epsilon=0.1, averages steps-per-episode across 50 runs, and saves a greedy-policy visualization for each algorithm.

### Part 3
Uses a linear value function `V_hat(s,w) = w dot x(s)` with shared weights. Both algorithms replay the same fixed shortest-path trajectory for 100 episodes and preserve learned weights across episodes. They start with identical zero weights. A deterministic shortest-path policy's value function is used as the reference for MAE; feature choices and this reference are documented assumptions.

## Outputs

All outputs are created automatically:
- `results/figures/part1_rms.png`
- `results/tables/part1_final_rms.csv`
- `results/figures/part2_learning_curves.png`
- `results/figures/part2_greedy_policies.png`
- `results/tables/part2_final_steps.csv`
- `results/figures/part3_mae.png`
- `results/tables/part3_estimates.csv`

The Part 1 and Part 2 experiments use 50 runs. If you want a quick smoke test during development, temporarily reduce `NUM_RUNS` or `NUM_EPISODES` in the corresponding experiment file, then restore the assignment settings before finalizing.

## Notes
- Results are stochastic; exact curves and final values can vary slightly with implementation changes.
- The 500-step limit is used in Part 2 to prevent unusually long exploratory episodes. In Part 1 and Part 3, episodes/trajectory handling is explicit.
- No Gym, Gymnasium, RL packages, or prebuilt GridWorld implementation is used.
