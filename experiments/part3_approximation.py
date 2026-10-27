import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from environment.gridworld import GridWorld
from algorithms.function_approximation import (
    features, value, mc_update_episode, td0_update_episode
)

GAMMA, ALPHA, NUM_EPISODES = 0.9, 0.01, 100
CHECKPOINTS = {1, 10, 25, 50, 100}
ROOT = Path(__file__).resolve().parents[1]
FIGURES, TABLES = ROOT / "results/figures", ROOT / "results/tables"

def shortest_safe_trajectory(env):
    """Find a shortest path to the goal without entering holes."""
    queue = deque([env.start])
    previous = {env.start: None}
    previous_action = {}

    while queue:
        state = queue.popleft()

        if state == env.goal:
            break

        row, col = divmod(state, env.size)

        for action, (dr, dc) in enumerate(env.moves):
            nr = max(0, min(env.size - 1, row + dr))
            nc = max(0, min(env.size - 1, col + dc))
            next_state = nr * env.size + nc

            if next_state in env.holes or next_state in previous:
                continue

            previous[next_state] = state
            previous_action[next_state] = action
            queue.append(next_state)

    if env.goal not in previous:
        raise RuntimeError("No safe path to the goal.")

    states = []
    current = env.goal

    while current is not None:
        states.append(current)
        current = previous[current]

    states.reverse()
    trajectory = []

    env.reset()

    for state, next_state in zip(states[:-1], states[1:]):
        env.state = state
        env.done = False

        action = previous_action[next_state]
        actual_next, reward, done = env.step(action)

        trajectory.append(
            (state, None if done else actual_next, reward)
        )

    env.reset()
    return trajectory

def safe_policy_reference(env, gamma=GAMMA):
    """Calculate reference values for a deterministic safe policy."""
    distances = {env.goal: 0}
    queue = deque([env.goal])

    while queue:
        state = queue.popleft()
        row, col = divmod(state, env.size)

        for dr, dc in env.moves:
            nr = row + dr
            nc = col + dc

            if not (0 <= nr < env.size and 0 <= nc < env.size):
                continue

            next_state = nr * env.size + nc

            if next_state in env.holes or next_state in distances:
                continue

            distances[next_state] = distances[state] + 1
            queue.append(next_state)

    policy = {}

    for state in range(env.n_states):
        if state in env.terminal_states or state not in distances:
            continue

        row, col = divmod(state, env.size)

        for action, (dr, dc) in enumerate(env.moves):
            nr = max(0, min(env.size - 1, row + dr))
            nc = max(0, min(env.size - 1, col + dc))
            next_state = nr * env.size + nc

            if (
                next_state not in env.holes
                and next_state in distances
                and distances[next_state] < distances[state]
            ):
                policy[state] = action
                break

    V = np.zeros(env.n_states)

    for _ in range(1000):
        old = V.copy()

        for state in range(env.n_states):
            if state in env.terminal_states:
                V[state] = 0.0
                continue

            if state not in policy:
                continue

            env.state = state
            env.done = False

            next_state, reward, done = env.step(policy[state])

            V[state] = (
                reward if done
                else reward + gamma * old[next_state]
            )

        if np.max(np.abs(V - old)) < 1e-12:
            break

    env.reset()
    return V

def fixed_shortest_path_trajectory(env):
    """A fixed path: right across the top row, then down to the goal.

    Returns (state, next_state, reward) tuples. Both algorithms replay exactly
    this same sequence and reward sequence on every episode.
    """
    state = env.reset()
    trajectory = []
    actions = [1] * (env.size - 1) + [2] * (env.size - 1)
    for action in actions:
        next_state, reward, done = env.step(action)
        trajectory.append((state, None if done else next_state, reward))
        state = next_state
    return trajectory


def deterministic_shortest_path_reference(env, gamma=GAMMA):
    """Exact values under a policy that moves right, then down toward goal."""
    values = np.zeros(env.n_states)
    for state in range(env.goal - 1, -1, -1):
        row, col = divmod(state, env.size)
        if state == env.goal:
            values[state] = 0.0
        else:
            # Policy moves right until last column, then down.
            action = 1 if col < env.size - 1 else 2
            env.state, env.done = state, False
            next_state, reward, done = env.step(action)
            values[state] = reward if done else reward + gamma * values[next_state]
    values[env.goal] = 0.0
    env.reset()
    return values


def predictions(weights, env):
    return np.array([value(s, weights, env.size) for s in range(env.n_states)])


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    env = GridWorld()
    trajectory = fixed_shortest_path_trajectory(env)
    reference = deterministic_shortest_path_reference(env)

    # Both algorithms start with exactly the same weights and retain them
    # across episodes. Do not reset weights inside the training loop.
    mc_weights = np.zeros(3, dtype=float)
    td_weights = np.zeros(3, dtype=float)
    mc_mae, td_mae = [], []
    rows = []

    for episode in range(1, NUM_EPISODES + 1):
        mc_update_episode(mc_weights, trajectory, GAMMA, ALPHA, env.size)
        td0_update_episode(td_weights, trajectory, GAMMA, ALPHA, env.size)

        mc_pred = predictions(mc_weights, env)
        td_pred = predictions(td_weights, env)

        # Exclude terminal goal from MAE because its terminal value is defined as 0.
        nonterminal = np.arange(env.n_states) != env.goal
        mc_mae.append(float(np.mean(np.abs(mc_pred[nonterminal] - reference[nonterminal]))))
        td_mae.append(float(np.mean(np.abs(td_pred[nonterminal] - reference[nonterminal]))))

        if episode in CHECKPOINTS:
            # Estimates for all states are retained in the table.
            rows.append([episode, "MC", *mc_pred.tolist()])
            rows.append([episode, "TD(0)", *td_pred.tolist()])

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPISODES + 1), mc_mae, label="MC with approximation")
    plt.plot(range(1, NUM_EPISODES + 1), td_mae, label="TD(0) with approximation")
    plt.xlabel("Episode")
    plt.ylabel("MAE against deterministic-policy reference")
    plt.title("Value Function Approximation: MC vs TD(0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "part3_mae.png", dpi=150)
    plt.close()

    with open(TABLES / "part3_estimates.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Algorithm"] + [f"V_state_{s}" for s in range(env.n_states)])
        writer.writerows(rows)

    print("Part 3 completed.")
    print("Fixed trajectory states:", [trajectory[0][0]] + [t[1] for t in trajectory[:-1]] + [env.goal])
    print("Final MC MAE:", f"{mc_mae[-1]:.6f}")
    print("Final TD MAE:", f"{td_mae[-1]:.6f}")
    print("Saved Part 3 figure and table in results/.")


if __name__ == "__main__":
    main()
