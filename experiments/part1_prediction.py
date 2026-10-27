import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.gridworld import GridWorld
from algorithms.monte_carlo import generate_episode, calculate_returns, mc_prediction
from algorithms.td_prediction import td0_episode

GAMMA, ALPHA = 0.9, 0.1
NUM_RUNS, NUM_EPISODES = 50, 10_000
ROOT = Path(__file__).resolve().parents[1]
FIGURES, TABLES = ROOT / "results/figures", ROOT / "results/tables"



def exact_random_policy_values(env, gamma=GAMMA, tolerance=1e-12):
    """Evaluate the uniform random policy, including terminal holes."""
    V = np.zeros(env.n_states)

    while True:
        old = V.copy()

        for state in range(env.n_states):
            if state in env.terminal_states:
                V[state] = 0.0
                continue

            total = 0.0

            for action in env.actions:
                env.state = state
                env.done = False

                next_state, reward, done = env.step(action)

                target = reward
                if not done:
                    target += gamma * old[next_state]

                total += target / env.n_actions

            V[state] = total

        if np.max(np.abs(V - old)) < tolerance:
            break

    env.reset()
    return V



def run_mc(method, reference, seed):
    env, rng = GridWorld(), np.random.default_rng(seed)
    V = np.zeros(env.n_states)
    counts = np.zeros(env.n_states, dtype=int)
    curve = np.zeros(NUM_EPISODES)
    for i in range(NUM_EPISODES):
        episode, terminated = generate_episode(env, rng)
        returns = calculate_returns(episode, GAMMA)
        settings = {
            "First-visit MC (constant alpha)": (True, True),
            "First-visit MC (incremental mean)": (True, False),
            "Every-visit MC (constant alpha)": (False, True),
            "Every-visit MC (incremental mean)": (False, False),
        }
        first, constant = settings[method]
        mc_prediction(V, counts, episode, returns, ALPHA, first, constant)
        curve[i] = np.sqrt(np.mean((V - reference) ** 2))
    return curve


def run_td(reference, seed):
    env, rng = GridWorld(), np.random.default_rng(seed)
    V = np.zeros(env.n_states)
    curve = np.zeros(NUM_EPISODES)
    for i in range(NUM_EPISODES):
        td0_episode(env, V, rng, ALPHA, GAMMA)
        curve[i] = np.sqrt(np.mean((V - reference) ** 2))
    return curve


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    reference = exact_random_policy_values(GridWorld())
    labels = [
        "First-visit MC (constant alpha)",
        "First-visit MC (incremental mean)",
        "Every-visit MC (constant alpha)",
        "Every-visit MC (incremental mean)",
        "TD(0)",
    ]
    curves = {}
    for label in labels:
        print("Running", label)
        runs = []
        for seed in range(NUM_RUNS):
            curve = run_td(reference, seed) if label == "TD(0)" else run_mc(label, reference, seed)
            runs.append(curve)
        curves[label] = np.mean(runs, axis=0)
        print(f"  Final average RMS error: {curves[label][-1]:.6f}")

    plt.figure(figsize=(9, 5))
    for label, curve in curves.items():
        plt.plot(curve, label=label)
    plt.xlabel("Episode")
    plt.ylabel("RMS error")
    plt.title("Monte Carlo vs TD(0) Prediction")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / "part1_rms.png", dpi=150)
    plt.close()

    with open(TABLES / "part1_final_rms.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Method", "Final average RMS error"])
        for label in labels:
            writer.writerow([label, curves[label][-1]])
    print("Saved Part 1 graph and table in results/.")


if __name__ == "__main__":
    main()
