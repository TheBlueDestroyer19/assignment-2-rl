import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.gridworld import GridWorld
from algorithms.sarsa import sarsa_episode
from algorithms.q_learning import q_learning_episode

NUM_RUNS, NUM_EPISODES = 50, 500
ALPHA, GAMMA, EPSILON = 0.1, 0.9, 0.1
MAX_STEPS = 500
ROOT = Path(__file__).resolve().parents[1]
FIGURES, TABLES = ROOT / "results/figures", ROOT / "results/tables"


def train(algorithm, seed):
    env = GridWorld()
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    steps = np.zeros(NUM_EPISODES)

    for ep in range(NUM_EPISODES):
        if algorithm == "Sarsa":
            n_steps, reached = sarsa_episode(
                env, Q, rng, ALPHA, GAMMA, EPSILON, MAX_STEPS
            )
        else:
            n_steps, reached = q_learning_episode(
                env, Q, rng, ALPHA, GAMMA, EPSILON, MAX_STEPS
            )
        steps[ep] = n_steps

    return steps, Q



def save_policy(Q, filename, title):
    env = GridWorld()
    arrows = ("↑", "→", "↓", "←")
    policy = np.argmax(Q, axis=1)

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks(range(env.size + 1))
    ax.set_yticks(range(env.size + 1))
    ax.grid(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)

    for state in range(env.n_states):
        row, col = divmod(state, env.size)
        x = col + 0.5
        y = env.size - row - 0.5

        if state == env.start:
            label = "S"
        elif state == env.goal:
            label = "G"
        elif state in env.holes:
            label = "H"
        else:
            label = arrows[policy[state]]

        ax.text(
            x, y, label,
            ha="center",
            va="center",
            fontsize=16
        )

    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)



def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    curves, final_q = {}, {}
    for algorithm in ("Sarsa", "Q-learning"):
        print("Running", algorithm)
        all_steps = []
        q_tables = []
        for seed in range(NUM_RUNS):
            steps, Q = train(algorithm, seed)
            all_steps.append(steps)
            q_tables.append(Q)
        curves[algorithm] = np.mean(all_steps, axis=0)
        # The displayed policy is from the final seeded run, not an average Q-table.
        final_q[algorithm] = q_tables[-1]
        print(f"  Mean steps over last 50 episodes: {np.mean(curves[algorithm][-50:]):.2f}")

    plt.figure(figsize=(8, 5))
    for algorithm, curve in curves.items():
        # Light smoothing makes the overall learning trend easier to see.
        window = 10
        smoothed = np.convolve(curve, np.ones(window) / window, mode="valid")
        plt.plot(np.arange(window, NUM_EPISODES + 1), smoothed, label=algorithm)
    plt.xlabel("Episode")
    plt.ylabel("Average steps per episode (10-episode moving average)")
    plt.title("Sarsa vs Q-learning (50 runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "part2_learning_curves.png", dpi=150)
    plt.close()

    save_policy(final_q["Sarsa"], FIGURES / "part2_sarsa_policy.png", "Sarsa greedy policy")
    save_policy(final_q["Q-learning"], FIGURES / "part2_q_learning_policy.png", "Q-learning greedy policy")

    with open(TABLES / "part2_final_steps.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Mean steps over final 50 episodes"])
        for algorithm, curve in curves.items():
            writer.writerow([algorithm, float(np.mean(curve[-50:]))])
    print("Saved Part 2 figures and table in results/.")


if __name__ == "__main__":
    main()
