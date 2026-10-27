import numpy as np
from algorithms.sarsa import epsilon_greedy


def q_learning_episode(env, Q, rng, alpha=0.1, gamma=0.9,
                       epsilon=0.1, max_steps=500):
    state = env.reset()

    for step in range(1, max_steps + 1):
        action = epsilon_greedy(Q, state, epsilon, rng)
        next_state, reward, done = env.step(action)

        target = reward
        if not done:
            target += gamma * np.max(Q[next_state])

        Q[state, action] += alpha * (target - Q[state, action])
        state = next_state

        if done:
            return step, True

    return max_steps, False
