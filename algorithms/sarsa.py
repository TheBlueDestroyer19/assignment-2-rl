import numpy as np


def epsilon_greedy(Q, state, epsilon, rng):
    """Choose a random action with probability epsilon, otherwise greedy."""
    if rng.random() < epsilon:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[state] == np.max(Q[state]))
    return int(rng.choice(best))


def sarsa_episode(env, Q, rng, alpha=0.1, gamma=0.9,
                  epsilon=0.1, max_steps=500):
    state = env.reset()
    action = epsilon_greedy(Q, state, epsilon, rng)

    for step in range(1, max_steps + 1):
        next_state, reward, done = env.step(action)

        if done:
            target = reward
            Q[state, action] += alpha * (target - Q[state, action])
            return step, True

        next_action = epsilon_greedy(Q, next_state, epsilon, rng)
        target = reward + gamma * Q[next_state, next_action]
        Q[state, action] += alpha * (target - Q[state, action])

        state, action = next_state, next_action

    return max_steps, False
