import numpy as np


def generate_episode(env, rng, max_steps=None):
    """Generate an episode under the uniform random policy.

    Each transition is (state, action, reward, next_state, done).
    Returns (episode, terminated). If max_steps is reached first,
    terminated is False.
    """
    state = env.reset()
    episode = []

    for _ in range(max_steps if max_steps is not None else 10**9):
        action = int(rng.integers(env.n_actions))
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            return episode, True

    return episode, False


def calculate_returns(episode, gamma=0.9):
    """Calculate discounted returns from a complete episode."""
    returns = np.zeros(len(episode), dtype=float)
    G = 0.0
    for t in range(len(episode) - 1, -1, -1):
        G = episode[t][2] + gamma * G
        returns[t] = G
    return returns


def mc_prediction(V, counts, episode, returns, alpha=0.1,
                  first_visit=True, constant_alpha=True):
    """Apply first-visit/every-visit MC updates in-place."""
    seen = set()
    for t, transition in enumerate(episode):
        state = transition[0]
        if first_visit and state in seen:
            continue
        seen.add(state)

        target = returns[t]
        if constant_alpha:
            V[state] += alpha * (target - V[state])
        else:
            counts[state] += 1
            V[state] += (target - V[state]) / counts[state]
    return V
