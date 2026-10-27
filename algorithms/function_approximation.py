import numpy as np


def features(state, size=4):
    """Linear features: bias, normalized row, normalized column.

    The same weights are shared by all states.
    """
    row, col = divmod(state, size)
    scale = max(size - 1, 1)
    return np.array([1.0, row / scale, col / scale], dtype=float)


def value(state, weights, size=4):
    return float(np.dot(weights, features(state, size)))


def mc_update_episode(weights, trajectory, gamma=0.9, alpha=0.01, size=4):
    """One MC gradient update per visited time step, using full returns."""
    returns = np.zeros(len(trajectory), dtype=float)
    G = 0.0
    for t in range(len(trajectory) - 1, -1, -1):
        G = trajectory[t][2] + gamma * G
        returns[t] = G

    for t, (state, _, _) in enumerate(trajectory):
        x = features(state, size)
        error = returns[t] - np.dot(weights, x)
        weights += alpha * error * x
    return weights


def td0_update_episode(weights, trajectory, gamma=0.9, alpha=0.01, size=4):
    """Replay a fixed trajectory and update weights after each transition."""
    for state, next_state, reward in trajectory:
        x = features(state, size)
        current = np.dot(weights, x)
        next_value = 0.0 if next_state is None else np.dot(weights, features(next_state, size))
        target = reward + gamma * next_value
        weights += alpha * (target - current) * x
    return weights
