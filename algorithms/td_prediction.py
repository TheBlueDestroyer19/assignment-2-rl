def td0_episode(env, V, rng, alpha=0.1, gamma=0.9, max_steps=None):
    """Run one TD(0) prediction episode under the uniform random policy."""
    state = env.reset()
    steps = 0

    while max_steps is None or steps < max_steps:
        action = int(rng.integers(env.n_actions))
        next_state, reward, done = env.step(action)

        target = reward
        if not done:
            target += gamma * V[next_state]
        V[state] += alpha * (target - V[state])

        state = next_state
        steps += 1
        if done:
            return steps, True

    return steps, False
