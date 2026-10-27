
class GridWorld:
    """4x4 GridWorld with three terminal holes.

    Actions: 0=up, 1=right, 2=down, 3=left.
    """

    ACTION_NAMES = ("U", "R", "D", "L")

    def __init__(
        self,
        size=4,
        step_reward=-1.0,
        goal_reward=10.0,
        hole_reward=-10.0,
        holes=None,
    ):
        if size < 2:
            raise ValueError("Grid size must be at least 2.")

        self.size = size
        self.start = 0
        self.goal = size * size - 1

        # Three holes for the default 4x4 grid
        default_holes = {5, 6, 9} if size == 4 else set()
        self.holes = set(default_holes if holes is None else holes)

        if self.start in self.holes or self.goal in self.holes:
            raise ValueError("Start and goal cannot be holes.")

        if any(h < 0 or h >= size * size for h in self.holes):
            raise ValueError("Hole state is outside the grid.")

        self.step_reward = float(step_reward)
        self.goal_reward = float(goal_reward)
        self.hole_reward = float(hole_reward)

        self.actions = (0, 1, 2, 3)
        self.moves = (
            (-1, 0),  # Up
            (0, 1),   # Right
            (1, 0),   # Down
            (0, -1),  # Left
        )

        self.state = self.start
        self.done = False

    @property
    def n_states(self):
        return self.size * self.size

    @property
    def n_actions(self):
        return len(self.actions)

    @property
    def terminal_states(self):
        return self.holes | {self.goal}

    def reset(self):
        self.state = self.start
        self.done = False
        return self.state

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Episode is over. Call reset() before stepping again."
            )

        if action not in self.actions:
            raise ValueError("Invalid action.")

        row, col = divmod(self.state, self.size)
        dr, dc = self.moves[action]

        new_row = max(0, min(self.size - 1, row + dr))
        new_col = max(0, min(self.size - 1, col + dc))

        next_state = new_row * self.size + new_col

        if next_state == self.goal:
            reward = self.goal_reward
            done = True
        elif next_state in self.holes:
            reward = self.hole_reward
            done = True
        else:
            reward = self.step_reward
            done = False

        self.state = next_state
        self.done = done

        return next_state, reward, done
