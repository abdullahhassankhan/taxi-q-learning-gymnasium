"""From-scratch tabular Q-learning agent for discrete environments."""

from pathlib import Path
import numpy as np


class QLearningAgent:
    """Store and learn state-action values using an epsilon-greedy policy."""

    def __init__(
        self, n_states: int, n_actions: int, alpha: float = 0.1,
        gamma: float = 0.95, epsilon_start: float = 1.0,
        epsilon_min: float = 0.01, epsilon_decay: float = 0.9995,
        seed: int = 42,
    ) -> None:
        if n_states < 1 or n_actions < 1:
            raise ValueError("State/action counts must be positive.")
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma <= 1.")
        if not 0 <= epsilon_min <= epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_min <= epsilon_start <= 1.")
        if not 0 < epsilon_decay <= 1:
            raise ValueError("Require 0 < epsilon_decay <= 1.")

        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def select_action(self, state: int, greedy: bool = False) -> int:
        """Choose randomly with probability epsilon; otherwise maximize Q.

        During training, ties between best actions are broken randomly.
        During evaluation, np.argmax breaks ties deterministically.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q_table.shape[1]))
        values = self.q_table[state]
        if greedy:
            return int(np.argmax(values))
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def update(
        self, state: int, action: int, reward: float,
        next_state: int, terminated: bool,
    ) -> None:
        """Apply one Q-learning update; never bootstrap from a terminal state.

        Time-limit truncation is intentionally NOT treated as termination.
        """
        future_value = 0.0 if terminated else float(self.q_table[next_state].max())
        td_target = reward + self.gamma * future_value
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Apply multiplicative exploration decay once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        """Save learned Q-values in NumPy .npy format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: str | Path) -> None:
        """Load Q-values and check dimensions, without allowing pickle."""
        loaded = np.load(path, allow_pickle=False)
        if loaded.shape != self.q_table.shape:
            raise ValueError(f"Expected {self.q_table.shape}; got {loaded.shape}.")
        if not np.issubdtype(loaded.dtype, np.number):
            raise ValueError("Q-table must contain numeric values.")
        self.q_table = np.asarray(loaded, dtype=np.float64)
