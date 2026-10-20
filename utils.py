"""Plotting, greedy evaluation, and GIF export utilities."""

from pathlib import Path
import gymnasium as gym
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")  # PNG files work even without a desktop display.
import matplotlib.pyplot as plt
import numpy as np

from agent import QLearningAgent


def moving_average(values, window: int = 100) -> np.ndarray:
    """Return a valid moving average (or [] for empty input)."""
    data = np.asarray(values, dtype=float)
    if window < 1:
        raise ValueError("window must be at least 1")
    if data.size == 0:
        return np.array([], dtype=float)
    window = min(window, data.size)
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_training_curves(
    rewards, steps, epsilons, output_dir: str | Path, window: int = 100,
) -> Path:
    """Save reward, episode length, and epsilon learning curves in one PNG."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = [np.asarray(x, dtype=float) for x in (rewards, steps, epsilons)]
    if not sequences[0].size or len(set(map(len, sequences))) != 1:
        raise ValueError("The three nonempty history arrays must have equal lengths.")

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    names = ["Episode reward", "Steps per episode", "Epsilon"]
    episodes = np.arange(1, len(sequences[0]) + 1)
    for ax, series, name in zip(axes, sequences, names):
        ax.plot(episodes, series, alpha=0.18, linewidth=0.7, label="Raw")
        smooth = moving_average(series, window)
        x_smooth = episodes[len(episodes) - len(smooth):]
        ax.plot(x_smooth, smooth, linewidth=1.8, label="Moving average")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Training episode")
    fig.suptitle("Taxi-v3: Q-learning training progress")
    fig.tight_layout()
    destination = output_dir / "training_curves.png"
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def evaluate_policy(
    agent: QLearningAgent, episodes: int = 100, seed: int = 10_000,
) -> dict[str, float]:
    """Evaluate a GREEDY policy; return mean reward, success fraction, steps.

    Each test episode gets a fixed, distinct seed for reproducibility.
    A successful Taxi-v3 episode ends with a +20 delivery reward.
    """
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = gym.make("Taxi-v3")
    total_rewards, total_steps, successes = [], [], 0
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed + episode)
            done, reward_sum, step_count = False, 0.0, 0
            while not done:
                action = agent.select_action(int(state), greedy=True)
                state, reward, terminated, truncated, _ = env.step(action)
                reward_sum += float(reward)
                step_count += 1
                successes += int(bool(terminated) and reward == 20)
                done = terminated or truncated
            total_rewards.append(reward_sum)
            total_steps.append(step_count)
    finally:
        env.close()
    return {
        "mean_reward": float(np.mean(total_rewards)),
        "success_rate": successes / episodes,  # 0 to 1
        "average_steps": float(np.mean(total_steps)),
    }


def save_agent_gif(
    agent: QLearningAgent, path: str | Path,
    seed: int = 20_000, max_attempts: int = 100, fps: int = 4,
) -> Path:
    """Save a successful greedy Taxi rollout; error if none is found.

    Never labels an unsuccessful rollout as a successful demonstration.
    """
    if max_attempts < 1 or fps < 1:
        raise ValueError("max_attempts and fps must be positive")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = gym.make("Taxi-v3", render_mode="rgb_array")
    try:
        for attempt in range(max_attempts):
            state, _ = env.reset(seed=seed + attempt)
            frames = [np.asarray(env.render()).copy()]
            while True:
                action = agent.select_action(int(state), greedy=True)
                state, reward, terminated, truncated, _ = env.step(action)
                frames.append(np.asarray(env.render()).copy())
                if terminated or truncated:
                    if terminated and reward == 20:
                        imageio.mimsave(path, frames, duration=1 / fps, loop=0)
                        return path
                    break
    finally:
        env.close()
    raise RuntimeError(
        "No successful greedy episode found for the GIF. "
        "Try more training episodes or slower epsilon decay."
    )
