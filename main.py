"""Train and evaluate a custom Taxi-v3 tabular Q-learning agent."""

import argparse
import json
from pathlib import Path
import random
import gymnasium as gym
import numpy as np

from agent import QLearningAgent
from utils import evaluate_policy, plot_training_curves, save_agent_gif


def train_agent(
    episodes: int = 15_000, alpha: float = 0.1, gamma: float = 0.95,
    epsilon_start: float = 1.0, epsilon_min: float = 0.01,
    epsilon_decay: float = 0.9995, seed: int = 42,
) -> tuple[QLearningAgent, dict[str, list[float]]]:
    """Train with the Gymnasium reset/step API and return episode histories."""
    if episodes < 1:
        raise ValueError("episodes must be positive")
    random.seed(seed)
    np.random.seed(seed)
    env = gym.make("Taxi-v3")
    env.action_space.seed(seed)
    agent = QLearningAgent(
        n_states=env.observation_space.n,
        n_actions=env.action_space.n,
        alpha=alpha, gamma=gamma,
        epsilon_start=epsilon_start, epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay, seed=seed,
    )
    history = {"rewards": [], "steps": [], "epsilons": []}
    try:
        for episode in range(episodes):
            # Seed only the first reset: subsequent states use the env's RNG.
            state, _ = env.reset(seed=seed) if episode == 0 else env.reset()
            reward_sum, steps = 0.0, 0
            while True:
                action = agent.select_action(int(state))
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.update(int(state), action, float(reward), int(next_state), terminated)
                reward_sum += float(reward)
                steps += 1
                state = next_state
                if terminated or truncated:
                    break
            history["rewards"].append(reward_sum)
            history["steps"].append(steps)
            history["epsilons"].append(agent.epsilon)  # epsilon used this episode
            agent.decay_epsilon()
    finally:
        env.close()
    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(description="From-scratch Taxi-v3 Q-learning")
    parser.add_argument("--episodes", type=int, default=15_000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gif-attempts", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    agent, history = train_agent(
        episodes=args.episodes, alpha=args.alpha, gamma=args.gamma,
        epsilon_start=args.epsilon_start, epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay, seed=args.seed,
    )
    agent.save(args.output_dir / "q_table.npy")
    np.savez(args.output_dir / "training_history.npz", **history)
    plot_training_curves(**history, output_dir=args.output_dir)

    result = evaluate_policy(agent, episodes=args.eval_episodes, seed=args.seed + 10_000)
    with open(args.output_dir / "evaluation.json", "w", encoding="utf-8") as file:
        json.dump({"configuration": vars(args) | {"output_dir": str(args.output_dir)},
                   "evaluation": result}, file, indent=2)
    print(f"Mean reward:   {result['mean_reward']:.2f}")
    print(f"Success rate:  {result['success_rate']:.1%}")
    print(f"Average steps: {result['average_steps']:.2f}")

    gif_path = save_agent_gif(
        agent, args.output_dir / "taxi_agent.gif",
        seed=args.seed + 20_000, max_attempts=args.gif_attempts,
    )
    print(f"Saved learned policy and visualizations to: {args.output_dir.resolve()}")
    print(f"Successful agent GIF: {gif_path.resolve()}")


if __name__ == "__main__":
    main()
