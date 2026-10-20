"""One-factor-at-a-time sensitivity analysis with three or more seeds."""

import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from main import train_agent
from utils import evaluate_policy, moving_average

# Proposed baseline; the best setting must be chosen from measured results.
BASE = {"alpha": 0.1, "gamma": 0.95, "epsilon_decay": 0.9995}
SWEEPS = {
    "alpha": [0.05, 0.1, 0.5],
    "gamma": [0.90, 0.95, 0.99],
    "epsilon_decay": [0.999, 0.9995, 0.9998],
}


def run_experiments(
    episodes: int = 8_000, seeds=(0, 1, 2), eval_episodes: int = 100,
    output_dir: str | Path = "outputs/experiments", window: int = 100,
) -> Path:
    """Run each sweep, save std-shaded curves and per-seed/summary CSVs."""
    if episodes < 1 or eval_episodes < 1 or window < 1 or len(set(seeds)) < 3:
        raise ValueError("Need positive episodes/window and at least 3 distinct seeds.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, summary_rows = [], []

    for parameter, values in SWEEPS.items():
        fig, ax = plt.subplots(figsize=(10, 5))
        for value in values:
            config = dict(BASE)
            config[parameter] = value  # change exactly one parameter
            curves, metrics = [], []
            for seed in seeds:
                print(f"Training {parameter}={value}, seed={seed} ...", flush=True)
                agent, history = train_agent(
                    episodes=episodes, epsilon_start=1.0,
                    epsilon_min=0.01, seed=int(seed), **config,
                )
                result = evaluate_policy(
                    agent, episodes=eval_episodes, seed=10_000 + int(seed) * 1000,
                )
                curves.append(moving_average(history["rewards"], window))
                metrics.append(result)
                raw_rows.append({
                    "parameter": parameter, "value": value, "seed": seed,
                    "alpha": config["alpha"], "gamma": config["gamma"],
                    "epsilon_decay": config["epsilon_decay"], **result,
                })

            curves = np.stack(curves)
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0, ddof=1)
            xs = np.arange(episodes - len(mean_curve) + 1, episodes + 1)
            ax.plot(xs, mean_curve, label=f"{parameter}={value:g}")
            ax.fill_between(xs, mean_curve - std_curve,
                            mean_curve + std_curve, alpha=0.15)
            summary_rows.append({
                "parameter": parameter, "value": value,
                "seeds": len(seeds), "train_episodes": episodes,
                "mean_eval_reward": float(np.mean([m["mean_reward"] for m in metrics])),
                "std_eval_reward": float(np.std([m["mean_reward"] for m in metrics], ddof=1)),
                "mean_success_rate": float(np.mean([m["success_rate"] for m in metrics])),
                "std_success_rate": float(np.std([m["success_rate"] for m in metrics], ddof=1)),
                "mean_average_steps": float(np.mean([m["average_steps"] for m in metrics])),
                "std_average_steps": float(np.std([m["average_steps"] for m in metrics], ddof=1)),
            })
        ax.set(title=f"Sensitivity to {parameter} (mean ± 1 SD across seeds)",
               xlabel="Training episode", ylabel=f"Reward ({window}-episode moving average)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{parameter}_comparison.png", dpi=160)
        plt.close(fig)

    for filename, rows in [
        ("experiments_results.csv", raw_rows),
        ("experiments_summary.csv", summary_rows),
    ]:
        with (output_dir / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Experiment results saved under {output_dir.resolve()}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Taxi-v3 hyperparameter sweeps")
    parser.add_argument("--episodes", type=int, default=8_000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()
    run_experiments(
        episodes=args.episodes, seeds=args.seeds,
        eval_episodes=args.eval_episodes, window=args.window,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
