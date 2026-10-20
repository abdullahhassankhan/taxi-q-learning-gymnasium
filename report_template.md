# Custom Reinforcement Learning Agent Implementation Using Farama Gymnasium

**Environment:** Taxi-v3  
**Algorithm:** Tabular Q-learning with epsilon-greedy exploration  
**Student:** [INSERT NAME AND REGISTRATION NUMBER]  
**Course:** Topics in AI — Assignment I  

## 1. Problem Description

The task is to train an autonomous taxi agent to collect a passenger and deliver the passenger to a specified destination on a 5 × 5 grid. The Taxi-v3 environment in Farama Gymnasium has **500 discrete encoded states**, representing the taxi's row (5), column (5), passenger location/status (5), and destination (4): 5 × 5 × 5 × 4 = 500. The action space has **six actions**: south (0), north (1), east (2), west (3), pickup (4), and drop-off (5). Standard rewards are **−1** per ordinary step, **−10** for an illegal pickup/drop-off, and **+20** for successful delivery. Episodes terminate after successful delivery and are truncated at the environment's 200-step time limit. The objective is to maximize cumulative reward while learning efficient, legal routes.

## 2. Methodology

A tabular action-value function, Q(s,a), was initialized to zero for all 500 × 6 state-action pairs. After each interaction, the agent updates one entry using the off-policy Q-learning rule:

**Q(s,a) ← Q(s,a) + α [r + γ maxₐ′ Q(s′,a′) − Q(s,a)].**

Here, **s** is the current state; **a** is the chosen action; **r** is the immediate reward; **s′** is the next state; **a′** denotes a possible next action; **Q(s,a)** estimates the expected discounted return; **α** is the learning rate; and **γ** is the discount factor. For a terminated transition, the future-value term is set to zero. Time-limit truncation stops the episode but is not considered a true terminal state in the learning target.

To balance exploration and exploitation, an epsilon-greedy policy chooses a random action with probability **ε** and a highest-Q action with probability **1 − ε**. Epsilon is decayed once after every episode according to **εₖ₊₁ = max(ε_min, εₖ × d)**, where **d** is the multiplicative decay factor. Equal-valued actions are randomized during training, while greedy evaluation uses a deterministic argmax.

The implementation separates concerns across `agent.py` (Q-table and update), `utils.py` (evaluation, figures and GIF), `main.py` (training, seeds and saved files), and `experiments.py` (one-factor-at-a-time sensitivity analysis). The notebook reproduces inspection, training, evaluation and comparisons.

## 3. Experimental Setup and Hyperparameters

The baseline configuration below is a **proposed starting point, not an empirically tuned optimum**. Replace final settings after reviewing experiment outputs.

| Parameter | Baseline configuration | Selected final value |
|:--|--:|:--|
| Environment | Taxi-v3 | Taxi-v3 |
| Training episodes | 15,000 | [INSERT FINAL EPISODE COUNT] |
| Learning rate (α) | 0.10 | [INSERT SELECTED α] |
| Discount factor (γ) | 0.95 | [INSERT SELECTED γ] |
| Initial epsilon | 1.00 | [INSERT FINAL START EPSILON] |
| Minimum epsilon | 0.01 | [INSERT FINAL MIN EPSILON] |
| Epsilon decay (d) | 0.9995 | [INSERT SELECTED DECAY] |
| Main training seed | 42 | [INSERT FINAL SEED] |
| Evaluation episodes | 100 | [INSERT EVALUATION COUNT] |
| Sensitivity seeds | 0, 1, 2 | [INSERT SEEDS USED] |

The one-factor-at-a-time sensitivity study compares **α = {0.05, 0.10, 0.50}**, **γ = {0.90, 0.95, 0.99}**, and **d = {0.999, 0.9995, 0.9998}**. All other settings are fixed within each sweep, with three independent training seeds for each setting. The study reports seed-averaged greedy evaluation scores, success fractions, episode lengths, and mean reward curves with ±1 standard deviation. Training and evaluation are separated; evaluation uses ε = 0 behavior without Q updates.

## 4. Results and Discussion

**Figure 1 — Training dynamics:** [INSERT training_curves.png]. Describe whether episode reward improves, the typical steps decrease, and epsilon follows the intended schedule. Define convergence operationally (e.g., a stable moving-average reward over a specified trailing window); report the first sustained convergence episode as **[INSERT CONVERGENCE EPISODE OR STATE NOT CLEARLY OBSERVED]**. Avoid claiming convergence without evidence.

**Figure 2 — Sensitivity study:** [INSERT alpha_comparison.png, gamma_comparison.png, epsilon_decay_comparison.png AS LEGIBLE PANEL(S)]. Compare learning speed, final evaluation, and between-seed variability at the same training budget. Excessively large α may generate volatile updates; a smaller γ may undervalue long routes, while an excessive γ may emphasize distant reward; rapid epsilon decay can cause under-exploration. These are hypotheses to check against the plots, not observed outcomes until verified.

| Final greedy-policy metric | Observed output |
|:--|:--|
| Mean total reward | [INSERT FINAL MEAN REWARD FROM MY RUN] |
| Successful deliveries | [INSERT SUCCESS RATE % FROM MY RUN] |
| Average steps per episode | [INSERT AVERAGE STEPS FROM MY RUN] |
| Convergence episode (if identifiable) | [INSERT EPISODE OR NOT CLEARLY IDENTIFIABLE] |
| Best α / γ / decay combination | [INSERT SELECTED VALUES AND SELECTION BASIS] |

An agent can fail through illegal pickup/drop-off actions, cycling between locations, low state coverage, or time-limit truncation. If performance is inconsistent across training seeds, report that variability instead of relying on a single favorable rollout. The saved GIF depicts one successful greedy episode and must not be interpreted as a replacement for aggregate evaluation.

## 5. Conclusion

A custom Q-learning implementation was developed to learn an action policy for the finite Taxi-v3 decision process without dedicated RL algorithm libraries. The learning curves, independent greedy-policy evaluation, and controlled hyperparameter sweeps provide evidence for evaluating learning progress and sensitivity. The measured policy performance was **[INSERT CONCISE EVIDENCE-BASED FINDING]**. Future work could investigate different exploration schedules, action masking, and more independent seeds.

**References:** Farama Foundation, *Gymnasium Taxi environment documentation*; Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Add citations/URLs according to the course format. 
