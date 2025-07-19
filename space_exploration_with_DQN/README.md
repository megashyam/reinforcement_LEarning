# 🚀 LunarLander DQN Agent

A **PyTorch-based Deep Q-Learning agent** for solving the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment **Double DQN**, **Dueling DQN**, **Experience Replay**, and **Target Networks**. Supports full training and evaluation with live epsilon decay tracking and model saving.

### Concept Overview

The LunarLander environment challenges an agent to land a spacecraft safely between two flags on the moon’s surface. Rewards are based on:

- Successful soft landing (+100 to +140)
- Crashing (-100)
- Firing engines (-0.3 per frame)
- Moving closer to the landing pad

![LunarLander Demo](https://user-images.githubusercontent.com/63813881/230761198-1b263b57-1a5e-45d3-983f-9b08f46e28a1.gif)

## Features

- **Dueling Network Architecture** - Separates value and advantage streams
- **Double DQN** - Reduces Q-value overestimation
- **Prioritized Experience Replay** - Efficient sampling of experiences
- **ε-Greedy Exploration** - With annealing schedule

## What is Deep Q-Network (DQN)?

DQN is a reinforcement learning algorithm that uses a deep neural network to approximate the Q-function \(Q(s, a)\), which predicts the expected reward of taking action \(a\) in state \(s\). It enables learning in environments with large or continuous state spaces where traditional Q-tables are infeasible.

### How It Works

- The network takes the current state as input and outputs Q-values for all possible actions.
- Actions are selected using an **epsilon-greedy** policy balancing exploration and exploitation.
- Experiences \((s, a, r, s')\) are stored in a replay buffer and sampled randomly to stabilize training.
- A separate **target network** is updated periodically to provide stable Q-value targets.
- The network is trained to minimize the difference between predicted Q-values and target Q-values computed via the Bellman equation

### Benefits

- Handles complex, high-dimensional environments.
- Provides stable and efficient learning compared to classical Q-learning.

## What is Double DQN?

Double DQN improves upon regular DQN by addressing the problem of **overestimation bias** in Q-value estimates.

### Overestimation Bias in Regular DQN

Regular DQN uses the same network both to select and evaluate the best next action:

![Double DQN](https://github.com/megashyam/reinforcement_LEarning/blob/main/space_exploration_with_DQN/Viz/1_nm0lt3oobxdBHTMACUZ-cg.png)

This can lead to overoptimistic value estimates, harming learning stability.

### How Double DQN Fixes This

Double DQN decouples action selection and evaluation by using two networks:

- **Policy network** selects the best next action:

- **Target network** evaluates that action:

### Benefits

- Reduces overestimation bias
- Improves learning stability
- Leads to better policy performance

## Architecture

### Double DQN
```python
# Action selection
max_actions = policy_net(next_states).argmax(1)
# Q-value evaluation
target_q = target_net(next_states).gather(1, max_actions.unsqueeze(1)).squeeze()
```

## What is Dueling DQN?

Dueling DQN improves on regular DQN by separately estimating:

![Dueling DQN](https://github.com/megashyam/reinforcement_LEarning/blob/main/space_exploration_with_DQN/Viz/dueling-dqn-framework.png)

- The **state-value function** : how good it is to be in a given state regardless of action.
- The **advantage function** : how much better taking a specific action is compared to the average action at that state.

These are combined to get the Q-value

### Advantages of Dueling DQN

- Learns state values more efficiently, especially when actions have similar outcomes.
- Helps the agent focus on states that matter, improving policy stability.
- Produces more robust Q-value estimates, reducing variance in training.


## Architecture

### Dueling DQN

```python
class DuelingDQN(nn.Module):
    def __init__(self, n_states, n_actions):
        super().__init__()
        # Shared layers
        self.fc1 = nn.Linear(n_states, 256)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

```

## Project Structure

```
LunarLander-DQN/
├── lunar.py                # Main script (training/evaluation logic)
├── dqn.py                  # DQN architecture (supports dueling and dual DQN)
├── buffer.py               # Experience replay buffer
├── hyperparameters.yaml    # YAML file with all training hyperparameters
├── runs/                   # Output directory (auto-generated)
│   ├── logs.log            # Training logs (best reward updates)
│   ├── *.pt                # Saved PyTorch models (best performing)
│   └── LunarLander_graph.png  # Training reward and epsilon decay graph
```

## Usage

### Training
```bash
python lunar.py --train

```

### Evaluation
```
python lunar.py --model-file filename.pt
```


