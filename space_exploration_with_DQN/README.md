# 🚀 LunarLander DQN Agent

A **PyTorch-based Deep Q-Learning agent** for solving the [LunarLander-v3](https://gymnasium.farama.org/environments/box2d/lunar_lander/) environment using advanced RL techniques like **Double DQN**, **Dueling DQN**, **Experience Replay**, and **Target Networks**. Supports full training and evaluation with live epsilon decay tracking and model saving.

![LunarLander Demo](https://user-images.githubusercontent.com/63813881/230761198-1b263b57-1a5e-45d3-983f-9b08f46e28a1.gif)

## Features

- **Dueling Network Architecture** - Separates value and advantage streams
- **Double DQN** - Reduces Q-value overestimation
- **Prioritized Experience Replay** - Efficient sampling of experiences
- **ε-Greedy Exploration** - With annealing schedule
- **Modular Design** - Easy to extend and modify



## Concept Overview

The LunarLander environment challenges an agent to land a spacecraft safely between two flags on the moon’s surface. Rewards are based on:

- Successful soft landing (+100 to +140)
- Crashing (-100)
- Firing engines (-0.3 per frame)
- Moving closer to the landing pad




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
![Double DQN](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2F%40qempsil0914%2Fdeep-q-learning-part2-double-deep-q-network-double-dqn-b8fc9212bbb2&psig=AOvVaw3AmCGLPuTddSYs-Vy1nyV1&ust=1753054480945000&source=images&cd=vfe&opi=89978449&ved=0CBYQjRxqFwoTCOjhioCLyo4DFQAAAAAdAAAAABAE)
\[
\text{target} = r + \gamma \max_{a'} Q(s', a'; \theta)
\]

This can lead to overoptimistic value estimates, harming learning stability.

### How Double DQN Fixes This

Double DQN decouples action selection and evaluation by using two networks:

- **Policy network** selects the best next action:

\[
a^* = \arg\max_{a'} Q(s', a'; \theta)
\]

- **Target network** evaluates that action:

target = r + γ * max_a' Q(s', a'; θ)


where \(\theta^-\) are the target network parameters.

### Benefits

- Reduces overestimation bias
- Improves learning stability
- Leads to better policy performance


## What is Deep Q-Network (DQN)?

Deep Q-Network (DQN) is a reinforcement learning algorithm that combines **Q-learning** with **deep neural networks** to solve problems with high-dimensional state spaces, such as video games or robotics tasks.

### Key Ideas

- **Q-Learning**: A value-based RL method where the agent learns a function \( Q(s, a) \) that estimates the expected cumulative reward (return) when taking action \( a \) in state \( s \), and then following the best policy thereafter.

- **Deep Neural Network**: Instead of maintaining a Q-table (which is infeasible for large or continuous state spaces), DQN uses a neural network to approximate the Q-function \( Q(s, a; \theta) \), where \( \theta \) are the network weights.

---



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

### Double DQN
```python
# Action selection
max_actions = policy_net(next_states).argmax(1)
# Q-value evaluation
target_q = target_net(next_states).gather(1, max_actions.unsqueeze(1)).squeeze()
```



### How DQN Works

- **State:** 8-dimensional vector representing position, velocity, angle, etc.
- **Actions:** 4 discrete actions:
  - 0: Do nothing
  - 1: Fire left engine
  - 2: Fire main engine
  - 3: Fire right engine
- **Neural Network:** Approximates Q-values for each action.
- **Experience Replay:** Stores past experiences for stable training.
- **Target Network:** A periodically updated network for stable Q-value targets.
- **Double DQN (optional):** Reduces overestimation by using the policy network for action selection and the target network for evaluation.
- **Dueling DQN (optional):** Separates state-value and advantage estimations for better learning.

---

## 🗂️ Project Structure


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


