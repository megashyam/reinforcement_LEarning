# 🐍 Classic Snake with classic Q-Learning

A Reinforcement Learning-based Snake Game where a Q-Learning agent learns to maximize its score by navigating the grid, avoiding collisions, and eating food. Includes real-time OpenCV rendering, visual themes, animation effects, and performance plots.

![Snake Gameplay](Viz/snek.gif) <!-- Optional: Add a demo gif -->


### Q-Learning Overview

Q-Learning is a model-free Reinforcement Learning algorithm. The agent (Snake) learns a Q-table that maps **states** to **actions** with estimated rewards:

## 🧠 What is Q-Learning?

**Q-Learning** is a value-based reinforcement learning algorithm. It learns a function:

![Q-Learning Formula](https://github.com/user-attachments/assets/3183fe12-e971-4508-885c-173eab222b12)

- `s`: current state  
- `a`: action taken  
- `r`: reward received  
- `s'`: new state after action  
- `α`: learning rate  
- `γ`: discount factor

The agent uses **epsilon-greedy exploration** to balance trying new actions (exploration) vs. using learned knowledge (exploitation).

---


## State Representation


(dx, dy, danger_left, danger_right, danger_up, danger_down)

- dx, dy: Direction to food (clipped to -1, 0, 1)
- danger_left/right/up/down: Whether a move in that direction would cause death
