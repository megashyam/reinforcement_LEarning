# py=3.11
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym

import yaml
import torch
import random
from torch import nn

from dqn import DQN
from buffer import BufferMemory

from datetime import datetime, timedelta
import argparse
import os
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

DATE_FORMAT = "%m-%d %H:%M:%S"

RUN_DIR = "runs"

os.makedirs(RUN_DIR, exist_ok=True)

matplotlib.use("Agg")


class Falcon:
    def __init__(self, args):

        with open("hyperparameters.yaml", "rb") as f:
            hpr = yaml.safe_load(f)

        self.epsilon_init = hpr["epsilon_init"]
        self.epsilon_decay = hpr["epsilon_decay"]
        self.learning_rate = hpr["learning_rate"]
        self.discount = hpr["discount_factor"]
        self.total_episodes = hpr["total_episodes"]
        self.total_steps_per_episode = hpr["total_steps_per_episode"]
        self.fc1_nodes = hpr["fc1_nodes"]
        self.fc_dueling = hpr["fc_dueling"]
        self.render = hpr["render"]

        self.target_sync_every = hpr["network_sync_rate"]
        self.mini_batch_size = hpr["mini_batch_size"]
        self.min_epsilon = hpr["min_epsilon"]
        self.enable_double_dqn = hpr["enable_double_dqn"]
        self.enable_dueling_dqn = hpr["enable_dueling_dqn"]
        self.show_every = hpr["show_every"]

        self.log_file = os.path.join(RUN_DIR, "logs.log")
        self.model_file = os.path.join(
            RUN_DIR, f'LunarLander_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pt'
        )
        self.graph_file = os.path.join(
            RUN_DIR,
            f"LunarLander_graph.png"
        )
        if args.model_file:
            self.model_path = os.path.join(RUN_DIR, args.model_file)
        else:
            self.model_path = None

        self.loss_fn = nn.SmoothL1Loss()

        #self.best_loss = 
        self.n_states = None
        self.n_actions = None

    def run(self, is_training=False, render=False, model_path=None):

        if is_training:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_msg = f"{start_time.strftime(DATE_FORMAT)}: Training..."
            print(log_msg)

            with open(self.log_file, "w") as f:
                f.write(log_msg + "\n")

        env_name = "LunarLander-v3"

        if is_training:
            env_no_render = gym.make(env_name, render_mode=None)
            env_render = gym.make(env_name, render_mode="human")
        else:
            env_eval = gym.make(env_name, render_mode="human")

        sample_env = env_render if (is_training and self.render) else env_eval if not is_training else env_no_render
        self.n_states = sample_env.observation_space.shape[0]
        self.n_actions = sample_env.action_space.n

        policy_net = DQN(
            self.n_states,
            self.n_actions,
            self.fc1_nodes,
            self.enable_dueling_dqn,
            self.fc_dueling if self.enable_dueling_dqn else None,
        ).to(device)

        epsilon = self.epsilon_init

        if is_training:

            buff_mem = BufferMemory()
            self.optimizer = torch.optim.Adam(
                policy_net.parameters(), lr=self.learning_rate
            )

            policy_net.train()

            target_net = DQN(
                self.n_states,
                self.n_actions,
                self.fc1_nodes,
                self.enable_dueling_dqn,
                self.fc_dueling if self.enable_dueling_dqn else None,
            ).to(device)

            target_net.load_state_dict(policy_net.state_dict())

            best_reward = -9999
            reward_per_episode = []
            epsilon_hist = []
            loss_hist = []
            target_sync = 0
        else:
            if self.model_path is None:
                raise ValueError("Model path must be specified for evaluation.")
            policy_net.load_state_dict(torch.load(self.model_path))
            policy_net.eval()

        for episode in range(self.total_episodes):

            if is_training:
                do_render = (episode % self.show_every == 0)
                env = env_render if do_render else env_no_render
                if do_render:
                    print(f"Rendering on Episode:{episode}")
            else:
                env = env_eval

            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            done = False

            step = 0
            episode_reward = 0.0

            while step < self.total_steps_per_episode and not done:

                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, device=device)
                else:
                    with torch.no_grad():
                        action = policy_net(state.unsqueeze(0)).squeeze().argmax()

                next_state, reward, terminated, truncated, _ = env.step(action.item())

                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                episode_reward += reward.item()

                done = terminated or truncated

                if is_training:
                    buff_mem.append((state, action, reward, next_state, terminated))
                    
                    target_sync += 1

                state = next_state

                if is_training and len(buff_mem) > self.mini_batch_size:

                    mini_batch = buff_mem.sample(self.mini_batch_size)

                    self.train_policy_net(mini_batch, policy_net, target_net)
                    epsilon = max(epsilon * self.epsilon_decay, self.min_epsilon)
                    epsilon_hist.append(epsilon)

                    if target_sync % self.target_sync_every == 0:
                        target_net.load_state_dict(policy_net.state_dict())
                        target_sync = 0

                if is_training and episode_reward > best_reward:
                    log_msg = (
                        f"{datetime.now().strftime(DATE_FORMAT)}: New best reward: "
                        f"{episode_reward:0.1f} "
                        f"({(episode_reward - best_reward) / episode_reward * 100:+.1f}%) "
                        f"at episode {episode}"
                    )
                    print(log_msg)

                    with open(self.log_file, "w") as f:
                        f.write(log_msg + "\n")

                    torch.save(policy_net.state_dict(), self.model_file)
                    best_reward = episode_reward

                current_time = datetime.now()
                if is_training and (current_time - last_graph_update_time > timedelta(seconds=2)):
                    self.save_graph(reward_per_episode, epsilon_hist)
                    last_graph_update_time = current_time

                step += 1

            # if is_training:
        	   #  if do_render:
        		  #   env.close()
            if is_training:
                reward_per_episode.append(episode_reward)
            #env.close()
            

    def save_graph(self, reward_per_episode, epsilon_hist):

        fig = plt.figure()
        mean_rewards = np.zeros(len(reward_per_episode))

        for i in range(len(mean_rewards)):
            mean_rewards[i] = np.mean(reward_per_episode[max(0, i - 99) : (i + 1)])

        plt.subplot(121)
        plt.ylabel("Mean Rewards (100-Episode avg)")
        plt.xlabel("Episodes")
        plt.plot(mean_rewards)

        plt.subplot(122)
        plt.ylabel("Epsilon Decay (100-Episode avg)")
        plt.xlabel("Episodes")
        plt.plot(epsilon_hist)

        plt.subplots_adjust(wspace=1, hspace=1)

        fig.savefig(self.graph_file)
        plt.close(fig)

    def train_policy_net(self, mini_batch, policy_net, target_net):

        states, actions, rewards, next_states, terminated = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_state = torch.stack(next_states)
        reward = torch.stack(rewards)
        terminated = torch.tensor(terminated, dtype=torch.float, device=device)

        current_q = (
            policy_net(states).gather(dim=1, index=actions.unsqueeze(1)).squeeze()
        )

        with torch.no_grad():
            if self.enable_double_dqn:
                max_future_q = policy_net(new_state).argmax(dim=1)
                target_q = (reward+ (1 - terminated)* self.discount* target_net(new_state).gather(dim=1, index=max_future_q.unsqueeze(1)).squeeze())

            else:
                target_q = (reward + (1 - terminated) * self.discount * target_net(new_state).max(dim=1)[0])

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        self.optimizer.step()

        # if self.best_loss > loss:
        #     self.best_loss = loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or Test Model")
    parser.add_argument("--train", help="Training Mode", action="store_true")
    parser.add_argument("--model-file", type=str, help="Path to pretrained model to run",
    )
    args = parser.parse_args()

    falcon = Falcon(args)

    if args.train:
        falcon.run(is_training=True)
    else:
        falcon.run(is_training=False, render=True)
