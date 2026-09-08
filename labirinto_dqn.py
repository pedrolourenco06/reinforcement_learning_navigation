import os
import random
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import maze_information_map.ackermann_env as cm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        obs_dim,
        num_actions,
        gamma=0.99,
        lr=1e-3,
        buffer_size=50000,
        device=None
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma

        if device is None:
            try:
                import torch_directml
                self.device = torch_directml.device()
                print("Usando GPU AMD via DirectML")
            except ImportError:
                self.device = torch.device("cpu")
                print("torch-directml não encontrado. Usando CPU.")
        else:
            self.device = device

        self.q_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions).to(self.device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, eps):
        # exploracao epsilon-greedy
        if random.random() < eps:
            return random.randrange(self.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_net(state_t)

        return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a) predito pela rede principal
        q_values = self.q_net(states_t).gather(1, actions_t)

        # alvo DQN: r + gamma * max_a' Q_target(s',a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            targets = rewards_t + self.gamma * next_q_values * (1.0 - dones_t)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # ajuda a evitar explosao de gradiente
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def save(self, filename):
        torch.save({
            "q_net_state_dict": self.q_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "obs_dim": self.obs_dim,
            "num_actions": self.num_actions,
            "gamma": self.gamma,
        }, filename)


def moving_average(values, window=50):
    if len(values) == 0:
        return []

    avg = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        avg.append(np.mean(values[start:i + 1]))

    return avg


def save_training_plots(rewards, success_rate, losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    avg_rewards = moving_average(rewards, window=50)

    plt.figure(1)
    plt.clf()
    plt.plot(rewards, "r", alpha=0.25, label="Recompensa por episodio")
    plt.plot(avg_rewards, "b", linewidth=2, label="Media movel 50 episodios")
    plt.title("DQN - Recompensa por episodio")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dqn_reward.png"), dpi=300, bbox_inches="tight")

    plt.figure(2)
    plt.clf()
    plt.plot(success_rate, "g", linewidth=2)
    plt.title("DQN - Taxa de sucesso")
    plt.xlabel("Episodios")
    plt.ylabel("Taxa de sucesso (media movel 50 episodios)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "dqn_success_rate.png"), dpi=300, bbox_inches="tight")

    if len(losses) > 0:
        avg_losses = moving_average(losses, window=200)

        plt.figure(3)
        plt.clf()
        plt.plot(losses, alpha=0.25, label="Loss")
        plt.plot(avg_losses, linewidth=2, label="Media movel 200")
        plt.title("DQN - Loss")
        plt.xlabel("Atualizacoes da rede")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "dqn_loss.png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    episodes = 5000

    gamma = 0.99
    lr = 1e-3

    batch_size = 64
    buffer_size = 100000

    # Atualiza a rede alvo a cada N passos de ambiente
    target_update_steps = 500

    # Treinar a rede a cada N passos
    train_every = 1

    # Epsilon-greedy
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.995

    # Renderizacao
    render = True
    render_every = 100

    env = cm.Maze(
        render=render,
        continuous_obs=True,
        window_layers=5,
        reset_known_map_each_episode=True
    )

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    print(f"Dimensao da observacao: {obs_dim}")
    print(f"Numero de acoes: {num_actions}")

    # -------------------------
    # Agente
    # -------------------------
    agent = DQNAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        gamma=gamma,
        lr=lr,
        buffer_size=buffer_size
    )

    print(f"Dispositivo usado: {agent.device}")

    rewards_history = []
    successes = []
    success_rate = []
    loss_history = []

    eps = eps_start
    global_step = 0

    for episode in range(1, episodes + 1):

        state = env.reset()
        total_reward = 0.0

        while True:
            global_step += 1

            action = agent.select_action(state, eps)
            next_state, reward, done, _ = env.step(action)

            agent.buffer.push(state, action, reward, next_state, done)

            if global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    loss_history.append(loss)

            if global_step % target_update_steps == 0:
                agent.update_target_network()

            state = next_state
            total_reward += reward

            
            if episode % render_every == 0:
                env.render_known_map()

            if done:
                break

        # Atualiza epsilon apos cada episodio
        eps = max(eps_end, eps * eps_decay)

        success = env.reached_goal()

        rewards_history.append(total_reward)
        successes.append(1 if success else 0)
        success_rate.append(np.mean(successes[-50:]))

        avg_reward_50 = np.mean(rewards_history[-50:])

        print(
            f"Episodio {episode:4d}/{episodes} | "
            f"Reward: {total_reward:8.2f} | "
            f"Avg50: {avg_reward_50:8.2f} | "
            f"Sucesso50: {success_rate[-1]:.2f} | "
            f"Eps: {eps:.3f} | "
            f"Buffer: {len(agent.buffer)}"
        )

        if episode % 100 == 0:
            env.save_known_map_image(f"results/known_map_ep_{episode}.png")


    agent.save("results/dqn_model.pt")
    save_training_plots(rewards_history, success_rate, loss_history, "results")

    final_success_rate = np.mean(successes[-100:]) * 100.0
    print(f"\nTaxa de sucesso nos ultimos 100 episodios: {final_success_rate:.2f}%")

    env.close()
