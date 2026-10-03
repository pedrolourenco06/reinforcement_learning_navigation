# # -*- coding: utf-8 -*-
# """
# DQN para navegacao Ackermann em bitmap.

# VERSAO CORRIGIDA. Mudancas em relacao a original:
#   1. Double DQN + Dueling + Huber loss.
#   2. terminated vs truncated: so 'terminated' zera o bootstrap.
#   3. epsilon decai por PASSO GLOBAL (nao por episodio) e chega ao piso.
#   4. Retornos n-step (n=3).
#   5. Curriculo por distancia geodesica ao alvo.
#   6. Avaliacao deterministica num conjunto FIXO de poses, sorteadas uma vez.
#   7. Logging util: motivo do fim do episodio (goal / colisao / timeout).
# """
# import os
# import random
# import time
# from collections import deque, Counter

# import numpy as np
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# import ackermann_env2 as cm


# SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)


# # ======================================================================
# # REDE  (Dueling)
# # ======================================================================
# class QNetwork(nn.Module):
#     def __init__(self, input_dim, num_actions, hidden=256):
#         super().__init__()
#         self.feat = nn.Sequential(
#             nn.Linear(input_dim, hidden), nn.ReLU(),
#             nn.Linear(hidden, hidden), nn.ReLU(),
#         )
#         self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
#         self.adv = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, num_actions))

#     def forward(self, x):
#         h = self.feat(x)
#         v = self.val(h)
#         a = self.adv(h)
#         return v + a - a.mean(dim=1, keepdim=True)


# # ======================================================================
# # REPLAY BUFFER com retornos n-step
# # ======================================================================
# class NStepReplayBuffer:
#     """
#     Guarda transicoes ja agregadas em n passos:
#         (s_t, a_t, R_t^{(n)}, s_{t+n}, done, n_efetivo)
#     'done' e SOMENTE terminated. Truncation encerra o episodio mas a
#     transicao continua bootstrapando normalmente.
#     """

#     def __init__(self, capacity, n_step, gamma):
#         self.buffer = deque(maxlen=capacity)
#         self.n_step = n_step
#         self.gamma = gamma
#         self.pending = deque(maxlen=n_step)

#     def _make(self, k):
#         """Agrega as k primeiras transicoes pendentes num retorno n-step."""
#         R = 0.0
#         for i in range(k):
#             R += (self.gamma ** i) * self.pending[i][2]
#         s, a = self.pending[0][0], self.pending[0][1]
#         s_n, done = self.pending[k - 1][3], self.pending[k - 1][4]
#         return (s, a, R, s_n, float(done), k)

#     def push(self, s, a, r, s2, terminated):
#         self.pending.append((s, a, r, s2, terminated))
#         if len(self.pending) == self.n_step:
#             self.buffer.append(self._make(self.n_step))
#             self.pending.popleft()
#         if terminated:
#             self.flush()

#     def end_episode(self):
#         """Chamar no fim do episodio (terminated OU truncated)."""
#         self.flush()

#     def flush(self):
#         while len(self.pending) > 0:
#             self.buffer.append(self._make(len(self.pending)))
#             self.pending.popleft()

#     def sample(self, batch_size):
#         batch = random.sample(self.buffer, batch_size)
#         s, a, r, s2, d, n = zip(*batch)
#         return (np.asarray(s, dtype=np.float32),
#                 np.asarray(a, dtype=np.int64),
#                 np.asarray(r, dtype=np.float32),
#                 np.asarray(s2, dtype=np.float32),
#                 np.asarray(d, dtype=np.float32),
#                 np.asarray(n, dtype=np.float32))

#     def __len__(self):
#         return len(self.buffer)


# # ======================================================================
# # AGENTE
# # ======================================================================
# class DQNAgent:
#     def __init__(self, obs_dim, num_actions, gamma=0.99, lr=3e-4,
#                  buffer_size=300000, n_step=3, device=None):
#         self.obs_dim = obs_dim
#         self.num_actions = num_actions
#         self.gamma = gamma
#         self.n_step = n_step

#         if device is None:
#             if torch.cuda.is_available():
#                 self.device = torch.device("cuda")
#             else:
#                 try:
#                     import torch_directml
#                     self.device = torch_directml.device()
#                     print("Usando GPU AMD via DirectML")
#                 except ImportError:
#                     self.device = torch.device("cpu")
#         else:
#             self.device = device

#         self.q_net = QNetwork(obs_dim, num_actions).to(self.device)
#         self.target_net = QNetwork(obs_dim, num_actions).to(self.device)
#         self.target_net.load_state_dict(self.q_net.state_dict())
#         self.target_net.eval()

#         self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
#         self.buffer = NStepReplayBuffer(buffer_size, n_step, gamma)

#     # ------------------------------------------------------------------
#     @torch.no_grad()
#     def greedy_action(self, state):
#         t = torch.as_tensor(state, dtype=torch.float32,
#                             device=self.device).unsqueeze(0)
#         return int(self.q_net(t).argmax(dim=1).item())

#     def select_action(self, state, eps):
#         if random.random() < eps:
#             return random.randrange(self.num_actions)
#         return self.greedy_action(state)

#     # ------------------------------------------------------------------
#     def train_step(self, batch_size):
#         if len(self.buffer) < batch_size:
#             return None

#         s, a, r, s2, d, n = self.buffer.sample(batch_size)

#         s_t = torch.as_tensor(s, device=self.device)
#         a_t = torch.as_tensor(a, device=self.device).unsqueeze(1)
#         r_t = torch.as_tensor(r, device=self.device).unsqueeze(1)
#         s2_t = torch.as_tensor(s2, device=self.device)
#         d_t = torch.as_tensor(d, device=self.device).unsqueeze(1)
#         n_t = torch.as_tensor(n, device=self.device).unsqueeze(1)

#         q = self.q_net(s_t).gather(1, a_t)

#         with torch.no_grad():
#             # --- DOUBLE DQN: acao escolhida pela online, valor pela target ---
#             a_star = self.q_net(s2_t).argmax(dim=1, keepdim=True)
#             q_next = self.target_net(s2_t).gather(1, a_star)
#             gamma_n = self.gamma ** n_t
#             target = r_t + gamma_n * q_next * (1.0 - d_t)

#         loss = F.smooth_l1_loss(q, target)          # Huber

#         self.optimizer.zero_grad(set_to_none=True)
#         loss.backward()
#         nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
#         self.optimizer.step()
#         return float(loss.item())

#     def soft_update(self, tau=0.005):
#         with torch.no_grad():
#             for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
#                 tp.mul_(1.0 - tau).add_(tau * p)

#     def save(self, filename):
#         os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
#         torch.save({"q_net_state_dict": self.q_net.state_dict(),
#                     "obs_dim": self.obs_dim,
#                     "num_actions": self.num_actions,
#                     "gamma": self.gamma}, filename)


# # ======================================================================
# # CURRICULO
# # ======================================================================
# class Curriculo:
#     """Aumenta a distancia geodesica inicial conforme a taxa de sucesso sobe."""

#     def __init__(self, env, d_inicial=5.0, d_passo=2.5, limiar=0.70, janela=50):
#         self.env = env
#         self.d = d_inicial
#         self.d_passo = d_passo
#         self.d_max = env.geo_max
#         self.limiar = limiar
#         self.janela = janela
#         self.hist = deque(maxlen=janela)

#     def sortear(self):
#         return self.env.getRand(dist_max=self.d)

#     def registrar(self, sucesso):
#         self.hist.append(1.0 if sucesso else 0.0)
#         if (len(self.hist) == self.janela
#                 and np.mean(self.hist) >= self.limiar
#                 and self.d < self.d_max):
#             self.d = min(self.d_max, self.d + self.d_passo)
#             self.hist.clear()
#             print(f"  >> CURRICULO: distancia inicial maxima -> {self.d:.1f} m")
#             return True
#         return False


# # ======================================================================
# # AVALIACAO DETERMINISTICA
# # ======================================================================
# def construir_conjunto_avaliacao(env, n=30, seed=123):
#     rng = np.random.RandomState(seed)
#     estado = np.random.get_state()
#     np.random.seed(seed)
#     poses = []
#     while len(poses) < n:
#         try:
#             poses.append(env.getRand())
#         except RuntimeError:
#             break
#     np.random.set_state(estado)
#     return poses


# @torch.no_grad()
# def avaliar(env, agent, poses):
#     treinando = agent.q_net.training
#     agent.q_net.eval()
#     sucessos, passos, colisoes = 0, [], 0
#     try:
#         for pose in poses:
#             state = env.reset(initial_pose=pose)
#             for k in range(1, cm.MAX_STEPS + 1):
#                 action = agent.greedy_action(state)
#                 state, _, term, trunc, info = env.step(action)
#                 if term or trunc:
#                     break
#             if env.reached_goal():
#                 sucessos += 1
#                 passos.append(k)
#             elif info["collision"]:
#                 colisoes += 1
#     finally:
#         agent.q_net.train(treinando)
#     media = float(np.mean(passos)) if passos else float("inf")
#     return sucessos, len(poses), media, colisoes


# # ======================================================================
# # GRAFICOS
# # ======================================================================
# def moving_average(v, w=50):
#     if len(v) == 0:
#         return []
#     return [np.mean(v[max(0, i - w + 1):i + 1]) for i in range(len(v))]


# def save_training_plots(rewards, success_rate, losses, curr_d, outdir):
#     os.makedirs(outdir, exist_ok=True)

#     plt.figure(1); plt.clf()
#     plt.plot(rewards, "r", alpha=0.2, label="Recompensa")
#     plt.plot(moving_average(rewards, 50), "b", lw=2, label="Media movel 50")
#     plt.xlabel("Episodios"); plt.ylabel("Recompensa"); plt.legend(); plt.grid(True)
#     plt.title("DQN - Recompensa por episodio")
#     plt.savefig(os.path.join(outdir, "dqn_reward.png"), dpi=150, bbox_inches="tight")

#     plt.figure(2); plt.clf()
#     ax = plt.gca()
#     ax.plot(success_rate, "g", lw=2); ax.set_ylim(-0.05, 1.05)
#     ax.set_xlabel("Episodios"); ax.set_ylabel("Taxa de sucesso (movel 50)")
#     ax2 = ax.twinx(); ax2.plot(curr_d, "k--", alpha=0.6)
#     ax2.set_ylabel("Distancia geodesica maxima do curriculo (m)")
#     plt.title("DQN - Taxa de sucesso e curriculo")
#     plt.savefig(os.path.join(outdir, "dqn_success_rate.png"), dpi=150, bbox_inches="tight")

#     if losses:
#         plt.figure(3); plt.clf()
#         plt.plot(losses, alpha=0.2); plt.plot(moving_average(losses, 500), lw=2)
#         plt.yscale("log"); plt.xlabel("Atualizacoes"); plt.ylabel("Loss"); plt.grid(True)
#         plt.title("DQN - Loss")
#         plt.savefig(os.path.join(outdir, "dqn_loss.png"), dpi=150, bbox_inches="tight")
#     plt.close("all")


# # ======================================================================
# # TREINO
# # ======================================================================
# def main():
#     outdir = "results"
#     os.makedirs(outdir, exist_ok=True)

#     # ---------------- orcamento ----------------
#     total_steps = 600_000          # DECISOES de ambiente (com action_repeat=4)
#     learning_starts = 5_000
#     train_every = 2
#     batch_size = 128

#     gamma = 0.99
#     lr = 3e-4
#     n_step = 3
#     buffer_size = 300_000
#     tau = 0.005                    # soft update da target net

#     # epsilon POR PASSO GLOBAL
#     eps_start, eps_end = 1.0, 0.05
#     eps_decay_steps = int(0.40 * total_steps)

#     # render=True cria a janela pygame. O DESENHO em si so acontece quando
#     # "episode % render_every == 0" (ver o "if render and episode %
#     # render_every == 0: env.render()" dentro do laco de treino, abaixo).
#     # Com render=False a janela nunca e criada e render_every nao faz nada.
#     render = True
#     render_every = 50               # desenha 1 a cada 50 episodios
#     eval_every_steps = 25_000

#     # ---------------- ambiente ----------------
#     env = cm.AckermannEnv(
#         img="labirinto6.png",
#         xlim=np.array([0.0, 19.2]),
#         ylim=np.array([0.0, 24.0]),
#         alvo=np.array([13.2, 12.2]),
#         render=render,
#         continuous_obs=True,
#         window_layers=3,
#         window_res=0.6,
#         sensor_layers=6,           # 2.4 m  >  raio de giro 1.12 m
#         reset_known_map_each_episode=True,
#         wheelbase=0.4,
#         robot_length=0.545,
#         robot_width=0.415,
#         max_steering_deg=20.0,
#         speed=0.5,
#         dt=0.2,
#         allow_reverse=True,
#         action_repeat=4,
#         n_rays=16,
#         gamma=gamma,
#         goal_tol=0.70,             # ver diagnostico.py: 0.30 m e menor que o robo
#     )
#     env.seed(SEED)

#     obs_dim = env.observation_space.shape[0]
#     num_actions = env.action_space.n
#     print(f"obs_dim={obs_dim}  n_acoes={num_actions}  "
#           f"raio_giro={env.raio_giro_min:.2f} m  sensor={env.sensor_radius_m:.2f} m  "
#           f"geo_max={env.geo_max:.1f} m")

#     agent = DQNAgent(obs_dim, num_actions, gamma=gamma, lr=lr,
#                      buffer_size=buffer_size, n_step=n_step)
#     print(f"Dispositivo: {agent.device}")

#     # ATENCAO: nao comece o curriculo perto demais. Com raio de giro de 1.12 m,
#     # poses muito proximas ao alvo sao MAIS dificeis (o robo orbita sem entrar).
#     curriculo = Curriculo(env, d_inicial=5.0, d_passo=2.5, limiar=0.70, janela=50)
#     poses_eval = construir_conjunto_avaliacao(env, n=30)

#     rewards_hist, success_rate, loss_hist, curr_hist = [], [], [], []
#     successes = deque(maxlen=50)
#     motivos = Counter()

#     global_step = 0
#     episode = 0
#     melhor = -1.0
#     t0 = time.time()

#     while global_step < total_steps:
#         episode += 1
#         pose0 = curriculo.sortear()
#         state = env.reset(initial_pose=pose0)
#         total_reward = 0.0
#         ep_steps = 0

#         while True:
#             eps = max(eps_end, eps_start + (eps_end - eps_start)
#                       * global_step / eps_decay_steps)

#             action = agent.select_action(state, eps)
#             next_state, reward, terminated, truncated, info = env.step(action)

#             # SO 'terminated' zera o bootstrap. Truncation nao.
#             agent.buffer.push(state, action, reward, next_state, terminated)

#             global_step += 1
#             ep_steps += 1
#             total_reward += reward
#             state = next_state

#             if global_step >= learning_starts and global_step % train_every == 0:
#                 loss = agent.train_step(batch_size)
#                 if loss is not None:
#                     loss_hist.append(loss)
#                 agent.soft_update(tau)

#             if render and episode % render_every == 0:
#                 env.render()

#             if terminated or truncated:
#                 agent.buffer.end_episode()
#                 break

#         sucesso = env.reached_goal()
#         if sucesso:
#             motivos["goal"] += 1
#         elif info["collision"]:
#             motivos["colisao"] += 1
#         else:
#             motivos["timeout"] += 1

#         successes.append(1.0 if sucesso else 0.0)
#         rewards_hist.append(total_reward)
#         success_rate.append(float(np.mean(successes)))
#         curr_hist.append(curriculo.d)
#         curriculo.registrar(sucesso)

#         if episode % 20 == 0:
#             tot = sum(motivos.values())
#             fps = global_step / max(time.time() - t0, 1e-6)
#             print(f"ep {episode:5d} | steps {global_step:7d} | "
#                   f"R {total_reward:7.1f} | R50 {np.mean(rewards_hist[-50:]):7.1f} | "
#                   f"suc50 {success_rate[-1]:.2f} | eps {eps:.3f} | "
#                   f"d_curr {curriculo.d:4.1f} | buf {len(agent.buffer):6d} | "
#                   f"goal/col/to {motivos['goal']/tot:.2f}/"
#                   f"{motivos['colisao']/tot:.2f}/{motivos['timeout']/tot:.2f} | "
#                   f"{fps:.0f} steps/s")
#             motivos.clear()

#         if global_step % eval_every_steps < ep_steps:
#             s, n, mp, col = avaliar(env, agent, poses_eval)
#             taxa = s / max(n, 1)
#             print(f"  == AVALIACAO @ {global_step} steps: {s}/{n} "
#                   f"({100*taxa:.1f}%) | passos medios {mp:.1f} | colisoes {col}")
#             if taxa > melhor:
#                 melhor = taxa
#                 agent.save(os.path.join(outdir, "dqn_best.pt"))
#                 print(f"  == melhor modelo salvo ({100*taxa:.1f}%)")
#             env.save_known_map_image(os.path.join(outdir, f"known_map_{global_step}.png"))
#             save_training_plots(rewards_hist, success_rate, loss_hist, curr_hist, outdir)

#     agent.save(os.path.join(outdir, "dqn_model.pt"))
#     save_training_plots(rewards_hist, success_rate, loss_hist, curr_hist, outdir)

#     s, n, mp, col = avaliar(env, agent, poses_eval)
#     print(f"\nAVALIACAO FINAL: {s}/{n} ({100*s/max(n,1):.1f}%) | "
#           f"passos medios {mp:.1f} | colisoes {col}")
#     env.close()


# if __name__ == "__main__":
#     main()