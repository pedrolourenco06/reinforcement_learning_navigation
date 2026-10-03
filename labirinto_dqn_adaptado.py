# -*- coding: utf-8 -*-
# Treinamento Dueling Double DQN com o ambiente original intacto.
# Interface: 5 ações para frente; step retorna obs, reward, done, info.
# Timeout permanece terminal, coerente com o horizonte e a recompensa originais.

import os
import random
import time
from collections import deque, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import hashlib
import ackermann_env as cm


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class QNetwork(nn.Module):
    def __init__(self, input_dim, num_actions, hidden=256):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.val = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.adv = nn.Sequential(
            nn.Linear(hidden, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        h = self.feat(x)
        v = self.val(h)
        a = self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    """
    Guarda transicoes ja agregadas em n passos:
        (s_t, a_t, R_t^{(n)}, s_{t+n}, done, n_efetivo)
    'done' segue terminal() do ambiente original: colisão, alvo ou timeout.
    A máscara terminal zera o bootstrap nas três situações.
    """

    def __init__(self, capacity, n_step=3, gamma=0.99):
        self.buffer = deque(maxlen=capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.pending = deque(maxlen=n_step)

    def make_transition(self, k):
        # Agrega as k primeiras transições pendentes.
        total_reward = 0.0

        for i in range(k):
            total_reward += (self.gamma ** i) * self.pending[i][2]

        state = self.pending[0][0]
        action = self.pending[0][1]
        next_state = self.pending[k - 1][3]
        done = self.pending[k - 1][4]

        return (
            state,
            action,
            total_reward,
            next_state,
            float(done),
            k,
        )

    def push(self, state, action, reward, next_state, done):
        self.pending.append((state, action, reward, next_state, done))
        if len(self.pending) == self.n_step:
            self.buffer.append(self.make_transition(self.n_step))
            self.pending.popleft()
        if done:
            self.flush()

    def end_episode(self):
        """Chamar no fim de cada episódio para descarregar a fila pendente."""
        self.flush()

    def flush(self):
        while len(self.pending) > 0:
            self.buffer.append(self.make_transition(len(self.pending)))
            self.pending.popleft()

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, n_steps = zip(*batch)
        return (np.asarray(states, dtype=np.float32),
                np.asarray(actions, dtype=np.int64),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(next_states, dtype=np.float32),
                np.asarray(dones, dtype=np.float32),
                np.asarray(n_steps, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        obs_dim,
        num_actions,
        gamma=0.99,
        lr=3e-4,
        buffer_size=300000,
        device=None,
        n_step=3,
    ):
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_step = n_step

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                try:
                    import torch_directml
                    self.device = torch_directml.device()
                    print("Usando GPU AMD via DirectML")
                except ImportError:
                    self.device = torch.device("cpu")
        else:
            self.device = device

        self.q_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net = QNetwork(obs_dim, num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, n_step, gamma)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy_action(self, state):
        t = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return int(self.q_net(t).argmax(dim=1).item())

    def select_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.num_actions)
        return self.greedy_action(state)

    # ------------------------------------------------------------------
    def train_step(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones, n_steps = self.buffer.sample(batch_size)

        states_t = torch.as_tensor(states, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_states_t = torch.as_tensor(next_states, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device).unsqueeze(1)
        n_steps_t = torch.as_tensor(n_steps, device=self.device).unsqueeze(1)

        q_values = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            # Double DQN: ação escolhida pela rede principal, valor pela rede alvo.
            a_star = self.q_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states_t).gather(1, a_star)
            gamma_n = self.gamma ** n_steps_t
            targets = rewards_t + gamma_n * next_q_values * (1.0 - dones_t)

        loss = F.smooth_l1_loss(q_values, targets)          # Huber

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def update_target_network(self, tau=0.005):
        with torch.no_grad():
            for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
                tp.mul_(1.0 - tau).add_(tau * p)

    def soft_update(self, tau=0.005):
        # Compatibilidade com chamadas da versao 2.
        self.update_target_network(tau)

    def save(self, filename):
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        torch.save({"q_net_state_dict": self.q_net.state_dict(),
                    "target_net_state_dict": self.target_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "n_step": int(self.n_step),
                    "obs_dim": int(self.obs_dim),
                    "num_actions": int(self.num_actions),
                    "gamma": float(self.gamma)}, filename)

def assinatura_ambiente():
    # Identifica o código do ambiente e o mapa usados no treinamento.
    hashes = []

    for caminho in (cm.__file__, "labirinto6.png"):
        with open(caminho, "rb") as arquivo:
            hashes.append(
                hashlib.sha256(arquivo.read()).hexdigest()
            )

    return hashes


def zip_buffer(buffer):
    states, actions, rewards, next_states, dones, n_steps = zip(*buffer)

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.int64),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(next_states, dtype=np.float32),
        np.asarray(dones, dtype=np.float64),
        np.asarray(n_steps, dtype=np.int64),
    )


def salvar_checkpoint(filename, agent, curriculo, estado, configuracao):
    # Salvar somente entre episódios.
    if agent.buffer.pending:
        raise RuntimeError(
            "Salve o checkpoint somente ao terminar o episódio."
        )

    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

    temporario = filename + ".tmp"

    # Aproveita o método save já existente.
    agent.save(temporario)

    dados = torch.load(
        temporario,
        map_location="cpu",
        weights_only=False,
    )

    replay = None

    if len(agent.buffer):
        replay = [
            torch.from_numpy(coluna.copy())
            for coluna in zip_buffer(agent.buffer.buffer)
        ]

    np_state = np.random.get_state()

    dados.update({
        "checkpoint_version": 1,
        "configuracao": configuracao,
        "estado_treino": estado,

        "curriculo": {
            "d": curriculo.d,
            "d_passo": curriculo.d_passo,
            "limiar": curriculo.limiar,
            "janela": curriculo.janela,
            "hist": list(curriculo.hist),
        },

        "replay": replay,
        "buffer_capacity": agent.buffer.buffer.maxlen,

        "random_state": random.getstate(),

        "numpy_state": [
            np_state[0],
            torch.tensor(np_state[1].astype(np.int64)),
            int(np_state[2]),
            int(np_state[3]),
            float(np_state[4]),
        ],

        "torch_state": torch.get_rng_state(),

        "cuda_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else []
        ),
    })

    torch.save(dados, temporario)

    # Substitui o checkpoint anterior somente após terminar a escrita.
    os.replace(temporario, filename)


def carregar_checkpoint(filename, agent, curriculo, configuracao):
    dados = torch.load(
        filename,
        map_location="cpu",
        weights_only=False,
    )

    for chave, esperado in (
        ("obs_dim", agent.obs_dim),
        ("num_actions", agent.num_actions),
        ("gamma", agent.gamma),
        ("n_step", agent.n_step),
    ):
        if dados.get(chave) != esperado:
            raise ValueError(
                f"Checkpoint incompatível: "
                f"{chave}={dados.get(chave)}, esperado {esperado}."
            )

    completo = "checkpoint_version" in dados

    if completo:
        if (
            dados["checkpoint_version"] != 1
            or dados["configuracao"] != configuracao
        ):
            raise ValueError(
                "Mapa, ambiente ou parâmetros diferentes "
                "do checkpoint completo."
            )

    agent.q_net.load_state_dict(dados["q_net_state_dict"])
    agent.target_net.load_state_dict(dados["target_net_state_dict"])
    agent.optimizer.load_state_dict(dados["optimizer_state_dict"])
    agent.target_net.eval()

    # O modelo antigo contém somente redes, otimizador e metadados.
    if not completo:
        return None

    c = dados["curriculo"]

    curriculo.d = c["d"]
    curriculo.d_passo = c["d_passo"]
    curriculo.limiar = c["limiar"]
    curriculo.janela = c["janela"]
    curriculo.hist = deque(c["hist"], maxlen=c["janela"])

    agent.buffer.buffer = deque(
        maxlen=dados["buffer_capacity"]
    )

    if dados["replay"] is not None:
        colunas = [t.numpy() for t in dados["replay"]]
        agent.buffer.buffer.extend(zip(*colunas))

    agent.buffer.pending.clear()

    random.setstate(dados["random_state"])

    ns = dados["numpy_state"]

    np.random.set_state((
        ns[0],
        ns[1].numpy().astype(np.uint32),
        ns[2],
        ns[3],
        ns[4],
    ))

    torch.set_rng_state(dados["torch_state"])

    if torch.cuda.is_available() and dados["cuda_states"]:
        if len(dados["cuda_states"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(dados["cuda_states"])
        else:
            print(
                "AVISO: número de GPUs diferente; "
                "sequência CUDA não restaurada."
            )

    return dados["estado_treino"]

def moving_average(values, window=50):
    if len(values) == 0:
        return []

    avg = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        avg.append(np.mean(values[start:i + 1]))

    return avg


def save_training_plots(rewards, success_rate, losses, output_dir, curr_d=None):
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(1)
    plt.clf()
    plt.plot(rewards, "r", alpha=0.2, label="Recompensa")
    plt.plot(moving_average(rewards, 50), "b", lw=2, label="Media movel 50")
    plt.xlabel("Episodios")
    plt.ylabel("Recompensa")
    plt.legend()
    plt.grid(True)
    plt.title("DQN - Recompensa por episodio")
    plt.savefig(os.path.join(output_dir, "dqn_reward.png"), dpi=150, bbox_inches="tight")

    plt.figure(2)
    plt.clf()
    ax = plt.gca()
    ax.plot(success_rate, "g", lw=2)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Episodios")
    ax.set_ylabel("Taxa de sucesso (movel 50)")
    if curr_d is not None:
        ax2 = ax.twinx()
        ax2.plot(curr_d, "k--", alpha=0.6)
        ax2.set_ylabel("Distancia euclidiana maxima do curriculo (m)")
    plt.title("DQN - Taxa de sucesso e curriculo")
    plt.savefig(os.path.join(output_dir, "dqn_success_rate.png"), dpi=150, bbox_inches="tight")

    if losses:
        plt.figure(3)
        plt.clf()
        plt.plot(losses, alpha=0.2)
        plt.plot(moving_average(losses, 500), lw=2)
        plt.yscale("log")
        plt.xlabel("Atualizacoes")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.title("DQN - Loss")
        plt.savefig(os.path.join(output_dir, "dqn_loss.png"), dpi=150, bbox_inches="tight")
    plt.close("all")


def avaliar_27_poses(env, agent):
    sucessos = 0
    passos_sucessos = []

    estava_treinando = agent.q_net.training
    agent.q_net.eval()

    try:
        for x in [15.4, 15.6, 15.8]:
            for y in [8.0, 9.0, 10.0]:
                for angulo in [85, 90, 95]:
                    inicio = np.array(
                        [x, y, np.deg2rad(angulo)],
                        dtype=np.float32,
                    )

                    state = env.reset(initial_pose=inicio)

                    for passo in range(1, cm.MAX_STEPS + 1):
                        # Sem exploração e sem consumir números aleatórios.
                        with torch.no_grad():
                            entrada = torch.as_tensor(
                                state,
                                dtype=torch.float32,
                                device=agent.device,
                            ).unsqueeze(0)

                            action = int(
                                agent.q_net(entrada).argmax(dim=1).item()
                            )

                        state, _, done, _ = env.step(action)

                        if done:
                            break

                    if env.reached_goal():
                        sucessos += 1
                        passos_sucessos.append(passo)

    finally:
        agent.q_net.train(estava_treinando)

    media_passos = (
        float(np.mean(passos_sucessos))
        if passos_sucessos
        else float("inf")
    )

    return sucessos, media_passos


def sortear_pose(env, dist_max=None, max_tentativas=20000):
    # Amostragem limitada, sem alterar a pose ou o mapa conhecido do ambiente.
    # Distância euclidiana: não verifica conectividade através de corredores.
    for _ in range(max_tentativas):
        pose = np.array([
            np.random.uniform(env.xlim[0], env.xlim[1]),
            np.random.uniform(env.ylim[0], env.ylim[1]),
            np.random.uniform(-np.pi, np.pi),
        ], dtype=np.float32)

        distancia = np.linalg.norm(pose[:2] - env.alvo)

        # Mesma tolerância de chegada definida no ambiente fornecido.
        if distancia <= 0.30:
            continue
        if dist_max is not None and distancia > dist_max:
            continue
        if env.collision(pose):
            continue

        return pose

    raise RuntimeError(
        "Não foi possível sortear uma pose válida. "
        "Verifique o mapa, o alvo e a distância máxima do currículo."
    )


class Curriculo:
    # Currículo do treino, sem campo geodésico nem alterações no ambiente.
    def __init__(self, env, d_inicial=5.0, d_passo=2.5, limiar=0.70, janela=50):
        self.env = env
        largura = float(env.xlim[1] - env.xlim[0])
        altura = float(env.ylim[1] - env.ylim[0])
        self.d_max = float(np.hypot(largura, altura))
        self.d = min(d_inicial, self.d_max)
        self.d_passo = d_passo
        self.limiar = limiar
        self.janela = janela
        self.hist = deque(maxlen=janela)

    def sortear(self):
        return sortear_pose(self.env, dist_max=self.d)

    def registrar(self, sucesso):
        self.hist.append(1.0 if sucesso else 0.0)

        if (
            len(self.hist) == self.janela
            and np.mean(self.hist) >= self.limiar
            and self.d < self.d_max
        ):
            self.d = min(self.d_max, self.d + self.d_passo)
            self.hist.clear()
            print(f"  >> CURRÍCULO: distância euclidiana máxima -> {self.d:.1f} m")
            return True

        return False


def construir_conjunto_avaliacao(env, n=30, seed=123):
    estado = np.random.get_state()
    poses = []

    try:
        np.random.seed(seed)
        for _ in range(n):
            poses.append(sortear_pose(env))
    finally:
        np.random.set_state(estado)

    return poses


@torch.no_grad()
def avaliar(env, agent, poses):
    treinando = agent.q_net.training
    agent.q_net.eval()
    sucessos, passos, colisoes = 0, [], 0
    try:
        for pose in poses:
            state = env.reset(initial_pose=pose)
            for k in range(1, cm.MAX_STEPS + 1):
                action = agent.greedy_action(state)
                state, _, done, info = env.step(action)
                if done:
                    break
            if env.reached_goal():
                sucessos += 1
                passos.append(k)
            elif info["collision"]:
                colisoes += 1
    finally:
        agent.q_net.train(treinando)
    media = float(np.mean(passos)) if passos else float("inf")
    return sucessos, len(poses), media, colisoes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DQN com retomada por checkpoint"
    )

    parser.add_argument(
        "--retomar",
        default=None,
        help="Modelo antigo ou checkpoint completo",
    )

    parser.add_argument(
        "--passos-adicionais",
        type=int,
        default=200_000,
    )

    parser.add_argument(
        "--passo-inicial",
        type=int,
        default=None,
        help="Obrigatório somente para modelo antigo sem contador",
    )

    parser.add_argument(
        "--curriculo-inicial",
        type=float,
        default=None,
        help="Obrigatório somente para modelo antigo sem currículo",
    )

    parser.add_argument(
        "--episodio-inicial",
        type=int,
        default=0,
        help="Modelo antigo: 0 reinicia somente a numeração",
    )

    parser.add_argument("--saida", default=None)

    args = parser.parse_args()

    if args.passos_adicionais <= 0:
        parser.error("--passos-adicionais deve ser positivo")

    output_dir = args.saida or (
        "results_retomada" if args.retomar else "results"
    )

    os.makedirs(output_dir, exist_ok=True)

    # ---------------- orcamento ----------------
    total_steps = 600_000          # uma decisão = um passo físico neste ambiente
    learning_starts = 5_000
    train_every = 2
    batch_size = 128

    gamma = 0.99
    lr = 3e-4
    n_step = 3
    buffer_size = 300_000
    tau = 0.005                    # soft update da target net

    # epsilon POR PASSO GLOBAL
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 240_000

    # render=True cria a janela pygame. O DESENHO em si so acontece quando
    # "episode % render_every == 0" (ver o "if render and episode %
    # render_every == 0: env.render()" dentro do laco de treino, abaixo).
    # Com render=False a janela nunca e criada e render_every nao faz nada.
    render = True
    render_every = 50               # desenha 1 a cada 50 episodios
    eval_every_steps = 25_000

    # ---------------- ambiente ----------------
    env = cm.AckermannEnv(
        img="labirinto6.png",
        xlim=np.array([0.0, 19.2]),
        ylim=np.array([0.0, 24.0]),
        alvo=np.array([13.2, 12.2]),
        render=render,
        continuous_obs=True,
        window_layers=5,
        reset_known_map_each_episode=True,
        wheelbase=0.4,
        robot_length=0.545,
        robot_width=0.415,
        max_steering_deg=20.0,
        speed=0.5,
        dt=0.2,
    )
    env.seed(SEED)

    obs_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print(f"Dimensão da observação: {obs_dim} | Número de ações: {num_actions}")

    agent = DQNAgent(
        obs_dim=obs_dim,
        num_actions=num_actions,
        gamma=gamma,
        lr=lr,
        buffer_size=buffer_size,
        n_step=n_step,
    )
    print(f"Dispositivo: {agent.device}")

    # Distância em linha reta ao alvo; não equivale à distância pelo labirinto.
    curriculo = Curriculo(env, d_inicial=5.0, d_passo=2.5, limiar=0.70, janela=50)
    poses_avaliacao = construir_conjunto_avaliacao(env, n=30)

    rewards_history = []
    success_rate = []
    loss_history = []
    curr_history = []
    successes = deque(maxlen=50)
    motivos = Counter()

    global_step = 0
    episode = 0
    melhor_sucesso_validacao = -1.0

    configuracao = {
        "assinatura_ambiente": assinatura_ambiente(),
        "gamma": gamma,
        "lr": lr,
        "n_step": n_step,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "train_every": train_every,
        "tau": tau,
        "learning_starts": learning_starts,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "eps_decay_steps": eps_decay_steps,
        "eval_every_steps": eval_every_steps,

        # Valores da configuração do ambiente atual.
        "ambiente": [
            0.0, 19.2,
            0.0, 24.0,
            13.2, 12.2,
            5,
            0.4, 0.545, 0.415,
            20.0, 0.5, 0.2,
        ],
    }

    treino_liberado_em = learning_starts

    if args.retomar:
        estado = carregar_checkpoint(
            args.retomar,
            agent,
            curriculo,
            configuracao,
        )

        if estado is None:
            # Primeira retomada a partir do modelo antigo.
            if (
                args.passo_inicial is None
                or args.curriculo_inicial is None
            ):
                parser.error(
                    "Modelo antigo: informe --passo-inicial "
                    "e --curriculo-inicial."
                )

            if (
                args.passo_inicial < 0
                or not 0.30 < args.curriculo_inicial <= curriculo.d_max
            ):
                parser.error(
                    "Passo inicial ou distância do currículo inválidos."
                )

            global_step = args.passo_inicial
            episode = args.episodio_inicial
            curriculo.d = args.curriculo_inicial

            # O buffer antigo não foi salvo.
            treino_liberado_em = global_step + learning_starts

            print(
                "Modelo antigo carregado. "
                "Buffer e históricos não estavam salvos; "
                "coletando 5000 novos passos antes de atualizar as redes."
            )

        else:
            # Retomada a partir de um checkpoint completo.
            global_step = estado["global_step"]
            episode = estado["episode"]
            melhor_sucesso_validacao = estado["melhor"]

            rewards_history = estado["rewards"]
            success_rate = estado["success_rate"]
            loss_history = estado["losses"]
            curr_history = estado["curr_history"]

            successes = deque(
                estado["successes"],
                maxlen=50,
            )

            motivos = Counter(estado["motivos"])

            poses_avaliacao = [
                np.asarray(p, dtype=np.float32)
                for p in estado["poses"]
            ]

            treino_liberado_em = estado["treino_liberado_em"]

        total_steps = global_step + args.passos_adicionais

        s, n, mp, col = avaliar(
            env,
            agent,
            poses_avaliacao,
        )

        melhor_sucesso_validacao = max(
            melhor_sucesso_validacao,
            s / max(n, 1),
        )

        agent.save(
            os.path.join(output_dir, "dqn_inicio_retomada.pt")
        )

        if s / max(n, 1) >= melhor_sucesso_validacao:
            agent.save(
                os.path.join(output_dir, "dqn_best.pt")
            )

        print(
            f"RETOMADA: passo {global_step}, "
            f"currículo {curriculo.d:.1f} m, "
            f"buffer {len(agent.buffer)}, "
            f"avaliação {s}/{n}; "
            f"limite {total_steps} passos."
        )

    def gravar_estado():
        estado = {
            "global_step": global_step,
            "episode": episode,
            "melhor": melhor_sucesso_validacao,
            "rewards": [float(r) for r in rewards_history],
            "success_rate": success_rate,
            "losses": loss_history,
            "curr_history": curr_history,
            "successes": list(successes),
            "motivos": dict(motivos),
            "poses": [p.tolist() for p in poses_avaliacao],
            "treino_liberado_em": treino_liberado_em,
        }

        salvar_checkpoint(
            os.path.join(output_dir, "dqn_checkpoint.pt"),
            agent,
            curriculo,
            estado,
            configuracao,
        )

    passo_inicio_sessao = global_step
    tempo_inicial = time.time()

    if args.retomar:
        gravar_estado()

    while global_step < total_steps:
        episode += 1
        initial_pose = curriculo.sortear()
        state = env.reset(initial_pose=initial_pose)
        total_reward = 0.0
        episode_steps = 0

        while True:
            eps = max(eps_end, eps_start + (eps_end - eps_start)
                      * global_step / eps_decay_steps)

            action = agent.select_action(state, eps)
            next_state, reward, done, info = env.step(action)

            # O ambiente considera colisão, alvo e limite de passos terminais.
            # Preserva a penalidade de timeout e o horizonte originais.
            agent.buffer.push(state, action, reward, next_state, done)

            global_step += 1
            episode_steps += 1
            total_reward += reward
            state = next_state

            if global_step >= treino_liberado_em and global_step % train_every == 0:
                loss = agent.train_step(batch_size)
                if loss is not None:
                    loss_history.append(loss)
                agent.update_target_network(tau)

            if render and episode % render_every == 0:
                env.render()

            if done:
                agent.buffer.end_episode()
                break

        sucesso = env.reached_goal()
        if sucesso:
            motivos["goal"] += 1
        elif info["collision"]:
            motivos["colisao"] += 1
        else:
            motivos["timeout"] += 1

        successes.append(1.0 if sucesso else 0.0)
        rewards_history.append(total_reward)
        success_rate.append(float(np.mean(successes)))
        curr_history.append(curriculo.d)
        curriculo.registrar(sucesso)

        if episode % 20 == 0:
            tot = sum(motivos.values())
            fps = (
                (global_step - passo_inicio_sessao)
                / max(time.time() - tempo_inicial, 1e-6)
            )
            print(f"ep {episode:5d} | steps {global_step:7d} | "
                  f"R {total_reward:7.1f} | R50 {np.mean(rewards_history[-50:]):7.1f} | "
                  f"suc50 {success_rate[-1]:.2f} | eps {eps:.3f} | "
                  f"d_curr {curriculo.d:4.1f} | buf {len(agent.buffer):6d} | "
                  f"goal/col/to {motivos['goal']/tot:.2f}/"
                  f"{motivos['colisao']/tot:.2f}/{motivos['timeout']/tot:.2f} | "
                  f"{fps:.0f} steps/s")
            motivos.clear()

        # if global_step % eval_every_steps < episode_steps:
        #     s, n, mp, col = avaliar(env, agent, poses_avaliacao)
        #     taxa = s / max(n, 1)
        #     print(f"  == AVALIACAO @ {global_step} steps: {s}/{n} "
        #           f"({100*taxa:.1f}%) | passos medios {mp:.1f} | colisoes {col}")
        #     if taxa > melhor_sucesso_validacao:
        #         melhor_sucesso_validacao = taxa
        #         agent.save(os.path.join(output_dir, "dqn_best.pt"))
        #         print(f"  == melhor modelo salvo ({100*taxa:.1f}%)")
        #     env.save_known_map_image(os.path.join(output_dir, f"known_map_{global_step}.png"))
        #     save_training_plots(rewards_history, success_rate, loss_history, output_dir, curr_history)
        if global_step % eval_every_steps < episode_steps:
            gravar_estado()

    gravar_estado()

    agent.save(os.path.join(output_dir, "dqn_model.pt"))
    save_training_plots(
        rewards_history,
        success_rate,
        loss_history,
        output_dir,
        curr_history,
    )
        

    s, n, mp, col = avaliar(env, agent, poses_avaliacao)
    print(f"\nAVALIACAO FINAL: {s}/{n} ({100*s/max(n,1):.1f}%) | "
          f"passos medios {mp:.1f} | colisoes {col}")
    env.close()
