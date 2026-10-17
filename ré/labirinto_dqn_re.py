# -*- coding: utf-8 -*-
# Treinamento Dueling Double DQN com o ambiente original intacto.
# Interface: 5 ações para frente e 3 de ré; observação original de 251 valores.
# Timeout permanece terminal, coerente com o horizonte e a recompensa originais.

import os
import argparse
import hashlib
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

import ackermann_env_re as cm
from pathlib import Path
import csv


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
        torch.save(para_cpu({"q_net_state_dict": self.q_net.state_dict(),
                    "target_net_state_dict": self.target_net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "n_step": int(self.n_step),
                    "obs_dim": int(self.obs_dim),
                    "num_actions": int(self.num_actions),
                    "gamma": float(self.gamma)}), filename)



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


# ---------- Transferência e checkpoints ----------
CONFIG_ENV = dict(
    img='labirinto6.png', xlim=np.array([0.0, 19.2]), ylim=np.array([0.0, 24.0]),
    alvo=np.array([13.2, 12.2]), continuous_obs=True, window_layers=5,
    reset_known_map_each_episode=True, wheelbase=0.4, robot_length=0.545,
    robot_width=0.415, max_steering_deg=20.0, speed=0.5, dt=0.2, allow_reverse=True,
)
ACOES = [[-20, 1], [-10, 1], [0, 1], [10, 1], [20, 1], [-20, -1], [0, -1], [20, -1]]


def para_cpu(valor):
    # Evita a serialização específica de DirectML nos NOVOS arquivos.
    if torch.is_tensor(valor):
        return valor.detach().cpu()
    if isinstance(valor, dict):
        return {k: para_cpu(v) for k, v in valor.items()}
    if isinstance(valor, (tuple, list)):
        return type(valor)(para_cpu(v) for v in valor)
    if isinstance(valor, np.generic):
        return valor.item()
    return valor


def ler_modelo(filename, confiar=False):
    try:
        return torch.load(filename, map_location='cpu', weights_only=not confiar)
    except Exception as erro:
        if not confiar and 'Weights only load failed' in str(erro):
            raise RuntimeError('Para seu checkpoint antigo DirectML, acrescente --confiar-checkpoint.') from erro
        raise


def transferir_rede(rede, pesos):
    novos = rede.state_dict()
    for chave in novos:
        if chave in ('adv.2.weight', 'adv.2.bias'):
            antigo = pesos[chave].detach().cpu()
            if antigo.shape[0] != 5 or novos[chave].shape[0] != 8:
                raise ValueError('A transferência requer exatamente 5 -> 8 ações.')
            novos[chave][:5].copy_(antigo)
            # A média das 8 vantagens permanece igual à média das 5 antigas.
            novos[chave][5:].copy_(antigo.mean(dim=0, keepdim=True).expand_as(novos[chave][5:]))
        else:
            novos[chave].copy_(pesos[chave])
    rede.load_state_dict(novos)


def restaurar_buffer(agent, dados):
    agent.buffer.buffer.clear()
    agent.buffer.pending.clear()
    if dados.get('replay') is not None:
        colunas = [t.detach().cpu().numpy() for t in dados['replay']]
        if len(colunas) != 6 or colunas[0].shape[1:] != (agent.obs_dim,):
            raise ValueError('Formato de replay incompatível.')
        if len(set(len(c) for c in colunas)) != 1:
            raise ValueError('Colunas do replay com tamanhos diferentes.')
        if np.any(colunas[1] < 0) or np.any(colunas[1] >= int(dados['num_actions'])):
            raise ValueError('Ações inválidas no replay.')
        agent.buffer.buffer.extend(zip(*colunas))


def transferir_modelo(agent, dados):
    for chave, esperado in [('obs_dim', 251), ('num_actions', 5), ('gamma', agent.gamma), ('n_step', agent.n_step)]:
        if dados.get(chave) != esperado:
            raise ValueError(f'Transferência incompatível: {chave}={dados.get(chave)}, esperado {esperado}.')
    transferir_rede(agent.q_net, dados['q_net_state_dict'])
    transferir_rede(agent.target_net, dados['target_net_state_dict'])
    agent.target_net.eval()
    restaurar_buffer(agent, dados)
    # O otimizador recém-criado é mantido: a camada de saída mudou de tamanho.


def salvar_checkpoint(filename, agent, curriculo, estado, configuracao):
    if agent.buffer.pending:
        raise RuntimeError('Checkpoint deve ser salvo entre episódios.')
    filename = str(filename)
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    temporario = filename + '.tmp'
    agent.save(temporario)
    dados = ler_modelo(temporario)
    replay = None
    if len(agent.buffer):
        colunas = zip(*agent.buffer.buffer)
        tipos = [np.float32, np.int64, np.float64, np.float32, np.float64, np.int64]
        replay = [torch.from_numpy(np.asarray(c, dtype=t)) for c, t in zip(colunas, tipos)]
    ns = np.random.get_state()
    dados.update(checkpoint_version=2, configuracao=configuracao, estado_treino=estado,
                 curriculo=dict(d=float(curriculo.d), d_passo=curriculo.d_passo,
                                limiar=curriculo.limiar, janela=curriculo.janela, hist=list(curriculo.hist)),
                 replay=replay, random_state=random.getstate(),
                 numpy_state=[ns[0], torch.tensor(ns[1].astype(np.int64)), int(ns[2]), int(ns[3]), float(ns[4])],
                 torch_state=torch.get_rng_state(),
                 cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    torch.save(para_cpu(dados), temporario)
    os.replace(temporario, filename)


def carregar_checkpoint(agent, curriculo, dados, configuracao):
    if dados.get('checkpoint_version') != 2 or dados.get('configuracao') != configuracao:
        raise ValueError('Use um checkpoint completo da versão com ré e a mesma configuração.')
    if dados.get('num_actions') != 8 or dados.get('obs_dim') != 251:
        raise ValueError('Dimensões incompatíveis.')
    agent.q_net.load_state_dict(dados['q_net_state_dict'])
    agent.target_net.load_state_dict(dados['target_net_state_dict'])
    agent.optimizer.load_state_dict(dados['optimizer_state_dict'])
    restaurar_buffer(agent, dados)
    c = dados['curriculo']
    for k in ('d', 'd_passo', 'limiar', 'janela'):
        setattr(curriculo, k, c[k])
    curriculo.hist = deque(c['hist'], maxlen=c['janela'])
    random.setstate(dados['random_state'])
    ns = dados['numpy_state']
    np.random.set_state((ns[0], ns[1].numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(dados['torch_state'])
    if torch.cuda.is_available() and len(dados['cuda_states']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(dados['cuda_states'])
    return dados['estado_treino']


def precisa_re(env, pose):
    # Não altera o ambiente. Procura poses sem saída imediata para frente,
    # mas com ao menos uma ação de ré sem colisão no próximo passo.
    def livre(a):
        futura = env.ackermann_step(env.action_to_steering(a), float(env.action_directions[a]), pose)
        return not env.collision(futura)
    return not any(livre(a) for a in range(5)) and any(livre(a) for a in range(5, 8))


def sortear_inicio_misto(env, curriculo, fracao_parede, fracao_inferior):
    sorteio = np.random.random()
    if sorteio >= fracao_parede + fracao_inferior:
        return curriculo.sortear(), 'curriculo'
    tipo = 'parede' if sorteio < fracao_parede else 'inferior'
    limite_y = env.ylim[0] + (env.ylim[1] - env.ylim[0]) / 3.0
    for _ in range(300):
        pose = sortear_pose(env)
        if tipo == 'parede' and precisa_re(env, pose):
            return pose, tipo
        if tipo == 'inferior' and pose[1] <= limite_y:
            return pose, tipo
    # Um mapa pode não oferecer poses desse tipo: mantém o sorteio limitado.
    return curriculo.sortear(), 'curriculo_fallback'


def main():
    parser = argparse.ArgumentParser(description='Transfere DQN de 5 ações para 8, incluindo ré.')
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument('--transferir', type=Path, help='Modelo/checkpoint original de 5 ações')
    grupo.add_argument('--retomar', type=Path, help='Checkpoint completo desta versão de 8 ações')
    parser.add_argument('--confiar-checkpoint', action='store_true')
    parser.add_argument('--passos-adicionais', type=int, default=200000)
    parser.add_argument('--passo-inicial', type=int, default=None, help='Necessário só se o modelo antigo não tem contador')
    parser.add_argument('--curriculo-inicial', type=float, default=None, help='Necessário só se o modelo antigo não tem currículo')
    parser.add_argument('--saida', type=Path, default=Path('results_re'))
    parser.add_argument('--sem-render', action='store_true')
    parser.add_argument('--fracao-parede', type=float, default=0.2)
    parser.add_argument('--fracao-inferior', type=float, default=0.2)
    parser.add_argument('--eps-transferencia', type=float, default=0.15)
    args = parser.parse_args()
    if args.passos_adicionais <= 0 or not 0 <= args.fracao_parede <= 1 or not 0 <= args.fracao_inferior <= 1 or args.fracao_parede + args.fracao_inferior > 1:
        parser.error('Passos devem ser positivos e frações válidas com soma <= 1.')
    if not 0.05 <= args.eps_transferencia <= 1:
        parser.error('--eps-transferencia deve estar entre 0.05 e 1.')
    output_dir = args.saida
    source = args.transferir or args.retomar
    # Evita misturar resultados ou sobrescrever um experimento já existente na migração.
    if args.transferir and output_dir.exists() and any(output_dir.iterdir()):
        parser.error('A pasta de saída já contém arquivos. Use outra --saida ou --retomar para continuar.')
    if source.resolve().parent == output_dir.resolve() and args.transferir:
        parser.error('Escolha uma pasta de saída diferente da pasta do modelo original.')

    learning_starts, train_every, batch_size = 5000, 2, 128
    gamma, lr, n_step, buffer_size, tau = 0.99, 3e-4, 3, 300000, 0.005
    eps_decay_steps, eval_every_steps, render_every = 50000, 25000, 50
    configuracao = dict(gamma=gamma, lr=lr, n_step=n_step, buffer_size=buffer_size, tau=tau,
        learning_starts=learning_starts, train_every=train_every, batch_size=batch_size,
        eps_decay_steps=eps_decay_steps, eval_every_steps=eval_every_steps,
        fracao_parede=args.fracao_parede, fracao_inferior=args.fracao_inferior,
        eps_transferencia=args.eps_transferencia, acoes=ACOES,
        ambiente={k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in CONFIG_ENV.items()},
        mapa_sha256=hashlib.sha256(Path(CONFIG_ENV['img']).read_bytes()).hexdigest(),
        ambiente_sha256=hashlib.sha256(Path(cm.__file__).read_text().encode()).hexdigest())
    agent = DQNAgent(251, 8, gamma=gamma, lr=lr, buffer_size=buffer_size, n_step=n_step)
    dados = ler_modelo(source, args.confiar_checkpoint)
    env = cm.AckermannEnv(**CONFIG_ENV, render=not args.sem_render)
    env.seed(SEED)
    curriculo = Curriculo(env)
    try:
        if args.transferir:
            cfg_antiga = dados.get('configuracao', {})
            assinatura = cfg_antiga.get('assinatura_ambiente')
            if assinatura and assinatura[1] != configuracao['mapa_sha256']:
                raise ValueError('O mapa difere do mapa usado no checkpoint original.')
            transferir_modelo(agent, dados)
            origem = dados.get('estado_treino', {})
            global_step = origem.get('global_step', args.passo_inicial)
            d = dados.get('curriculo', {}).get('d', args.curriculo_inicial)
            if global_step is None or d is None:
                parser.error('Modelo sem estado completo: informe --passo-inicial e --curriculo-inicial.')
            if global_step < 0 or not 0.30 < d <= curriculo.d_max:
                parser.error('Contador ou distância de currículo inválidos.')
            curriculo.d = d
            poses = origem.get('poses')
            if poses is None:
                poses = [p.tolist() for p in construir_conjunto_avaliacao(env)]
            estado = dict(global_step=int(global_step), episode=0, inicio_re=int(global_step),
                          treino_liberado_em=int(global_step) + learning_starts, melhor=-1.0,
                          rewards=[], success_rate=[], losses=[], curr_history=[], successes=[],
                          motivos={}, poses=poses, avaliacoes=[], origem=str(source.resolve()))
            # Métricas novas: tarefa com novas ações e nova distribuição de inícios.
            print(f'Transferência 5 -> 8 | buffer reaproveitado: {len(agent.buffer)} | currículo {d:.1f} m')
            print('Otimizador reiniciado. Coleta de 5000 passos antes das atualizações; epsilon 0.15 -> 0.05 por padrão.')
        else:
            estado = carregar_checkpoint(agent, curriculo, dados, configuracao)
        del dados
        output_dir.mkdir(parents=True, exist_ok=True)
        poses_avaliacao = [np.asarray(p, dtype=np.float32) for p in estado['poses']]
        successes = deque(estado['successes'], maxlen=50)
        motivos = Counter(estado['motivos'])
        limite = estado['global_step'] + args.passos_adicionais
        inicio_sessao = estado['global_step']
        t0 = time.time()

        def gravar():
            estado['successes'] = list(successes)
            estado['motivos'] = dict(motivos)
            salvar_checkpoint(output_dir / 'dqn_checkpoint.pt', agent, curriculo, estado, configuracao)

        def validar():
            s, n, mp, col = avaliar(env, agent, poses_avaliacao)
            taxa = s / max(n, 1)
            estado['avaliacoes'].append(dict(passos=estado['global_step'], sucessos=s, episodios=n,
                                             taxa=taxa, passos_medios=mp, colisoes=col))
            with (output_dir / 'avaliacoes.csv').open('w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=list(estado['avaliacoes'][0]))
                writer.writeheader(); writer.writerows(estado['avaliacoes'])
            print(f"AVALIAÇÃO @ {estado['global_step']}: {s}/{n} ({100*taxa:.1f}%) | colisões {col}", flush=True)
            if taxa > estado['melhor']:
                estado['melhor'] = taxa
                agent.save(str(output_dir / 'dqn_best.pt'))

        if args.transferir:
            agent.save(str(output_dir / 'dqn_inicio_transferencia.pt'))
            validar()
        gravar()
        print(f"Dispositivo {agent.device} | passos {estado['global_step']} -> {limite}", flush=True)
        while estado['global_step'] < limite:
            estado['episode'] += 1
            pose, tipo = sortear_inicio_misto(env, curriculo, args.fracao_parede, args.fracao_inferior)
            state = env.reset(initial_pose=pose)
            total_reward, ep_steps, passos_re = 0.0, 0, 0
            while True:
                passos_novos = estado['global_step'] - estado['inicio_re']
                eps = max(0.05, args.eps_transferencia + (0.05 - args.eps_transferencia) * passos_novos / eps_decay_steps)
                action = agent.select_action(state, eps)
                next_state, reward, done, info = env.step(action)
                agent.buffer.push(state, action, reward, next_state, done)
                estado['global_step'] += 1
                ep_steps += 1
                passos_re += int(action >= 5)
                total_reward += float(reward)
                state = next_state
                if estado['global_step'] >= estado['treino_liberado_em'] and estado['global_step'] % train_every == 0:
                    loss = agent.train_step(batch_size)
                    if loss is not None:
                        estado['losses'].append(loss)
                    agent.update_target_network(tau)
                if not args.sem_render and estado['episode'] % render_every == 0:
                    env.render()
                if done:
                    agent.buffer.end_episode()
                    break
            sucesso = env.reached_goal()
            motivos['goal' if sucesso else ('colisao' if info['collision'] else 'timeout')] += 1
            successes.append(float(sucesso))
            estado['rewards'].append(total_reward)
            estado['success_rate'].append(float(np.mean(successes)))
            estado['curr_history'].append(float(curriculo.d))
            # Apenas os episódios sorteados pelo currículo decidem seu avanço.
            if tipo.startswith('curriculo'):
                curriculo.registrar(sucesso)
            if estado['episode'] % 20 == 0:
                fps = (estado['global_step'] - inicio_sessao) / max(time.time() - t0, 1e-6)
                print(f"ep {estado['episode']} | steps {estado['global_step']} | R {total_reward:.1f} | "
                      f"suc50 {np.mean(successes):.2f} | eps {eps:.3f} | d_curr {curriculo.d:.1f} | "
                      f"início {tipo} | ré {passos_re}/{ep_steps} | buf {len(agent.buffer)} | {fps:.0f} steps/s", flush=True)
                motivos.clear()
            if estado['global_step'] % eval_every_steps < ep_steps:
                validar()
                gravar()
                save_training_plots(estado['rewards'], estado['success_rate'], estado['losses'], output_dir, estado['curr_history'])
        if not estado['avaliacoes'] or estado['avaliacoes'][-1]['passos'] != estado['global_step']:
            validar()
        agent.save(str(output_dir / 'dqn_model.pt'))
        gravar()
        save_training_plots(estado['rewards'], estado['success_rate'], estado['losses'], output_dir, estado['curr_history'])
    finally:
        env.close()


if __name__ == '__main__':
    main()
