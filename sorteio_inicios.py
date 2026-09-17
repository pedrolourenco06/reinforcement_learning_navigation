"""Inícios variados para labirinto6.png, alvo (13.2, 12.2).

Requer mapa sem np.flipud, res=0.4, wheelbase=0.4, corpo 0.545 x 0.415 m,
velocidade 0.5 m/s, dt=0.2 s e cinco esterçamentos de -20 a +20 graus.
O controle de referência foi usado apenas na validação externa das regiões;
nenhuma ação ou trajetória de referência é fornecida à DQN por este módulo.

Integração:
    from sorteio_inicios import validar_configuracao, sortear_inicio, avaliar_regioes
    validar_configuracao(env)  # uma vez, antes do treinamento
    rng_inicio = np.random.default_rng(SEED)
    # Dentro do for de episódios, antes do while:
    initial_pose, regiao = sortear_inicio(env, rng_inicio)
    state = env.reset(initial_pose=initial_pose)
    # Validação no final de um episódio, a cada 100 episódios:
    sucessos_validacao, media_passos = avaliar_regioes(env, agent)
"""

import hashlib
from itertools import product

import numpy as np


ALVO = np.array([13.2, 12.2], dtype=np.float32)

# Intervalos de x e y em metros; orientação em graus.
REGIOES = (
    {"nome": "facil", "peso": 0.50,
     "x": (10.75, 11.05), "y": (12.84, 13.14), "angulo": (-105.0, -75.0)},
    {"nome": "intermediario", "peso": 0.30,
     "x": (9.75, 10.05), "y": (18.10, 18.40), "angulo": (-12.67, 17.33)},
    {"nome": "dificil", "peso": 0.20,
     "x": (3.45, 3.75), "y": (20.25, 20.55), "angulo": (-105.0, -75.0)},
)


def validar_configuracao(env):
    """Verifica uma vez as condições usadas na validação das regiões."""
    parametros = {
        "wheelbase": 0.4, "robot_length": 0.545, "robot_width": 0.415,
        "speed": 0.5, "dt": 0.2, "max_steering_deg": 20.0, "res": 0.4,
    }
    for nome, esperado in parametros.items():
        if not np.isclose(getattr(env, nome), esperado):
            raise ValueError(f"Regiões validadas com {nome}={esperado}.")
    if not np.allclose(env.xlim, [0, 19.2]) or not np.allclose(env.ylim, [0, 24]):
        raise ValueError("Use xlim=[0,19.2] e ylim=[0,24].")
    if not np.allclose(env.alvo, ALVO):
        raise ValueError("Este sorteio foi preparado para alvo=[13.2,12.2].")
    if not env.reset_known_map_each_episode or not env.continuous_obs:
        raise ValueError("Ative continuous_obs e reset_known_map_each_episode.")
    if env.action_space.n != 5 or not np.allclose(
        env.steering_actions_deg, [-20, -10, 0, 10, 20]
    ):
        raise ValueError("Esperados os cinco esterçamentos de -20 a +20 graus.")
    mapa = np.asarray(env.mapa, dtype=np.uint8)
    digest = hashlib.sha256(mapa.tobytes()).hexdigest()
    if mapa.shape != (2400, 1920) or digest != (
    "784c2c8b2e7c1d5b1001a6cd7a311acc976b74dc5fb2cb6fc11536a72ae4a432"
    ):
        raise ValueError(
            "Mapa diferente do validado: esperado labirinto6.png "
            "com 1920 × 2400 pixels, sem np.flipud."
        )


def sortear_inicio(env, rng):
    """Retorna (pose, nome_regiao). Use um gerador persistente entre episódios.

    Sorteia primeiro a região, depois a pose. Em caso de colisão repete na
    mesma região, sem deslocar silenciosamente a distribuição para as fáceis.
    As proporções são probabilidades, não contagens exatas por bloco.
    """
    if not np.allclose(env.alvo, ALVO):
        raise ValueError("Mantenha o alvo fixo em [13.2,12.2].")
    indice = int(rng.choice(len(REGIOES), p=[r["peso"] for r in REGIOES]))
    regiao = REGIOES[indice]
    for _ in range(100):
        pose = np.array([
            rng.uniform(*regiao["x"]),
            rng.uniform(*regiao["y"]),
            np.deg2rad(rng.uniform(*regiao["angulo"])),
        ], dtype=np.float32)
        if (not env.collision(pose)
                and np.linalg.norm(pose[:2] - env.alvo) >= 1.0):
            return pose, regiao["nome"]
    raise RuntimeError(f"Não foi possível obter início válido em {regiao['nome']}.")


def poses_validacao():
    """81 poses fixas: centro e extremos dos três intervalos de cada região."""
    for regiao in REGIOES:
        for x, y, angulo in product(
            np.linspace(*regiao["x"], 3),
            np.linspace(*regiao["y"], 3),
            np.linspace(*regiao["angulo"], 3),
        ):
            yield regiao["nome"], np.array(
                [x, y, np.deg2rad(angulo)], dtype=np.float32
            )


def avaliar_regioes(env, agent, max_steps=500):
    """Avalia sem exploração/treinamento. Chame somente após encerrar episódio.

    Retorna (sucessos_totais, media_passos_nos_sucessos), mantendo o contrato
    da função anterior. Imprime também resultados separados por região.
    Não usa nem avança os geradores aleatórios do treinamento.
    """
    import torch

    if not np.allclose(env.alvo, ALVO):
        raise ValueError("A avaliação requer alvo=[13.2,12.2].")
    if not env.reset_known_map_each_episode:
        raise ValueError("Reinicie o mapa conhecido a cada episódio.")
    resultados = {r["nome"]: [] for r in REGIOES}
    passos_sucessos = []
    modo_anterior = agent.q_net.training
    agent.q_net.eval()
    try:
        with torch.no_grad():
            for nome, pose in poses_validacao():
                state = env.reset(initial_pose=pose)
                for passo in range(1, max_steps + 1):
                    entrada = torch.as_tensor(
                        state, dtype=torch.float32, device=agent.device
                    ).unsqueeze(0)
                    action = int(agent.q_net(entrada).argmax(dim=1).item())
                    state, _, done, _ = env.step(action)
                    if done:
                        break
                sucesso = bool(env.reached_goal())
                resultados[nome].append(int(sucesso))
                if sucesso:
                    passos_sucessos.append(passo)
    finally:
        agent.q_net.train(modo_anterior)
    for nome, historico in resultados.items():
        print(f"Validação {nome}: {sum(historico)}/{len(historico)} "
              f"({100 * np.mean(historico):.2f}%)")
    total = sum(sum(h) for h in resultados.values())
    media = float(np.mean(passos_sucessos)) if passos_sucessos else float("inf")
    return total, media
