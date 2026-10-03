# -*- coding: utf-8 -*-
"""Avalia a última retomada no ambiente original, sem atualizar a rede."""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pygame
import torch
import torch.nn as nn

import ackermann_env as cm


class QNetwork(nn.Module):
    # Mesma arquitetura e nomes das camadas do treinamento adaptado.
    def __init__(self, input_dim, num_actions, hidden=256):
        super().__init__()
        self.feat = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(),
                                  nn.Linear(hidden, hidden), nn.ReLU())
        self.val = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, 1))
        self.adv = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Linear(128, num_actions))

    def forward(self, x):
        h = self.feat(x)
        v, a = self.val(h), self.adv(h)
        return v + a - a.mean(dim=1, keepdim=True)


def escolher_dispositivo(nome):
    if nome == 'cpu':
        return torch.device('cpu')
    if nome in ('auto', 'cuda') and torch.cuda.is_available():
        return torch.device('cuda')
    if nome == 'cuda':
        raise RuntimeError('CUDA indisponível.')
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        if nome == 'directml':
            raise RuntimeError('torch-directml não está instalado.')
        return torch.device('cpu')


def escolher_modelo(pasta):
    # O melhor modelo pode ser anterior ao último estado treinado.
    candidatos = [pasta / 'dqn_model.pt', pasta / 'dqn_checkpoint.pt']
    existentes = [p for p in candidatos if p.is_file()]
    if not existentes:
        raise FileNotFoundError(f'Nenhum modelo final ou checkpoint em {pasta}. Use --modelo CAMINHO.')
    return max(existentes, key=lambda p: p.stat().st_mtime_ns)


def sortear_pose(env, rng):
    # Mesma sequência de sorteio da avaliação do treinamento: seed 123.
    for _ in range(20000):
        pose = np.array([rng.uniform(*env.xlim), rng.uniform(*env.ylim),
                         rng.uniform(-np.pi, np.pi)], dtype=np.float32)
        if np.linalg.norm(pose[:2] - env.alvo) > 0.30 and not env.collision(pose):
            return pose
    raise RuntimeError('Não foi possível sortear um início válido.')


def escrever_csv(caminho, linhas):
    if not linhas:
        return
    with caminho.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0]))
        writer.writeheader()
        writer.writerows(linhas)


def esperar(segundos=0.0, aguardar_enter=False):
    # Processa eventos antes de env.render(), que também consome eventos pygame.
    fim = time.monotonic() + segundos
    pausado = False
    while True:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                return False
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_ESCAPE:
                    return False
                if evento.key == pygame.K_RETURN and aguardar_enter:
                    return True
                if evento.key == pygame.K_SPACE:
                    pausado = not pausado
        if not pausado and not aguardar_enter and time.monotonic() >= fim:
            return True
        pygame.time.wait(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', type=Path, default=None)
    parser.add_argument('--pasta-modelos', type=Path, default=Path('results_retomada'))
    parser.add_argument('--mapa', type=Path, default=Path('labirinto6.png'))
    parser.add_argument('--saida', type=Path, default=Path('results_retomada/avaliacao'))
    parser.add_argument('--episodios', type=int, default=30)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--pose', type=float, nargs=3, action='append', metavar=('X', 'Y', 'ANGULO_GRAUS'))
    parser.add_argument('--velocidade', type=float, default=1.0,
                        help='Velocidade visual: 1 = tempo simulado; 4 = quatro vezes mais rápido')
    parser.add_argument('--automatico', action='store_true', help='Avança sem pressionar Enter')
    parser.add_argument('--sem-render', action='store_true')
    parser.add_argument('--dispositivo', choices=['auto', 'cpu', 'cuda', 'directml'], default='auto')
    parser.add_argument('--confiar-checkpoint', action='store_true',
                        help='Permite carregar seu checkpoint DirectML com weights_only=False')
    args = parser.parse_args()
    if args.episodios <= 0 or not np.isfinite(args.velocidade) or args.velocidade <= 0:
        parser.error('Episódios e velocidade devem ser positivos.')

    modelo = args.modelo or escolher_modelo(args.pasta_modelos)
    device = escolher_dispositivo(args.dispositivo)
    print(f'Modelo: {modelo.resolve()}\nDispositivo: {device}', flush=True)
    try:
        dados = torch.load(modelo, map_location='cpu', weights_only=not args.confiar_checkpoint)
    except Exception as erro:
        if not args.confiar_checkpoint and 'Weights only load failed' in str(erro):
            raise RuntimeError('Checkpoint DirectML: para um arquivo do seu próprio treinamento, '
                               'execute novamente acrescentando --confiar-checkpoint.') from erro
        raise
    if dados.get('obs_dim') != 251 or dados.get('num_actions') != 5:
        raise ValueError('Este avaliador requer o modelo adaptado de 251 observações e 5 ações.')
    rede = QNetwork(251, 5).to(device)
    rede.load_state_dict(dados['q_net_state_dict'])
    rede.eval()
    # As poses registradas são preservadas quando foi escolhido um checkpoint completo.
    poses_salvas = dados.get('estado_treino', {}).get('poses')
    del dados  # Libera o replay e o otimizador, desnecessários para avaliação.

    env = cm.AckermannEnv(
        img=str(args.mapa), xlim=np.array([0.0, 19.2]), ylim=np.array([0.0, 24.0]),
        alvo=np.array([13.2, 12.2]), render=not args.sem_render,
        continuous_obs=True, window_layers=5, reset_known_map_each_episode=True,
        wheelbase=0.4, robot_length=0.545, robot_width=0.415,
        max_steering_deg=20.0, speed=0.5, dt=0.2,
    )
    args.saida.mkdir(parents=True, exist_ok=True)
    resumos = []
    interrompido = False
    try:
        if args.pose:
            poses = [np.array([x, y, np.deg2rad(angulo)], dtype=np.float32)
                     for x, y, angulo in args.pose]
        elif poses_salvas is not None and args.seed == 123:
            poses = [np.asarray(p, dtype=np.float32) for p in poses_salvas[:args.episodios]]
            if args.episodios > len(poses_salvas):
                raise ValueError('Checkpoint tem menos poses que --episodios; use --seed diferente para outro conjunto.')
        else:
            rng = np.random.RandomState(args.seed)
            poses = [sortear_pose(env, rng) for _ in range(args.episodios)]

        print('Avaliação gulosa: epsilon=0; sem treinamento.', flush=True)
        if not args.sem_render:
            print('Espaço: pausar | Enter: próximo episódio ao finalizar | Esc: encerrar', flush=True)
        for ep, pose in enumerate(poses, 1):
            state = env.reset(initial_pose=pose)
            total_reward = 0.0
            minimo = float(np.linalg.norm(env.p - env.alvo))
            linhas = [{'passo': 0, 'x': float(env.pose[0]), 'y': float(env.pose[1]),
                       'angulo_graus': float(np.rad2deg(env.pose[2])), 'acao': '', 'recompensa': 0.0}]
            if not args.sem_render:
                if not esperar():
                    interrompido = True
                    break
                env.render()
            motivo = 'timeout'
            for passo in range(1, cm.MAX_STEPS + 1):
                if not args.sem_render and not esperar(env.dt / args.velocidade):
                    interrompido = True
                    motivo = 'interrompido'
                    break
                with torch.no_grad():
                    entrada = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                    acao = int(rede(entrada).argmax(dim=1).item())
                state, recompensa, done, info = env.step(acao)
                total_reward += float(recompensa)
                minimo = min(minimo, float(np.linalg.norm(env.p - env.alvo)))
                linhas.append({'passo': passo, 'x': float(env.pose[0]), 'y': float(env.pose[1]),
                               'angulo_graus': float(np.rad2deg(env.pose[2])),
                               'acao': acao, 'recompensa': float(recompensa)})
                if not args.sem_render:
                    pygame.display.set_caption(f'Avaliação {ep}/{len(poses)} | passo {passo}')
                    env.render()
                    if not pygame.display.get_init():
                        interrompido = True
                        motivo = 'interrompido'
                        break
                if done:
                    motivo = 'alvo' if env.reached_goal() else ('colisao' if info['collision'] else 'timeout')
                    break
            passos = len(linhas) - 1
            resumo = {'episodio': ep, 'inicio_x': float(pose[0]), 'inicio_y': float(pose[1]),
                      'inicio_angulo_graus': float(np.rad2deg(pose[2])), 'motivo': motivo,
                      'passos': passos, 'recompensa': total_reward, 'menor_distancia': minimo}
            resumos.append(resumo)
            escrever_csv(args.saida / f'trajeto_{ep:03d}.csv', linhas)
            if not args.sem_render and pygame.display.get_init():
                pygame.image.save(env.screen, str(args.saida / f'trajeto_{ep:03d}.png'))
            escrever_csv(args.saida / 'resumo.csv', resumos)
            print(f'Episódio {ep}/{len(poses)} | {motivo} | passos {passos} | '
                  f'R {total_reward:.1f} | menor distância {minimo:.2f} m', flush=True)
            if interrompido:
                break
            if not args.sem_render:
                if not args.automatico:
                    print('Trajeto salvo. Pressione Enter para continuar ou Esc para sair.', flush=True)
                if not esperar(1.0 if args.automatico else 0.0, aguardar_enter=not args.automatico):
                    break
    finally:
        env.close()
    concluidos = [r for r in resumos if r['motivo'] != 'interrompido']
    sucessos = [r for r in concluidos if r['motivo'] == 'alvo']
    colisoes = sum(r['motivo'] == 'colisao' for r in concluidos)
    media = np.mean([r['passos'] for r in sucessos]) if sucessos else float('nan')
    print(f'FINAL: {len(sucessos)}/{len(concluidos)} sucessos '
          f'({100 * len(sucessos) / max(len(concluidos), 1):.1f}%) | '
          f'colisões {colisoes} | passos médios dos sucessos {media:.1f}')
    print(f'Resultados: {args.saida.resolve()}')


if __name__ == '__main__':
    main()
