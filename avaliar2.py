"""Avalia checkpoints DQN sem executar treinamento.

Na pasta deste arquivo devem estar ackermann_env2.py e labirinto6.png.
Uso: python avaliar_ackermann_env2.py
     python avaliar_ackermann_env2.py --modelo results/dqn_model.pt --sem-render
     python avaliar_ackermann_env2.py --pose 3.6 20.4 -90

Reconhece MLP (net) e Dueling DQN (feat/val/adv), com dimensoes dos pesos.
Pressupoe ReLU e, na Dueling, Q = V + A - media(A). Essas operacoes nao
sao registradas no state_dict: devem coincidir com a classe do treinamento.
CONFIG_ENV deve reproduzir os parametros do treinamento (dimensoes iguais
nao garantem observacoes semanticamente iguais). Nao carregue checkpoints
antigos do ambiente de 130 entradas/5 acoes neste ambiente novo.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parent

# Reproduza aqui a configuracao usada no treinamento do checkpoint.
CONFIG_ENV = dict(
    xlim=[0.0, 19.2], ylim=[0.0, 24.0], alvo=[13.2, 12.2],
    res=0.4, continuous_obs=True,
    window_layers=3, window_res=0.6, sensor_layers=6,
    reset_known_map_each_episode=True,
    wheelbase=0.4, robot_length=0.545, robot_width=0.415,
    max_steering_deg=20.0, speed=0.5, dt=0.2,
    action_repeat=4, n_rays=16, ray_max_range=3.0, ray_step=0.10,
    gamma=0.99, goal_tol=0.70, geodesic_res=0.05,
    render_playback_speed=1.0,
)


def bloco_linear(pesos, prefixo, indices, ativar_ultima=False):
    """Reconstroi blocos Linear/ReLU com indices explicitos do checkpoint."""
    camadas = []
    saida_anterior = None
    for posicao, indice in enumerate(indices):
        chave = f"{prefixo}.{indice}.weight"
        if chave not in pesos or pesos[chave].ndim != 2:
            raise ValueError(f"Camada ausente ou invalida: {chave}")
        saida, entrada = pesos[chave].shape
        if saida_anterior is not None and entrada != saida_anterior:
            raise ValueError(f"Dimensoes desconectadas em {chave}")
        camadas.append(nn.Linear(entrada, saida))
        if posicao < len(indices) - 1 or ativar_ultima:
            camadas.append(nn.ReLU())
        saida_anterior = saida
    return nn.Sequential(*camadas)


class QNetwork(nn.Module):
    def __init__(self, pesos):
        super().__init__()
        self.net = bloco_linear(pesos, 'net', [0, 2, 4])

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, pesos):
        super().__init__()
        self.feat = bloco_linear(pesos, 'feat', [0, 2], ativar_ultima=True)
        self.val = bloco_linear(pesos, 'val', [0, 2])
        self.adv = bloco_linear(pesos, 'adv', [0, 2])
        tamanho = pesos['feat.2.weight'].shape[0]
        if (pesos['val.0.weight'].shape[1] != tamanho
                or pesos['adv.0.weight'].shape[1] != tamanho
                or pesos['val.2.weight'].shape[0] != 1):
            raise ValueError('Dimensoes incompativeis nos ramos Dueling.')

    def forward(self, x):
        features = self.feat(x)
        valor = self.val(features)
        vantagem = self.adv(features)
        return valor + vantagem - vantagem.mean(dim=1, keepdim=True)


def carregar_rede(checkpoint, device):
    pesos = checkpoint['q_net_state_dict']
    if 'feat.0.weight' in pesos:
        rede = DuelingQNetwork(pesos)
        entrada = pesos['feat.0.weight'].shape[1]
        saida = pesos['adv.2.weight'].shape[0]
        print('Rede: Dueling DQN; pressupostos: ReLU e Q = V + A - media(A).')
    elif 'net.0.weight' in pesos:
        rede = QNetwork(pesos)
        entrada = pesos['net.0.weight'].shape[1]
        saida = pesos['net.4.weight'].shape[0]
        print('Rede: MLP com ReLU.')
    else:
        raise ValueError('Arquitetura desconhecida; use a classe do treinamento.')
    if entrada != int(checkpoint['obs_dim']) or saida != int(checkpoint['num_actions']):
        raise ValueError('Metadados obs_dim/num_actions diferem dos pesos.')
    # Nunca ignorar chaves: todos os parametros precisam ser carregados.
    rede.load_state_dict(pesos, strict=True)
    return rede.to(device)


def avaliar_modelo(env, rede, poses, render=False):
    """Poses em (x, y, graus). Nao altera pesos; reinicia o mapa por episodio."""
    poses = list(poses)
    if not poses:
        raise ValueError('Informe pelo menos uma pose.')
    if not env.continuous_obs or not env.reset_known_map_each_episode:
        raise ValueError('Use observacao continua e reset_known_map_each_episode=True.')
    if render and not env.render_env:
        raise ValueError('Crie o ambiente com render=True.')
    device = next(rede.parameters()).device
    modo = rede.training
    resultados = []
    rede.eval()
    try:
        with torch.no_grad():
            for numero, (x, y, angulo) in enumerate(poses, 1):
                state = env.reset(initial_pose=[x, y, np.deg2rad(angulo)])
                retorno = 0.0
                if render:
                    env.render()
                while True:
                    state = np.asarray(state, dtype=np.float32)
                    if state.shape != env.observation_space.shape or not np.isfinite(state).all():
                        raise ValueError('Observacao invalida: formato ou valores nao finitos.')
                    entrada = torch.as_tensor(state, device=device).unsqueeze(0)
                    q = rede(entrada)
                    if q.shape != (1, env.action_space.n):
                        raise ValueError('Numero de saidas da rede diferente das acoes do ambiente.')
                    if not torch.isfinite(q).all().item():
                        raise ValueError('A rede produziu NaN ou infinito.')
                    action = int(q.argmax(dim=1).item())
                    state, reward, terminated, truncated, info = env.step(action)
                    retorno += float(reward)
                    if render:
                        env.render()
                        if env.screen is None:
                            raise KeyboardInterrupt
                    if terminated or truncated:
                        break

                colisao = bool(info['collision'])
                sucesso = bool(info['success']) and not colisao
                motivo = ('sucesso' if sucesso else 'colisao' if colisao
                          else 'limite_de_passos' if truncated else 'outro_termino')
                traj = np.asarray(env.traj, dtype=float)
                dist = np.linalg.norm(traj - env.alvo, axis=1)
                percurso = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
                item = dict(
                    teste=numero, inicio=[float(x), float(y), float(angulo)],
                    sucesso=sucesso, colisao=colisao, motivo=motivo,
                    terminated=bool(terminated), truncated=bool(truncated),
                    decisoes=int(env.steps), subpassos=int(env.sub_steps),
                    tempo_simulado_s=float(env.sub_steps * env.dt),
                    reward=retorno, distancia_inicial_m=float(dist[0]),
                    menor_distancia_m=float(dist.min()), distancia_final_m=float(dist[-1]),
                    trajeto_m=percurso, posicao_final=env.p.tolist(),
                    explorado_pct=float(env.get_percentage_explored(only_free=True)),
                )
                resultados.append(item)
                print(
                    f'Teste {numero:02d}/{len(poses)} | Inicio: ({x}, {y}, {angulo} graus) | '
                    f'Fim: {motivo} | Decisoes: {env.steps} | Subpassos: {env.sub_steps} | '
                    f'Reward: {retorno:.2f} | Menor distancia: {dist.min():.3f} m | '
                    f'Distancia final: {dist[-1]:.3f} m | '
                    f'Posicao final: ({env.p[0]:.2f}, {env.p[1]:.2f})'
                )
    finally:
        rede.train(modo)
    bons = [r for r in resultados if r['sucesso']]
    resumo = dict(
        testes=len(resultados), sucessos=len(bons),
        taxa_sucesso_pct=100.0 * len(bons) / len(resultados),
        colisoes=sum(r['colisao'] for r in resultados),
        truncamentos=sum(r['truncated'] for r in resultados),
        media_decisoes_sucessos=float(np.mean([r['decisoes'] for r in bons])) if bons else None,
        resultados=resultados,
    )
    print(f"\nSucesso: {len(bons)}/{len(resultados)} ({resumo['taxa_sucesso_pct']:.2f}%) | "
          f"Colisoes: {resumo['colisoes']} | Limites de passos: {resumo['truncamentos']}")
    return resumo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modelo', type=Path, default=BASE / 'results' / 'dqn_best.pt')
    parser.add_argument('--mapa', type=Path, default=BASE / 'labirinto6.png')
    parser.add_argument('--pose', nargs=3, type=float, action='append', metavar=('X', 'Y', 'GRAUS'),
                        help='Repetivel; padrao: 10.9 13.0 -90 (caso facil).')
    parser.add_argument('--sem-render', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--saida', type=Path, help='Opcional: salva o relatorio em JSON.')
    args = parser.parse_args()
    if not args.modelo.is_file():
        parser.error(f'Checkpoint nao encontrado: {args.modelo}')
    if not args.mapa.is_file():
        parser.error(f'Bitmap nao encontrado: {args.mapa}')

    device = torch.device('cpu')
    if not args.cpu:
        try:
            import torch_directml
            device = torch_directml.device()
        except ImportError:
            pass

    # weights_only=False e necessario para alguns checkpoints DirectML.
    # Carregue somente arquivos do seu proprio treinamento ou de fonte confiavel.
    checkpoint = torch.load(args.modelo, map_location='cpu', weights_only=False)
    necessarias = {'obs_dim', 'num_actions', 'q_net_state_dict'}
    if not isinstance(checkpoint, dict) or not necessarias.issubset(checkpoint):
        raise ValueError('Checkpoint precisa de obs_dim, num_actions e q_net_state_dict.')
    obs_dim, n_actions = int(checkpoint['obs_dim']), int(checkpoint['num_actions'])
    if n_actions not in (3, 6):
        raise ValueError(f'ackermann_env2 usa 3 acoes (frente) ou 6 (com re); modelo tem {n_actions}.')
    esperado = CONFIG_ENV['n_rays'] + (2 * CONFIG_ENV['window_layers'] + 1) ** 2 + 13
    if obs_dim != esperado:
        raise ValueError(f'Modelo tem {obs_dim} entradas; CONFIG_ENV define {esperado}. '
                         'Use os mesmos parametros do treinamento.')
    rede = carregar_rede(checkpoint, device)

    # Importa apenas o ambiente: nenhum modulo de treinamento e executado.
    import ackermann_env2 as cm
    config = dict(CONFIG_ENV)
    config.update(img=str(args.mapa.resolve()), render=not args.sem_render,
                  allow_reverse=(n_actions == 6), cache_dir=str(BASE / 'cache'))
    print(f'Modelo: {args.modelo.resolve()}\nDispositivo: {device}\n'
          f'Observacao: {obs_dim} | Acoes: {n_actions} | Re: {n_actions == 6}')
    print('CONFIG_ENV deve coincidir com o treinamento, inclusive tolerancia e action_repeat.')
    env = cm.AckermannEnv(**config)
    try:
        resumo = avaliar_modelo(env, rede, args.pose or [(10.9, 13.0, -90.0)],
                                render=not args.sem_render)
        if args.saida:
            args.saida.parent.mkdir(parents=True, exist_ok=True)
            relatorio = dict(modelo=str(args.modelo.resolve()), configuracao=config, **resumo)
            args.saida.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f'Relatorio salvo: {args.saida}')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nAvaliacao interrompida pelo usuario.')
