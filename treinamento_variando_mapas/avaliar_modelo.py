# Avalia um ou mais modelos DQN treinados em mapas escolhidos, com cenários
# (mapa, alvo, pose inicial) sorteados com semente fixa e política gulosa (sem epsilon).
#
# Resultado por mapa: sucesso, colisão e timeout, com intervalo de confiança de 95% (Wilson).
#
# Os cenários dependem apenas da semente, do nome do mapa e dos parâmetros de sorteio.
# Assim, o mesmo comando dá os mesmos cenários para modelos diferentes (comparação justa)
# e acrescentar um mapa à lista não muda os cenários dos outros mapas.
#
# Exemplos:
#   python avaliar_modelo.py --modelo results_multi/dqn_best.pt --mapas mapas_treino --cenarios-por-mapa 20
#   python avaliar_modelo.py --modelo a/dqn_best.pt b/dqn_best.pt --mapas labirinto6.png --saida aval_labirinto6
#
# Por padrão TODOS os episódios são renderizados (janela do pygame); --sem-render desliga.
# Dica: renderizar é muito mais lento. Para estatística use --sem-render e muitos cenários;
# para ver o comportamento use poucos cenários (ex.: --cenarios-por-mapa 3) e, se quiser, --fps.
#
# Deve ficar na mesma pasta de labirinto_dqn_multi.py e ackermann_env_multi.py.
# A área do ambiente (xlim/ylim) vem de CONFIG_ENV em labirinto_dqn_multi.py.

import argparse
import csv
import math
import time
import zlib
from pathlib import Path

import numpy as np
import pygame
import torch  # noqa: F401  (usado pelo agente)

import labirinto_dqn_multi as L

cm = L.cm


class JanelaFechada(Exception):
    pass


def wilson(k, n, z=1.96):
    # Intervalo de confiança de Wilson para uma proporção k/n.
    if n == 0:
        return float('nan'), float('nan')
    p = k / n
    d = 1.0 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centro - meia), min(1.0, centro + meia)


def gerar_cenarios(env, nome, n, semente, dist_max=None):
    # Cenários de um mapa, independentes dos demais mapas e do estado global do numpy.
    estado = np.random.get_state()
    cenarios = []
    try:
        np.random.seed((int(semente) + zlib.crc32(nome.encode('utf-8'))) % (2 ** 32))
        env.set_map(nome)
        for _ in range(n):
            alvo = L.sortear_alvo(env)
            env.set_alvo(alvo)
            pose = L.sortear_pose(env, dist_max=dist_max)
            cenarios.append(dict(img=nome,
                                 alvo=[float(alvo[0]), float(alvo[1])],
                                 pose=[float(v) for v in pose]))
    finally:
        np.random.set_state(estado)
    return cenarios


def rodar_cenario(env, agent, c, render=False, fps=None, pausa=0.0, legenda=''):
    # Mesma lógica de L.avaliar: sucesso se chegou ao alvo; senão colisão se colidiu; senão timeout.
    # Com render=True desenha cada passo (o desenho não altera a simulação).
    state = env.reset(initial_pose=np.asarray(c['pose'], dtype=np.float32),
                      alvo=np.asarray(c['alvo'], dtype=np.float64),
                      img=c['img'])

    def desenhar(k):
        pygame.display.set_caption(f'{legenda} | passo {k}')
        env.render()   # o ambiente limita a 120 FPS
        if not pygame.display.get_init():   # janela fechada: o ambiente encerrou o pygame
            raise JanelaFechada()
        if fps:
            time.sleep(max(0.0, 1.0 / fps - 1.0 / 120))

    if render:
        desenhar(0)
    info = {}
    passos, passos_re = 0, 0
    for k in range(1, cm.MAX_STEPS + 1):
        action = agent.greedy_action(state)
        state, _, done, info = env.step(action)
        passos = k
        passos_re += int(action >= 5)
        if render:
            desenhar(k)
        if done:
            break
    if render and pausa > 0:
        time.sleep(pausa)   # mantém o quadro final na tela
    if env.reached_goal():
        resultado = 'sucesso'
    elif info.get('collision', False):
        resultado = 'colisao'
    else:
        resultado = 'timeout'
    dist = float(math.hypot(c['pose'][0] - c['alvo'][0], c['pose'][1] - c['alvo'][1]))
    return dict(img=c['img'], alvo_x=c['alvo'][0], alvo_y=c['alvo'][1],
                pose_x=c['pose'][0], pose_y=c['pose'][1], pose_theta=c['pose'][2],
                dist_inicial=dist, resultado=resultado, passos=passos, passos_re=passos_re)


def resumir(linhas):
    n = len(linhas)
    suc = [r for r in linhas if r['resultado'] == 'sucesso']
    col = sum(r['resultado'] == 'colisao' for r in linhas)
    tmo = sum(r['resultado'] == 'timeout' for r in linhas)
    lo, hi = wilson(len(suc), n)
    return dict(n=n, sucessos=len(suc), colisoes=col, timeouts=tmo,
                taxa=len(suc) / n if n else float('nan'), ic_inf=lo, ic_sup=hi,
                passos_medios=float(np.mean([r['passos'] for r in suc])) if suc else float('nan'),
                dist_media=float(np.mean([r['dist_inicial'] for r in linhas])) if n else float('nan'))


def imprimir_tabela(titulo, resumos, nomes):
    print(f'\n=== {titulo} ===')
    cab = f"{'mapa':<28}{'n':>4}{'suc':>5}{'col':>5}{'tmo':>5}{'taxa':>8}   {'IC95%':<13}{'passos(suc)':>12}{'dist.ini':>9}"
    print(cab)
    print('-' * len(cab))
    nomes = [n for n in nomes if resumos[n]['n'] > 0]
    for nome in nomes + ['TOTAL']:
        r = resumos[nome]
        if nome == 'TOTAL':
            print('-' * len(cab))
        ic = f"[{100 * r['ic_inf']:.0f}-{100 * r['ic_sup']:.0f}%]"
        ps = f"{r['passos_medios']:.0f}" if not math.isnan(r['passos_medios']) else '-'
        print(f"{nome:<28}{r['n']:>4}{r['sucessos']:>5}{r['colisoes']:>5}{r['timeouts']:>5}"
              f"{100 * r['taxa']:>7.1f}%   {ic:<13}{ps:>12}{r['dist_media']:>8.1f}m")
    macro = np.mean([resumos[n]['taxa'] for n in nomes])
    print(f'média simples por mapa: {100 * macro:.1f}%')


def verificar_escala(env):
    # Avisa se as imagens não casam com a área xlim x ylim do ambiente (pixels não quadrados
    # ou px/m diferente entre mapas), o que deforma a geometria e a escala do bônus de exploração.
    escalas = {k: (d['mx'], d['my']) for k, d in env.mapas.items()}
    ref = next(iter(escalas.values()))[0]
    for k, (mx, my) in sorted(escalas.items()):
        if abs(mx - my) / max(mx, my) > 0.01:
            print(f'AVISO: {k}: pixels não quadrados ({mx:.1f} x {my:.1f} px/m). '
                  f'A imagem não tem a mesma proporção da área xlim x ylim.')
        elif abs(mx - ref) / ref > 0.01:
            print(f'AVISO: {k}: {mx:.1f} px/m, diferente dos demais mapas ({ref:.1f} px/m).')


def carregar_agente(caminho, confiar):
    dados = L.ler_modelo(caminho, confiar)
    agent = L.DQNAgent(251, 8, buffer_size=1000)
    L.carregar_pesos(agent, dados)
    agent.q_net.eval()
    return agent


def main():
    parser = argparse.ArgumentParser(description='Avalia modelos DQN por mapa, com cenários fixos por semente.')
    parser.add_argument('--modelo', type=Path, nargs='+', required=True,
                        help='Um ou mais .pt (dqn_best.pt, dqn_model.pt ou dqn_checkpoint.pt)')
    parser.add_argument('--mapas', nargs='+', required=True,
                        help='Imagens, pastas (*.png) ou padrões, ex.: mapas/*.png')
    parser.add_argument('--cenarios-por-mapa', type=int, default=20)
    parser.add_argument('--semente', type=int, default=2024,
                        help='Semente dos cenários (a avaliação do treino usa 123; use outra para um conjunto novo)')
    parser.add_argument('--dist-max', type=float, default=None,
                        help='Distância euclidiana máxima pose-alvo (padrão: sem limite, como na avaliação do treino)')
    parser.add_argument('--saida', type=Path, default=None, help='Pasta para gravar os CSVs')
    parser.add_argument('--confiar-checkpoint', action='store_true')
    parser.add_argument('--sem-render', action='store_true', help='Não abre janela (padrão: renderiza todos os episódios)')
    parser.add_argument('--fps', type=float, default=None,
                        help='Limita os quadros por segundo para assistir mais devagar (padrão: o limite de 120 FPS do ambiente)')
    parser.add_argument('--pausa', type=float, default=0.5, help='Segundos de pausa no quadro final de cada episódio')
    args = parser.parse_args()
    if args.cenarios_por_mapa <= 0:
        parser.error('--cenarios-por-mapa deve ser positivo.')
    if args.dist_max is not None and args.dist_max <= 0.50:
        parser.error('--dist-max deve ser maior que 0.30 m.')
    if args.fps is not None and args.fps <= 0:
        parser.error('--fps deve ser positivo.')
    if args.pausa < 0:
        parser.error('--pausa não pode ser negativa.')
    for m in args.modelo:
        if not m.is_file():
            parser.error(f'Modelo não encontrado: {m}')
    try:
        mapas = L.expandir_mapas(args.mapas)
    except (FileNotFoundError, ValueError) as erro:
        parser.error(str(erro))
    nomes = [m.name for m in mapas]

    env = cm.AckermannEnv(**L.CONFIG_ENV, img=str(mapas[0]), imgs=[str(m) for m in mapas],
                          alvo=L.ALVO_INICIAL.copy(), render=not args.sem_render)
    env.seed(L.SEED)
    render = not args.sem_render
    verificar_escala(env)

    print(f'Mapas ({len(nomes)}): {", ".join(nomes)}')
    cenarios = []
    for nome in nomes:
        cenarios.extend(gerar_cenarios(env, nome, args.cenarios_por_mapa, args.semente, args.dist_max))
    print(f'{len(cenarios)} cenários ({args.cenarios_por_mapa} por mapa), semente {args.semente}, '
          f'distância máx. {"sem limite" if args.dist_max is None else f"{args.dist_max} m"}, '
          f'timeout em {cm.MAX_STEPS} passos.')

    if args.saida is not None:
        args.saida.mkdir(parents=True, exist_ok=True)
    todos = {}
    try:
        for caminho in args.modelo:
            agent = carregar_agente(caminho, args.confiar_checkpoint)
            print(f'\nAvaliando {caminho} ...', flush=True)
            linhas = []
            interrompido = False
            for i, c in enumerate(cenarios, 1):
                legenda = f'{caminho.name} | {c["img"]} | cenário {i}/{len(cenarios)}'
                try:
                    r = rodar_cenario(env, agent, c, render=render, fps=args.fps, pausa=args.pausa, legenda=legenda)
                except JanelaFechada:
                    print('Janela fechada: interrompendo. Resultados parciais (cenário em andamento descartado).')
                    interrompido = True
                    break
                linhas.append(r)
                if render:
                    print(f'  [{i}/{len(cenarios)}] {c["img"]}: {r["resultado"]} ({r["passos"]} passos)', flush=True)
                elif i % args.cenarios_por_mapa == 0:
                    print(f'  {c["img"]}: {sum(x["resultado"] == "sucesso" for x in linhas[-args.cenarios_por_mapa:])}'
                          f'/{args.cenarios_por_mapa}', flush=True)
            if not linhas:
                break
            resumos = {n: resumir([r for r in linhas if r['img'] == n]) for n in nomes}
            resumos['TOTAL'] = resumir(linhas)
            imprimir_tabela(str(caminho), resumos, nomes)
            todos[str(caminho)] = resumos
            if args.saida is not None:
                rotulo = '_'.join(caminho.with_suffix('').parts[-2:]).replace(':', '')
                with (args.saida / f'cenarios_{rotulo}.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    w = csv.DictWriter(f, fieldnames=list(linhas[0]))
                    w.writeheader()
                    w.writerows(linhas)
                with (args.saida / f'resumo_{rotulo}.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    campos = ['mapa'] + list(resumos['TOTAL'])
                    w = csv.DictWriter(f, fieldnames=campos)
                    w.writeheader()
                    for n in nomes + ['TOTAL']:
                        if resumos[n]['n'] > 0:
                            w.writerow(dict(mapa=n, **resumos[n]))
            if interrompido:
                break
    finally:
        env.close()

    if len(args.modelo) > 1 and not interrompido:
        print('\n=== Comparação: taxa de sucesso (%) por mapa, nos mesmos cenários ===')
        chaves = list(todos)
        print(f"{'mapa':<28}" + ''.join(f'{"modelo " + str(i + 1):>11}' for i in range(len(chaves))))
        for n in nomes + ['TOTAL']:
            print(f'{n:<28}' + ''.join(f'{100 * todos[k][n]["taxa"]:>10.1f}%' for k in chaves))
        for i, k in enumerate(chaves):
            print(f'  modelo {i + 1} = {k}')


if __name__ == '__main__':
    main()