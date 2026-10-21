"""Avaliação original adaptada: renderização Pygame ativa por padrão, 251 entradas."""
import argparse
from pathlib import Path
import numpy as np
import pygame
import torch
from labirinto_dqn_re import QNetwork
from tarefas_multi import GestorMapas,escolher_dispositivo,esperar,escrever_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--modelo',type=Path,default=Path('results_re_multi/dqn_best.pt'))
    p.add_argument('--mapas',type=Path,default=Path('mapas/manifesto.json'))
    p.add_argument('--split',choices=['treino','validacao','teste'],default='teste')
    p.add_argument('--mapa',type=Path,help='Opcional: avaliar apenas este PNG')
    p.add_argument('--px-por-metro',type=float,default=100.)
    p.add_argument('--episodios',type=int,default=30)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--saida',type=Path,default=Path('results_re_multi/avaliacao'))
    p.add_argument('--dispositivo',choices=['auto','cpu','cuda','directml'],default='auto')
    p.add_argument('--velocidade',type=float,default=1.)
    p.add_argument('--automatico',action='store_true',help='Avança sem Enter')
    p.add_argument('--sem-render',action='store_true')
    a=p.parse_args()
    if a.episodios<1 or not np.isfinite(a.velocidade) or a.velocidade<=0: p.error('Parâmetros inválidos')
    device=escolher_dispositivo(a.dispositivo)
    data=torch.load(a.modelo,map_location='cpu',weights_only=True)
    if data.get('obs_dim')!=251 or data.get('num_actions')!=8 or data.get('configuracao',{}).get('versao')!='original_multi_v1':
        raise ValueError('Use um modelo desta revisão (251 entradas). Para o pacote anterior de 248 entradas, use compat_248/avaliar_multi.py.')
    net=QNetwork(251,8).to(device);net.load_state_dict(data['q_net_state_dict']);net.eval();del data
    g=GestorMapas(a.mapas,a.split,mapa_unico=a.mapa,ppm=a.px_por_metro);env=g.env
    np.random.seed(a.seed);a.saida.mkdir(parents=True,exist_ok=True)
    rows=[];interrompido=False
    try:
        env.visualizar(not a.sem_render)
        for ep in range(1,a.episodios+1):
            tarefa=g.sorteio();s=g.reset(tarefa);total=0.;mini=float(np.linalg.norm(env.p-env.alvo))
            traj=[];motivo='timeout'
            if not a.sem_render: env.render()
            while True:
                if not a.sem_render and not esperar(env.dt/a.velocidade):
                    interrompido=True;motivo='interrompido';break
                with torch.no_grad():
                    action=int(net(torch.as_tensor(s,device=device).unsqueeze(0)).argmax(1).item())
                s,r,done,info=env.step(action);total+=r
                distance=float(np.linalg.norm(env.p-env.alvo));mini=min(mini,distance)
                traj.append(dict(passo=env.steps,x=float(env.p[0]),y=float(env.p[1]),theta=float(env.pose[2]),
                    acao=action,direcao='re' if action>=5 else 'frente',distancia=distance,recompensa=r))
                if not a.sem_render:
                    pygame.display.set_caption(f'Avaliação {ep}/{a.episodios} | passo {env.steps} | {Path(tarefa["arquivo"]).name}')
                    env.render()
                    if not pygame.display.get_init():
                        interrompido=True;motivo='interrompido';break
                if done:
                    motivo='alvo' if env.reached_goal() else ('colisao' if info['collision'] else 'timeout');break
            rows.append(dict(episodio=ep,mapa=tarefa['arquivo'],inicio_x=tarefa['pose'][0],inicio_y=tarefa['pose'][1],
                inicio_theta=tarefa['pose'][2],alvo_x=tarefa['alvo'][0],alvo_y=tarefa['alvo'][1],motivo=motivo,
                passos=env.steps,recompensa=total,menor_distancia=mini))
            escrever_csv(a.saida/f'trajeto_{ep:03d}.csv',traj);escrever_csv(a.saida/'resumo.csv',rows)
            if not a.sem_render and pygame.display.get_init():
                pygame.image.save(env.screen,str(a.saida/f'trajeto_{ep:03d}.png'))
            print(f'{ep}/{a.episodios}: {motivo}; passos={env.steps}; distância mínima={mini:.2f} m',flush=True)
            if interrompido:break
            if not a.sem_render:
                if not a.automatico: print('Enter: próximo | Espaço: pausa | Esc: sair',flush=True)
                if not esperar(.5 if a.automatico else 0.,aguardar_enter=not a.automatico): break
    except KeyboardInterrupt: print('Avaliação interrompida.')
    finally: env.close();pygame.quit()
    completed=[r for r in rows if r['motivo']!='interrompido']
    print(f'Sucesso: {sum(r["motivo"]=="alvo" for r in completed)}/{len(completed)}')
if __name__=='__main__':main()
