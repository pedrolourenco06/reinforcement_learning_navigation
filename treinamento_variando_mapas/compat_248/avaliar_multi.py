"""Avalia sem treinamento; exporta trajetórias PNG e CSV."""
import argparse
import csv
import numpy as np
from pathlib import Path
import torch
import pygame
import time
from ambiente_multi import Ambiente, CONFIG, OBS_DIM, ACOES
from treinar_multi import Rede, device_for, action

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--modelo',default='results_multi/melhor.pt')
    ap.add_argument('--mapas',default='mapas/manifesto.json')
    ap.add_argument('--split',choices=['treino','validacao','teste'],default='teste')
    ap.add_argument('--episodios',type=int,default=100)
    ap.add_argument('--seed',type=int,default=9876)
    ap.add_argument('--saida',default='avaliacao_multi')
    ap.add_argument('--dispositivo',choices=['auto','cpu','cuda','directml'],default='auto')
    ap.add_argument('--sem-render',action='store_true')
    ap.add_argument('--automatico',action='store_true')
    ap.add_argument('--velocidade',type=float,default=1.)
    args=ap.parse_args()
    if not np.isfinite(args.velocidade) or args.velocidade<=0: ap.error('Velocidade deve ser positiva')
    if args.episodios<1: ap.error('episodios deve ser positivo')
    device=device_for(args.dispositivo)
    data=torch.load(args.modelo,map_location='cpu',weights_only=True)
    if data['config']['environment']!=CONFIG or data['config']['obs_dim']!=OBS_DIM or data['config']['actions']!=ACOES.tolist():
        raise ValueError('Modelo incompatível com este ambiente')
    net=Rede().to(device); net.load_state_dict(data['weights']); net.eval()
    env=Ambiente(args.mapas,args.split,args.seed); out=Path(args.saida); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    screen=None
    def esperar(segundos=0.,enter=False):
        end=time.monotonic()+segundos;paused=False
        while True:
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE): raise KeyboardInterrupt
                if event.type==pygame.KEYDOWN:
                    if event.key==pygame.K_SPACE: paused=not paused
                    if event.key==pygame.K_RETURN and enter:return
            if not paused and not enter and time.monotonic()>=end:return
            pygame.time.wait(10)
    def desenhar():
        nonlocal screen
        frame=env.image()
        if screen is None:
            pygame.init();screen=pygame.display.set_mode(frame.size)
        surface=pygame.image.fromstring(frame.tobytes(),frame.size,'RGB')
        screen.blit(surface,(0,0));pygame.display.flip()
    if not args.sem_render: pygame.init()
    for ep in range(args.episodios):
        s=env.reset(stage=2); start=env.pose.copy(); minimum=float('inf'); total=0; trajectory=[]
        if not args.sem_render: desenhar()
        while True:
            if not args.sem_render: esperar(CONFIG['dt']/args.velocidade)
            a=action(net,s,device); s,r,term,trunc,info=env.step(a)
            minimum=min(minimum,info['distance']); total+=r
            trajectory.append(dict(step=env.steps,x=env.pose[0],y=env.pose[1],theta=env.pose[2],action=a,distance=info['distance'],reward=r))
            if not args.sem_render:
                pygame.display.set_caption(f'Avaliação 248 | {ep+1}/{args.episodios} | passo {env.steps}')
                desenhar()
            if term or trunc: break
        row=dict(episode=ep+1,map=Path(env.map.path).name,start_x=start[0],start_y=start[1],start_theta=start[2],
                 goal_x=env.goal[0],goal_y=env.goal[1],success=int(info['success']),collision=int(info['collision']),
                 steps=env.steps,min_distance=minimum,reward=total)
        rows.append(row); env.image().save(out/f'trajeto_{ep+1:03d}.png')
        with (out/f'trajeto_{ep+1:03d}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=trajectory[0].keys()); w.writeheader(); w.writerows(trajectory)
        print(f'{ep+1}/{args.episodios}: sucesso={info["success"]}, distância mínima={minimum:.2f}',flush=True)
        if not args.sem_render:
            if not args.automatico: print('Enter: próximo | Espaço: pausa | Esc: sair',flush=True)
            esperar(.5 if args.automatico else 0.,enter=not args.automatico)
    pygame.quit()
    with (out/'resumo.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f'Sucesso: {sum(r["success"] for r in rows)/len(rows):.1%}')
if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print('Avaliação interrompida.')
    finally: pygame.quit()
