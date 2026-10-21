"""Ambiente independente: cinemática do código original, sensor com oclusão.
Mapa completo só é usado pelo simulador para colisão, sensor e sorteio.
A política recebe apenas mapa local observado e alvo relativo.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

ACOES=np.array([[-20,1],[-10,1],[0,1],[10,1],[20,1],[-20,-1],[0,-1],[20,-1]],float)
CONFIG=dict(wheelbase=.4,length=.545,width=.415,speed=.5,dt=.2,
            sensor_range=4.,sensor_rays=72,cell=.2,window_spacing=.4,
            window_layers=5,goal_radius=.3,max_steps=800)
OBS_DIM=248  # 121 ocupação + 121 conhecido + 6 variáveis

def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi

class Mapa:
    def __init__(self,path,ppm):
        self.path=str(path); self.ppm=float(ppm)
        self.ocupado=np.asarray(Image.open(path).convert('L'))<128
        self.h,self.w=self.ocupado.shape
        self.width=self.w/self.ppm; self.height=self.h/self.ppm
        # Grade conservadora de 20 cm para amostragem e conhecimento.
        k=round(CONFIG['cell']*ppm)
        if abs(k-CONFIG['cell']*ppm)>1e-6 or self.h%k or self.w%k:
            raise ValueError('Imagem deve ser divisível pela célula de 0.2 m em pixels.')
        self.grid=self.ocupado.reshape(self.h//k,k,self.w//k,k).any(axis=(1,3))
        padded=np.pad(~self.grid,1,constant_values=False)
        clearance=distance_transform_edt(padded)[1:-1,1:-1]*CONFIG['cell']
        # Disco circunscrito + meia diagonal da célula: margem conservadora.
        radius=np.hypot(CONFIG['length'],CONFIG['width'])/2
        self.safe=clearance > radius+np.sqrt(2)*CONFIG['cell']/2+.05
        self.labels,_=label(self.safe,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        ids,counts=np.unique(self.labels[self.safe],return_counts=True)
        self.components=[np.argwhere(self.labels==i) for i,n in zip(ids,counts) if n>=40]
        if not self.components: raise ValueError(f'Sem região segura: {path}')

    def world(self,rc):
        r,c=rc
        return np.array([(c+.5)*CONFIG['cell'],self.height-(r+.5)*CONFIG['cell']])

class Ambiente:
    def __init__(self,manifesto='mapas/manifesto.json',split='treino',seed=42):
        p=Path(manifesto); self.rng=np.random.default_rng(seed)
        self.entries=[e for e in json.loads(p.read_text()) if e['split']==split]
        if not self.entries: raise ValueError(f'Sem mapas para {split}')
        self.maps=[Mapa(p.parent/e['arquivo'],e['px_por_metro']) for e in self.entries]
        self.map=None; self.last_direction=1
        # Malha inclui as bordas do retângulo.
        xs=np.linspace(-CONFIG['length']/2,CONFIG['length']/2,19)
        ys=np.linspace(-CONFIG['width']/2,CONFIG['width']/2,15)
        self.foot=np.array(np.meshgrid(xs,ys)).reshape(2,-1).T

    def collision(self,pose):
        x,y,t=pose; c,s=np.cos(t),np.sin(t)
        pts=self.foot@np.array([[c,s],[-s,c]])+np.array([x,y])
        px=np.floor(pts[:,0]*self.map.ppm).astype(int)
        py=np.floor((self.map.height-pts[:,1])*self.map.ppm).astype(int)
        if np.any((px<0)|(py<0)|(px>=self.map.w)|(py>=self.map.h)): return True
        return bool(self.map.ocupado[py,px].any())

    def motion(self,pose,action,dt=None):
        delta,direction=ACOES[action]; v=CONFIG['speed']*direction
        dt=CONFIG['dt'] if dt is None else dt
        beta=np.arctan(.5*np.tan(np.deg2rad(delta))); x,y,t=pose
        return np.array([x+v*np.cos(t+beta)*dt,y+v*np.sin(t+beta)*dt,
                         wrap(t+v/(CONFIG['wheelbase']/2)*np.sin(beta)*dt)])

    def reset(self,stage=0,task=None):
        # Mapas elegíveis uniformes; mapas simples continuam nas fases seguintes.
        if task is None:
            eligible=[i for i,e in enumerate(self.entries) if e['dificuldade']<=stage]
            idx=int(self.rng.choice(eligible)); self.map=self.maps[idx]
            max_dist=[4.,8.,np.inf][stage]
            for _ in range(20000):
                component=self.map.components[int(self.rng.integers(len(self.map.components)))]
                start,goal=component[self.rng.integers(len(component),size=2)]
                xy=self.map.world(start); target=self.map.world(goal)
                if not .8 <= np.linalg.norm(xy-target)<=max_dist: continue
                pose=np.r_[xy,self.rng.uniform(-np.pi,np.pi)]
                if not self.collision(pose): break
            else: raise RuntimeError('Não foi possível sortear início/alvo.')
            # 20%: tenta orientar o início para perto de obstáculo frontal.
            # Não garante necessidade de ré; garante uma oportunidade útil.
            if stage>0 and self.rng.random()<.2:
                for angle in self.rng.permutation(np.linspace(-np.pi,np.pi,36,endpoint=False)):
                    candidate=np.r_[xy,angle]
                    fwd=candidate.copy(); back=candidate.copy()
                    front_blocked=False; back_clear=True
                    for _ in range(8):
                        fwd=self.motion(fwd,2); back=self.motion(back,6)
                        front_blocked |= self.collision(fwd)
                        back_clear &= not self.collision(back)
                    if front_blocked and back_clear: pose=candidate; break
            task=dict(map_index=idx,pose=pose.tolist(),goal=target.tolist())
        else:
            self.map=self.maps[int(task['map_index'])]
            pose=np.array(task['pose'],float); target=np.array(task['goal'],float)
        self.pose=pose.copy(); self.goal=target.copy(); self.task=task
        if self.collision(self.pose): raise ValueError('Início em colisão.')
        self.steps=0; self.known=np.full(self.map.grid.shape,-1,np.int8)
        self.traj=[self.pose[:2].copy()]; self.last_direction=1
        self.sense()
        return self.observation()

    def sense(self):
        # Raios interrompidos no primeiro obstáculo: não revela através de paredes.
        angles=self.pose[2]+np.linspace(-np.pi,np.pi,CONFIG['sensor_rays'],endpoint=False)
        active=np.ones(len(angles),bool)
        for d in np.arange(0,CONFIG['sensor_range']+.05,.1):
            x=self.pose[0]+d*np.cos(angles); y=self.pose[1]+d*np.sin(angles)
            c=np.floor(x/CONFIG['cell']).astype(int)
            r=np.floor((self.map.height-y)/CONFIG['cell']).astype(int)
            valid=active&(r>=0)&(c>=0)&(r<self.known.shape[0])&(c<self.known.shape[1])
            ii=np.flatnonzero(valid)
            self.known[r[ii],c[ii]]=self.map.grid[r[ii],c[ii]].astype(np.int8)
            active &= valid
            active[ii] &= ~self.map.grid[r[ii],c[ii]]
            if not active.any(): break

    def observation(self):
        # Janela gira com o robô: frente para cima e esquerda para a esquerda.
        vals=[]; x,y,t=self.pose; c,s=np.cos(t),np.sin(t)
        for forward in range(5,-6,-1):
            for right in range(-5,6):
                f=forward*.4; left=-right*.4
                wx=x+c*f-s*left; wy=y+s*f+c*left
                col=int(np.floor(wx/.2)); row=int(np.floor((self.map.height-wy)/.2))
                vals.append(self.known[row,col] if 0<=row<self.known.shape[0] and 0<=col<self.known.shape[1] else 1)
        vals=np.asarray(vals); dx,dy=self.goal-self.pose[:2]
        f=c*dx+s*dy; left=-s*dx+c*dy; d=np.hypot(dx,dy)
        scale=30.  # escala fixa entre mapas
        extra=[np.clip(f/scale,-1,1),np.clip(left/scale,-1,1),
               left/max(d,1e-9),f/max(d,1e-9),min(d/scale,1),self.steps/CONFIG['max_steps']]
        return np.r_[vals==1,vals!=-1,extra].astype(np.float32)

    def step(self,action):
        if not 0<=action<len(ACOES): raise ValueError('Ação inválida')
        previous=np.linalg.norm(self.goal-self.pose[:2]); self.steps+=1
        collision=False; success=False
        # Subpassos para conferir colisão e chegada durante o deslocamento.
        for _ in range(4):
            candidate=self.motion(self.pose,action,CONFIG['dt']/4)
            if self.collision(candidate): collision=True; break
            self.pose=candidate
            if np.linalg.norm(self.goal-self.pose[:2])<=CONFIG['goal_radius']:
                success=True; break
        distance=np.linalg.norm(self.goal-self.pose[:2])
        reverse=ACOES[action,1]<0
        # Sem bônus de exploração nesta primeira referência experimental.
        reward=-.03+2*(previous-distance)-.005*reverse
        if collision: reward-=50
        elif success: reward+=100
        truncated=self.steps>=CONFIG['max_steps'] and not (collision or success)
        self.sense(); self.traj.append(self.pose[:2].copy())
        info=dict(success=success,collision=collision,distance=float(distance),reverse=bool(reverse))
        return self.observation(),float(reward),bool(collision or success),bool(truncated),info

    def image(self):
        im=Image.fromarray(np.where(self.map.ocupado,0,255).astype('uint8')).convert('RGB')
        im.thumbnail((650,650)); draw=ImageDraw.Draw(im)
        def xy(p): return (float(p[0]/self.map.width*im.width),float((1-p[1]/self.map.height)*im.height))
        if len(self.traj)>1: draw.line([xy(p) for p in self.traj],fill='purple',width=2)
        x,y=xy(self.goal); draw.line((x-5,y-5,x+5,y+5),fill='green',width=2); draw.line((x-5,y+5,x+5,y-5),fill='green',width=2)
        x,y=xy(self.pose); draw.ellipse((x-4,y-4,x+4,y+4),fill='blue')
        return im
