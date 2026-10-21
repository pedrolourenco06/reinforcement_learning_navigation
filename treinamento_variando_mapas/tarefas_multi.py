"""Troca mapas/alvos sem alterar observação, ações, recompensa ou cinemática original."""
import json
import time
from pathlib import Path
import numpy as np
import pygame
from PIL import Image
from scipy.ndimage import distance_transform_edt, label
from ackermann_env_re import AckermannEnv, SCREEN_SIZE

class AmbienteMapas(AckermannEnv):
    def update_known_map(self, layers=2):
        # Equivalente ao laço original, vetorizado para a mesma janela em pixels.
        lin,col=self.get_robot_cell()
        ry=int(np.ceil(layers*self.res*self.my)); rx=int(np.ceil(layers*self.res*self.mx))
        r0,r1=max(0,lin-ry),min(self.nrow,lin+ry+1)
        c0,c1=max(0,col-rx),min(self.ncol,col+rx+1)
        view=self.known_map[r0:r1,c0:c1]
        self.info_gain=int(np.count_nonzero(view==-1))
        view[:]=(self.mapa[r0:r1,c0:c1]<127).astype(np.int8)
        x,y=self.mts2px(self.alvo)
        r=int(np.clip(int(y),0,self.nrow-1)); c=int(np.clip(int(x),0,self.ncol-1))
        if r0<=r<r1 and c0<=c<c1: self.known_map[r,c]=2

    def visualizar(self, ativo):
        if ativo:
            pygame.init()
            if not pygame.display.get_init() or pygame.display.get_surface() is None:
                self.screen_size=(SCREEN_SIZE,SCREEN_SIZE)
                self.screen=pygame.display.set_mode(self.screen_size)
                self.clock=pygame.time.Clock()
            self.render_env=True
            self.atualizar_fundo()
        else:
            # Fecha a janela nos episódios sem render, sem encerrar o treino.
            self.render_env=False
            if pygame.display.get_init(): pygame.display.quit()

    def atualizar_fundo(self):
        if not self.render_env: return
        rgb=np.repeat(self.mapa[:,:,None],3,axis=2)
        surf=pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
        self.map_surface=pygame.transform.scale(surf,self.screen_size)

class GestorMapas:
    def __init__(self,manifesto,split='treino',env=None,mapa_unico=None,ppm=100.):
        self.path=Path(manifesto).resolve()
        if mapa_unico is not None:
            self.entries=[dict(arquivo=str(Path(mapa_unico).resolve()),px_por_metro=ppm,split=split)]
        else:
            self.entries=[e for e in json.loads(self.path.read_text(encoding='utf-8')) if e['split']==split]
        if not self.entries: raise ValueError(f'Sem mapas para {split}')
        self.cache={}; self.current=None
        first=self.dados(0)
        self.env=env or AmbienteMapas(img=str(first['path']),xlim=np.array([0.,first['width']]),
            ylim=np.array([0.,first['height']]),alvo=np.array([first['width']/2,first['height']/2]),
            render=False,continuous_obs=True,window_layers=5,reset_known_map_each_episode=True,
            wheelbase=.4,robot_length=.545,robot_width=.415,max_steering_deg=20.,speed=.5,dt=.2,allow_reverse=True)
        self.max_dist=max(np.hypot(self.dados(i)['width'],self.dados(i)['height']) for i in range(len(self.entries)))

    def dados(self,i):
        if i in self.cache: return self.cache[i]
        entry=self.entries[i]; p=Path(entry['arquivo'])
        if not p.is_absolute(): p=self.path.parent/p
        ppm=float(entry['px_por_metro'])
        if not np.isfinite(ppm) or ppm<=0: raise ValueError('Escala inválida')
        img=np.where(np.asarray(Image.open(p).convert('L'))>127,255,0).astype(np.uint8)
        h,w=img.shape
        # Grade conservadora apenas para sortear tarefas conectadas.
        k=max(1,round(.20*ppm)); nr,nc=h//k,w//k
        if min(nr,nc)<4: raise ValueError('Mapa pequeno demais')
        blocks=img[:nr*k,:nc*k].reshape(nr,k,nc,k)
        free=(blocks>127).all(axis=(1,3)); cell=k/ppm
        clearance=distance_transform_edt(np.pad(free,1,constant_values=False))[1:-1,1:-1]*cell
        safe=clearance>np.hypot(.545,.415)/2+np.sqrt(2)*cell/2+.02
        labels,_=label(safe,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        ids,counts=np.unique(labels[safe],return_counts=True)
        components=[np.argwhere(labels==v) for v,n in zip(ids,counts) if n>10]
        if not components: raise ValueError(f'Sem área conectada suficiente em {p}')
        data=dict(path=p,ppm=ppm,img=img,width=w/ppm,height=h/ppm,labels=labels,components=components,cell=cell)
        self.cache[i]=data
        return data

    def ativar(self,i):
        d=self.dados(i); e=self.env
        key=str(d['path'].resolve())
        if getattr(e,'_map_key',None)==key:
            self.current=i
            return
        e.xlim=np.array([0.,d['width']]);e.ylim=np.array([0.,d['height']])
        e.mapa=d['img'];e.nrow,e.ncol=e.mapa.shape
        e.mx=e.my=d['ppm']; e.known_map=np.full(e.mapa.shape,-1,np.int8)
        self.current=i; e._map_key=key; e.atualizar_fundo()

    def sorteio(self,dist_max=None):
        i=int(np.random.randint(len(self.entries))); self.ativar(i)
        d=self.dados(i); e=self.env
        for _ in range(20000):
            component=d['components'][int(np.random.randint(len(d['components'])))]
            rc1,rc2=component[np.random.randint(len(component),size=2)]
            def world(rc):
                # Pequeno jitter conserva a margem do disco de segurança.
                r,c=rc
                return np.array([(c+.5)*d['cell'],d['height']-(r+.5)*d['cell']])
            start,goal=world(rc1),world(rc2)
            distance=np.linalg.norm(goal-start)
            if distance<=.30 or (dist_max is not None and distance>dist_max): continue
            pose=np.r_[start,np.random.uniform(-np.pi,np.pi)].astype(np.float32)
            if e.collision(pose) or e.collision(np.r_[goal,0.]): continue
            return dict(mapa=i,arquivo=self.entries[i]['arquivo'],alvo=goal.tolist(),pose=pose.tolist())
        raise RuntimeError('Não foi possível sortear início/alvo conectados.')

    def reset(self,tarefa):
        self.ativar(int(tarefa['mapa']))
        self.env.alvo=np.asarray(tarefa['alvo'],dtype=np.float64)
        return self.env.reset(initial_pose=np.asarray(tarefa['pose'],np.float32))

    def tarefas_fixas(self,n,seed):
        state=np.random.get_state()
        try:
            np.random.seed(seed)
            return [self.sorteio() for _ in range(n)]
        finally: np.random.set_state(state)

def escolher_dispositivo(nome):
    import torch
    if nome=='cpu': return torch.device('cpu')
    if nome in ('cuda','auto') and torch.cuda.is_available(): return torch.device('cuda')
    if nome=='cuda': raise RuntimeError('CUDA indisponível')
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        if nome=='directml': raise
        return torch.device('cpu')

def esperar(segundos=0.,aguardar_enter=False):
    fim=time.monotonic()+segundos; pausado=False
    while True:
        for evento in pygame.event.get():
            if evento.type==pygame.QUIT: return False
            if evento.type==pygame.KEYDOWN:
                if evento.key==pygame.K_ESCAPE: return False
                if evento.key==pygame.K_RETURN and aguardar_enter: return True
                if evento.key==pygame.K_SPACE: pausado=not pausado
        if not pausado and not aguardar_enter and time.monotonic()>=fim: return True
        pygame.time.wait(10)

def escrever_csv(path,rows):
    import csv
    if not rows: return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
