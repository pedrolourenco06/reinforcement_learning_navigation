"""Recupera gráficos dos CSVs do pacote anterior ou desta revisão."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def media(v,n=50):
    v=np.asarray(v,float); s=np.r_[0.,np.cumsum(v)];b=np.arange(1,len(v)+1);a=np.maximum(0,b-n)
    return (s[b]-s[a])/(b-a)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pasta',type=Path,required=True)
    a=p.parse_args(); source=a.pasta/'episodios.csv'
    with source.open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError('CSV sem episódios')
    def values(*keys):
        key=next((k for k in keys if k in rows[0]),None)
        if key is None:return None
        return [float(r[key]) for r in rows]
    ep=values('episodio','episode')
    for name,keys,label in [('dqn_reward',('recompensa','reward'),'Recompensa'),
                            ('dqn_success_rate',('sucesso','success'),'Taxa de sucesso'),
                            ('dqn_steps',('passos','steps'),'Passos por episódio')]:
        v=values(*keys)
        if v is None:continue
        fig,ax=plt.subplots()
        if name!='dqn_success_rate': ax.plot(ep,v,alpha=.25,label='Por episódio')
        ax.plot(ep,media(v),label='Média móvel 50');ax.set(xlabel='Episódios',ylabel=label)
        if name=='dqn_success_rate':ax.set_ylim(-.05,1.05)
        ax.legend();ax.grid(True);fig.savefig(a.pasta/(name+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    print('Gráficos salvos em',a.pasta.resolve())
    print('A loss não pode ser reconstruída se não foi registrada no treino anterior.')
if __name__=='__main__':main()
