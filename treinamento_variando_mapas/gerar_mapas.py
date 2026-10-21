"""Gera mapas binários; branco=livre, preto=ocupado. python gerar_mapas.py"""
from pathlib import Path
import json
import numpy as np
from PIL import Image

PX_POR_METRO = 100
TAMANHO_CELULA_M = 1.20
CELL = round(TAMANHO_CELULA_M * PX_POR_METRO)
# Aproximação do circuito da referência; sem a margem branca externa.
GRID_CIRCUITO = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,1,1,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.uint8)

def base():
    g = np.ones((16,14), np.uint8)
    g[[0,-1],:] = 0
    g[:,[0,-1]] = 0
    return g

def gerar(destino='mapas'):
    destino = Path(destino)
    itens=[]
    def salvar(nome,g,split,dificuldade):
        pasta=destino/split
        pasta.mkdir(parents=True,exist_ok=True)
        path=pasta/(nome+'.png')
        Image.fromarray(np.kron(g,np.ones((CELL,CELL),np.uint8))*255).save(path)
        itens.append(dict(arquivo=str(path.relative_to(destino)),px_por_metro=PX_POR_METRO,
                          dificuldade=dificuldade,split=split))
    salvar('aberto',base(),'treino',0)
    g=base(); g[5:7,4:6]=0; g[10:12,9:11]=0
    salvar('dois_obstaculos',g,'treino',1)
    salvar('circuito_referencia',GRID_CIRCUITO,'treino',2)
    g=base(); g[5:11,4]=0; g[10,4:10]=0; g[5:11,9]=0
    salvar('rua_sem_saida',g,'treino',2)
    for split,seeds in [('treino',range(10,18)),('validacao',range(110,113)),('teste',range(210,213))]:
        for seed in seeds:
            rng=np.random.default_rng(seed); g=base()
            for _ in range(3):
                y=int(rng.integers(3,12)); x=int(rng.integers(3,10))
                g[y:y+2,x:x+2]=0
            salvar(f'obstaculos_{seed}',g,split,1)
    g=base(); g[4:12,5:9]=0
    salvar('circuito_validacao',g,'validacao',2)
    g=base(); g[5:11,4:10]=0
    salvar('circuito_teste',g,'teste',2)
    (destino/'manifesto.json').write_text(json.dumps(itens,indent=2),encoding='utf-8')
    print(f'{len(itens)} mapas em {destino.resolve()}')

if __name__=='__main__': gerar()
