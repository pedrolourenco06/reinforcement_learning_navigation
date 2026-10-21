# Preenche mapas PNG com borda preta (imagem centralizada) até o tamanho exigido pela área do ambiente.
#
# Padrão: 1920 x 2400 px = 19,2 x 24 m a 100 px/m, que é a área xlim x ylim do CONFIG_ENV em
# labirinto_dqn_multi.py (a mesma do labirinto6.png). Um mapa de 1680 x 1920 px (grade 14 x 16 com
# células de 1,2 m) ganha 120 px de borda em cada lado na horizontal e 240 px em cima e embaixo.
# O preenchimento mantém 100 px/m e o desenho original; não estica nada.
#
# Equivale a, no gerador de mapas:  grid = np.pad(grid, ((2, 2), (1, 1)))   # 16x14 -> 20x16 células
#
# Exemplos:
#   python preencher_mapas.py mapas_teste --saida mapas_teste_corrigidos
#   python preencher_mapas.py circuito_teste.png obstaculos_21*.png --saida corrigidos

import argparse
import glob
from pathlib import Path

import numpy as np
from PIL import Image


def expandir(entradas):
    # Aceita arquivos, pastas (*.png) e padrões (o shell do Windows não expande padrões).
    arquivos = []
    for e in entradas:
        p = Path(e)
        if p.is_dir():
            arquivos.extend(sorted(p.glob('*.png')))
        elif any(c in str(e) for c in '*?['):
            arquivos.extend(Path(x) for x in sorted(glob.glob(str(e))))
        else:
            arquivos.append(p)
    vistos, unicos = set(), []
    for a in arquivos:
        if a.resolve() not in vistos:
            vistos.add(a.resolve())
            unicos.append(a)
    return unicos


def main():
    parser = argparse.ArgumentParser(description='Preenche mapas PNG até o tamanho da área do ambiente.')
    parser.add_argument('entradas', nargs='+', help='Arquivos, pastas (*.png) ou padrões')
    parser.add_argument('--saida', type=Path, required=True, help='Pasta de destino (diferente da de origem)')
    parser.add_argument('--largura', type=int, default=1920, help='Largura final em px (padrão: 1920)')
    parser.add_argument('--altura', type=int, default=2400, help='Altura final em px (padrão: 2400)')
    args = parser.parse_args()

    arquivos = expandir(args.entradas)
    faltando = [str(a) for a in arquivos if not a.is_file()]
    if not arquivos or faltando:
        parser.error(f'Arquivos não encontrados: {faltando or args.entradas}')
    nomes = [a.name for a in arquivos]
    if len(set(nomes)) != len(nomes):
        parser.error('Dois arquivos com o mesmo nome; renomeie um deles.')
    args.saida.mkdir(parents=True, exist_ok=True)
    for a in arquivos:
        if a.resolve() == (args.saida / a.name).resolve():
            parser.error('A pasta de saída não pode ser a mesma da origem (sobrescreveria o original).')

    # valida tudo antes de gravar qualquer arquivo
    imagens = {}
    for a in arquivos:
        cinza = np.array(Image.open(a).convert('L'))
        h, w = cinza.shape
        if w > args.largura or h > args.altura:
            parser.error(f'{a.name}: {w}x{h} px é maior que o destino {args.largura}x{args.altura}.')
        if (args.largura - w) % 2 or (args.altura - h) % 2:
            parser.error(f'{a.name}: {w}x{h} px não dá para centralizar em {args.largura}x{args.altura} '
                         f'(a diferença precisa ser par).')
        imagens[a] = cinza

    for a, cinza in imagens.items():
        h, w = cinza.shape
        px, py = (args.largura - w) // 2, (args.altura - h) // 2
        # mesma binarização do ambiente (limiar 127)
        saida = np.zeros((args.altura, args.largura), dtype=np.uint8)
        saida[py:py + h, px:px + w] = np.where(cinza > 127, 255, 0)
        Image.fromarray(saida).save(args.saida / a.name)
        print(f'{a.name}: {w}x{h} -> {args.largura}x{args.altura} (borda: {px} px nas laterais, {py} px em cima e embaixo)')
    print(f'{len(imagens)} mapa(s) gravado(s) em {args.saida}')


if __name__ == '__main__':
    main()