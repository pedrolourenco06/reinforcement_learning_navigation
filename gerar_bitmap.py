from PIL import Image
import numpy as np

# =========================================================
# PARAMETROS
# =========================================================

PX_POR_METRO = 100
LARGURA_CORREDOR_M = 2.40
CELL = int(round(LARGURA_CORREDOR_M * PX_POR_METRO))   # 240 px

# grade do mapa
# 1 = corredor (branco)
# 0 = parede (preto)
#
# Este layout foi montado para ficar mais parecido com o estilo
# que você gostou: grandes áreas pretas e corredores largos,
# com poucas bifurcações, mais limpo e mais controlado.

grid = np.array([
    [0,0,0,0,0,0,0,0],
    [0,1,0,0,1,1,1,0],
    [0,1,1,1,1,0,1,0],
    [0,0,1,0,1,1,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,1,1,1,1,1,0],
    [0,1,1,0,0,0,1,0],
    [0,1,0,0,1,0,1,0],
    [0,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0],
], dtype=np.uint8)

# =========================================================
# GERACAO DA IMAGEM
# =========================================================

h, w = grid.shape
img_array = np.kron(grid, np.ones((CELL, CELL), dtype=np.uint8)) * 255

img = Image.fromarray(img_array, mode="L")
img.save("labirinto6.png")

# =========================================================
# INFORMACOES
# =========================================================

altura_px, largura_px = img_array.shape
altura_m = altura_px / PX_POR_METRO
largura_m = largura_px / PX_POR_METRO

print("Arquivo gerado: labirinto3.png")
print(f"Resolucao: {largura_px} x {altura_px} px")
print(f"Tamanho fisico: {largura_m:.2f} m x {altura_m:.2f} m")
print(f"Largura do corredor: {CELL} px = {LARGURA_CORREDOR_M:.2f} m")
print(f"Escala: {PX_POR_METRO} px/m")