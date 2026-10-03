# # -*- coding: utf-8 -*-
# # Introducao ao Aprendizado por Reforco - PPGEE
# # Prof. Armando Alves Neto
# #
# # VERSAO CORRIGIDA
# # Principais mudancas em relacao a versao original:
# #   1. update_known_map vetorizado  (~1000x mais rapido) e raio de sensor
# #      configuravel (sensor_layers) -> tem de ser MAIOR que o raio de giro.
# #   2. collision vetorizado (sem laco Python sobre o footprint).
# #   3. Campo de distancia geodesica ate o alvo (Dijkstra) -> shaping sem
# #      minimos locais. Campo de clearance (distancia ao obstaculo mais proximo).
# #   4. Observacao egocentrica: raios tipo lidar + janela rotacionada pelo
# #      heading + alvo no referencial do robo + erro de heading em sin/cos.
# #   5. step() devolve terminated e truncated separados (bootstrap correto).
# #   6. Recompensa reescrita: shaping potencial normalizado, bonus de
# #      informacao normalizado (nao saturado), terminais na mesma escala.
# #   7. Acao de re opcional e action_repeat.
# #   8. Espaco de acoes = EXATAMENTE 3 esterçamentos (-max,0,+max) x 2 sentidos
# #      (frente/re) = 6 acoes, como especificado (nao 5 esterçamentos).
# #   9. Controle de QUANDO renderizar fica so no script de treino (chama
# #      env.render() por fora, gated por episode % render_every), como no
# #      codigo original. render_fps (derivado de dt) evita que a janela
# #      rode mais rapido que a fisica simulada.
# ########################################################################
# try:
#     import gymnasium as gym
#     from gymnasium import spaces
# except ImportError:
#     import gym
#     from gym import spaces

# import os
# import heapq
# import hashlib
# import numpy as np

# # Globais
# MAX_STEPS = 500          # numero maximo de DECISOES por episodio
# SCREEN_SIZE = 500


# ########################################################################
# class AckermannEnv(gym.Env):

#     ####################################################################
#     def __init__(
#             self,
#             xlim=np.array([0.0, 19.2]),
#             ylim=np.array([0.0, 24.0]),
#             res=0.4,
#             img='labirinto6.png',
#             alvo=np.array([13.2, 12.2]),
#             render=False,
#             continuous_obs=True,
#             window_layers=3,                 # janela egocentrica grossa (7x7)
#             window_res=0.6,                  # espacamento da janela, em metros
#             sensor_layers=6,                 # raio do sensor = sensor_layers*res
#             reset_known_map_each_episode=True,
#             wheelbase=0.4,
#             robot_length=0.545,
#             robot_width=0.415,
#             max_steering_deg=20.0,
#             speed=0.5,
#             dt=0.2,
#             # ---- novos ----
#             allow_reverse=True,
#             action_repeat=4,
#             n_rays=16,
#             ray_max_range=3.0,
#             ray_step=0.10,
#             gamma=0.99,                      # usado no shaping potencial
#             goal_tol=0.70,   # >= comprimento do robo: com raio de giro 1.12 m
#                              # um alvo de 0.30 m e inalcancavel a curta distancia
#             geodesic_res=0.05,               # resolucao do campo geodesico
#             cache_dir='cache',
#             render_playback_speed=1.0,       # 1.0 = tempo real (1 sub-passo = dt segundos)
#     ):
#         # ---------------- geometria do mundo ----------------
#         self.xlim = np.asarray(xlim, dtype=np.float64)
#         self.ylim = np.asarray(ylim, dtype=np.float64)
#         self.res = res

#         # ---------------- parametros fisicos ----------------
#         self.wheelbase = wheelbase
#         self.robot_length = robot_length
#         self.robot_width = robot_width
#         self.max_steering_deg = max_steering_deg
#         self.speed = speed
#         self.dt = dt
#         self.passo_max = speed * dt                      # metros por sub-passo

#         # ---------------- acoes ----------------
#         # 3 esterçamentos apenas: -max, 0, +max (nao ha meio-esterçamento)
#         steering = np.array([
#             -max_steering_deg,
#             0.0,
#             max_steering_deg,
#         ], dtype=np.float32)

#         self.allow_reverse = allow_reverse
#         if allow_reverse:
#             # (esterçamento, sentido)  sentido = +1 frente, -1 re
#             self.actions = [(s, +1) for s in steering] + \
#                            [(s, -1) for s in steering]
#         else:
#             self.actions = [(s, +1) for s in steering]
#         self.actions = np.array(self.actions, dtype=np.float32)

#         self.action_space = spaces.Discrete(len(self.actions))
#         self.action_repeat = int(max(1, action_repeat))

#         # ---------------- observacao ----------------
#         self.continuous_obs = continuous_obs
#         self.window_layers = window_layers
#         self.window_res = window_res
#         self.sensor_layers = sensor_layers
#         self.n_rays = n_rays
#         self.ray_max_range = ray_max_range
#         self.ray_step = ray_step
#         self.gamma = gamma
#         self.goal_tol = goal_tol
#         self.reset_known_map_each_episode = reset_known_map_each_episode

#         self.alvo = np.asarray(alvo, dtype=np.float64)
#         self.render_env = render

#         # ---------------- mapa ----------------
#         self.init2D(img)
#         self.build_footprint_samples(spacing=0.04)

#         # raio do sensor, em pixels
#         self.sensor_radius_m = self.sensor_layers * self.res
#         self.sensor_rx = int(np.ceil(self.sensor_radius_m * self.mx))
#         self.sensor_ry = int(np.ceil(self.sensor_radius_m * self.my))

#         # verificacao de sanidade: sensor precisa enxergar mais longe que o
#         # raio de giro minimo, senao o desvio e geometricamente impossivel
#         self.raio_giro_min = self._raio_giro_minimo()
#         if self.sensor_radius_m < 1.2 * self.raio_giro_min:
#             print(f"[AVISO] raio do sensor ({self.sensor_radius_m:.2f} m) < 1.2x "
#                   f"raio de giro minimo ({self.raio_giro_min:.2f} m). "
#                   f"Aumente sensor_layers para >= "
#                   f"{int(np.ceil(1.2*self.raio_giro_min/self.res))}.")

#         # ganho maximo de informacao por sub-passo (para normalizar o bonus)
#         passo_px = self.passo_max * self.mx
#         self.ganho_max = max(1.0, passo_px * (2 * self.sensor_ry + 1))

#         # ---------------- pesos da recompensa ----------------
#         # densos (acumulam ate 2000x por episodio!)
#         # Calibrados para: colisao (-30) ser PIOR que timeout (~-10).
#         # Soma das penalidades densas <= ~0.009/sub-passo * 2000 = -18 no pior caso.
#         self.W_PROG = 0.200     # por (metro de progresso / passo_max)
#         self.W_TEMPO = 0.0020
#         self.W_ALINHA = 0.0015  # penalidade pura, <= 0
#         self.W_CLEAR = 0.0030   # penalidade pura, <= 0
#         self.W_SUAVE = 0.0010
#         self.W_RE = 0.0015
#         self.W_INFO = 5.0       # orcamento TOTAL: +5 se explorar 100% do mapa

#         # ---------------- campos pre-computados ----------------
#         self.cache_dir = cache_dir
#         self._build_clearance_field()
#         self._build_geodesic_field(geodesic_res, img)

#         self.known_map = -np.ones_like(self.mapa, dtype=np.int8)
#         self.n_cells = self.known_map.size

#         # offsets dos raios e da janela (pre-computados no frame do robo)
#         self._build_ray_offsets()
#         self._build_window_offsets()

#         obs_dim = self.n_rays + (2 * self.window_layers + 1) ** 2 + 13
#         self.observation_space = spaces.Box(
#             low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
#         self.obs_dim = obs_dim

#         # ---------------- render ----------------
#         # fps = velocidade_de_reproducao / dt  ->  1 sub-passo do robo (dt
#         # segundos de simulacao) e desenhado em (dt / velocidade) segundos reais
#         self.render_fps = max(1.0, render_playback_speed / self.dt)
#         self.screen = None
#         if self.render_env:
#             self._init_render()

#     # ==================================================================
#     # MAPA
#     # ==================================================================
#     def init2D(self, image):
#         """Carrega o bitmap. Usa PIL (sem depender de janela do pygame)."""
#         try:
#             from PIL import Image
#             I_gray = np.array(Image.open(image).convert('L'))
#         except ImportError:
#             import pygame
#             pygame.init()
#             pygame.display.set_mode((1, 1))
#             surf = pygame.image.load(image).convert()
#             arr = pygame.surfarray.pixels3d(surf).transpose(1, 0, 2)
#             I_gray = np.mean(arr, axis=2).astype(np.uint8)
#             del arr
#             pygame.quit()

#         self.nrow, self.ncol = I_gray.shape
#         self.mapa = np.where(I_gray > 127, 255, 0).astype(np.uint8)
#         self.free = self.mapa > 127

#         self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
#         self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])

#     def build_footprint_samples(self, spacing=0.04):
#         xs = np.arange(-self.robot_length / 2, self.robot_length / 2 + spacing, spacing)
#         ys = np.arange(-self.robot_width / 2, self.robot_width / 2 + spacing, spacing)
#         xx, yy = np.meshgrid(xs, ys)
#         self.footprint_local = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)

#     def _raio_giro_minimo(self):
#         lr = self.wheelbase / 2.0
#         delta = np.deg2rad(self.max_steering_deg)
#         beta = np.arctan(lr / self.wheelbase * np.tan(delta))
#         dtheta = (self.speed / lr) * np.sin(beta) * self.dt
#         return abs(self.passo_max / dtheta) if abs(dtheta) > 1e-9 else np.inf

#     # ------------------------------------------------------------------
#     def _cache_path(self, tag, img):
#         os.makedirs(self.cache_dir, exist_ok=True)
#         h = hashlib.md5(
#             f"{img}|{self.nrow}x{self.ncol}|{self.alvo}|{self.robot_width}".encode()
#         ).hexdigest()[:10]
#         return os.path.join(self.cache_dir, f"{tag}_{h}.npy")

#     def _build_clearance_field(self):
#         """clearance[lin,col] = distancia em metros ate o obstaculo mais proximo."""
#         try:
#             from scipy.ndimage import distance_transform_edt
#             d = distance_transform_edt(self.free, sampling=(1.0 / self.my, 1.0 / self.mx))
#         except ImportError:
#             print("[AVISO] scipy nao encontrado: clearance desativado (pip install scipy)")
#             d = np.full(self.free.shape, 10.0)
#             d[~self.free] = 0.0
#         self.clearance = d.astype(np.float32)
#         self.clearance_max = float(self.clearance.max())

#     # ------------------------------------------------------------------
#     def _build_geodesic_field(self, gres, img):
#         """
#         Dijkstra 8-conectado a partir do alvo, sobre as celulas onde o centro
#         do robo cabe (clearance > meia-largura). Substitui a distancia
#         euclidiana no shaping e elimina os minimos locais do labirinto.
#         """
#         path = self._cache_path(f"geo_{gres}", img)
#         if os.path.exists(path):
#             self.geo = np.load(path)
#             self.geo_res = gres
#             self.geo_nrow, self.geo_ncol = self.geo.shape
#             self.geo_max = float(np.median(self.geo)) * 0 + float(self.geo.max()) / 2.0
#             return

#         gr = int(round((self.ylim[1] - self.ylim[0]) / gres))
#         gc = int(round((self.xlim[1] - self.xlim[0]) / gres))

#         # amostra o campo de clearance na grade grossa
#         lin = np.clip(((np.arange(gr) + 0.5) * gres * self.my).astype(int), 0, self.nrow - 1)
#         col = np.clip(((np.arange(gc) + 0.5) * self.mx * gres).astype(int), 0, self.ncol - 1)
#         # atencao: geo[0] = y baixo -> precisa inverter para casar com a imagem
#         lin_img = self.nrow - 1 - lin
#         clear_g = self.clearance[np.ix_(lin_img, col)]

#         # margem = meia-diagonal: garante que o robo cabe em QUALQUER orientacao
#         margem = np.hypot(self.robot_length, self.robot_width) / 2.0 + 0.02
#         livre = clear_g > margem

#         # custo extra perto de parede -> o campo geodesico prefere o meio do
#         # corredor, e o shaping deixa de guiar o robo para raspar nos muros
#         penal = 1.0 + 3.0 * np.exp(-(np.maximum(clear_g - margem, 0.0)) / 0.35)

#         INF = np.inf
#         dist = np.full((gr, gc), INF, dtype=np.float64)

#         gj = int(np.clip((self.alvo[0] - self.xlim[0]) / gres, 0, gc - 1))
#         gi = int(np.clip((self.alvo[1] - self.ylim[0]) / gres, 0, gr - 1))
#         if not livre[gi, gj]:
#             # procura a celula livre mais proxima do alvo
#             ii, jj = np.nonzero(livre)
#             k = np.argmin((ii - gi) ** 2 + (jj - gj) ** 2)
#             gi, gj = int(ii[k]), int(jj[k])

#         dist[gi, gj] = 0.0
#         pq = [(0.0, gi, gj)]
#         s2 = np.sqrt(2.0)
#         nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
#               (-1, -1, s2), (-1, 1, s2), (1, -1, s2), (1, 1, s2)]

#         while pq:
#             d, i, j = heapq.heappop(pq)
#             if d > dist[i, j]:
#                 continue
#             for di, dj, w in nb:
#                 ni, nj = i + di, j + dj
#                 if ni < 0 or ni >= gr or nj < 0 or nj >= gc:
#                     continue
#                 if not livre[ni, nj]:
#                     continue
#                 nd = d + w * gres * 0.5 * (penal[i, j] + penal[ni, nj])
#                 if nd < dist[ni, nj]:
#                     dist[ni, nj] = nd
#                     heapq.heappush(pq, (nd, ni, nj))

#         finite = np.isfinite(dist)
#         alcancavel_max = float(dist[finite].max()) if finite.any() else 1.0
#         dist[~finite] = 2.0 * alcancavel_max          # celulas nao conectadas

#         self.geo = dist.astype(np.float32)
#         self.geo_res = gres
#         self.geo_nrow, self.geo_ncol = gr, gc
#         self.geo_max = alcancavel_max                 # usado para normalizar
#         np.save(path, self.geo)

#     def dist_geodesica(self, p):
#         """Distancia geodesica (em metros) do ponto p ate o alvo."""
#         j = int(np.clip((p[0] - self.xlim[0]) / self.geo_res, 0, self.geo_ncol - 1))
#         i = int(np.clip((p[1] - self.ylim[0]) / self.geo_res, 0, self.geo_nrow - 1))
#         return float(self.geo[i, j])

#     def get_clearance(self, p):
#         px, py = self.mts2px(p)
#         lin = int(np.clip(py, 0, self.nrow - 1))
#         col = int(np.clip(px, 0, self.ncol - 1))
#         return float(self.clearance[lin, col])

#     # ==================================================================
#     # CINEMATICA E COLISAO
#     # ==================================================================
#     @staticmethod
#     def wrap_angle(theta):
#         return (theta + np.pi) % (2 * np.pi) - np.pi

#     def ackermann_step(self, steering_deg, direction=+1):
#         x, y, theta = self.pose
#         delta = np.deg2rad(steering_deg)
#         lr = self.wheelbase / 2.0
#         beta = np.arctan(lr / self.wheelbase * np.tan(delta))
#         v = self.speed * direction
#         x_new = x + v * np.cos(theta + beta) * self.dt
#         y_new = y + v * np.sin(theta + beta) * self.dt
#         theta_new = self.wrap_angle(theta + (v / lr) * np.sin(beta) * self.dt)
#         return np.array([x_new, y_new, theta_new], dtype=np.float64)

#     def collision(self, q):
#         """Versao vetorizada: sem laco Python sobre os pontos do footprint."""
#         q = np.asarray(q, dtype=np.float64)
#         if q.size == 2:
#             x, y, theta = q[0], q[1], 0.0
#         else:
#             x, y, theta = q[0], q[1], q[2]

#         c, s = np.cos(theta), np.sin(theta)
#         lx = self.footprint_local[:, 0]
#         ly = self.footprint_local[:, 1]
#         wx = x + c * lx - s * ly
#         wy = y + s * lx + c * ly

#         if (wx <= self.xlim[0]).any() or (wx >= self.xlim[1]).any():
#             return True
#         if (wy <= self.ylim[0]).any() or (wy >= self.ylim[1]).any():
#             return True

#         col = ((wx - self.xlim[0]) * self.mx).astype(np.int32)
#         lin = (self.nrow - (wy - self.ylim[0]) * self.my).astype(np.int32)

#         if (lin < 0).any() or (lin >= self.nrow).any():
#             return True
#         if (col < 0).any() or (col >= self.ncol).any():
#             return True

#         return bool((self.mapa[lin, col] < 127).any())

#     def mts2px(self, q):
#         px = (q[0] - self.xlim[0]) * self.mx
#         py = self.nrow - (q[1] - self.ylim[0]) * self.my
#         return px, py

#     def get_robot_cell(self):
#         px, py = self.mts2px(self.p)
#         return (int(np.clip(py, 0, self.nrow - 1)),
#                 int(np.clip(px, 0, self.ncol - 1)))

#     # ==================================================================
#     # MAPA DE INFORMACAO  (vetorizado)
#     # ==================================================================
#     def update_known_map(self):
#         lin, col = self.get_robot_cell()
#         ry, rx = self.sensor_ry, self.sensor_rx

#         i0, i1 = max(0, lin - ry), min(self.nrow, lin + ry + 1)
#         j0, j1 = max(0, col - rx), min(self.ncol, col + rx + 1)

#         sub_known = self.known_map[i0:i1, j0:j1]
#         prev_unknown = int(np.count_nonzero(sub_known == -1))

#         sub_map = self.mapa[i0:i1, j0:j1]
#         np.copyto(sub_known, np.where(sub_map < 127, 1, 0).astype(np.int8))

#         # marca a celula do alvo, se estiver dentro do campo de visao
#         apx, apy = self.mts2px(self.alvo)
#         ai = int(np.clip(apy, 0, self.nrow - 1))
#         aj = int(np.clip(apx, 0, self.ncol - 1))
#         if i0 <= ai < i1 and j0 <= aj < j1:
#             self.known_map[ai, aj] = 2

#         self.info_gain = prev_unknown            # celulas novas reveladas
#         self.n_known += prev_unknown

#     # ==================================================================
#     # OBSERVACAO
#     # ==================================================================
#     def _build_ray_offsets(self):
#         """Pontos de amostragem de cada raio, no referencial do robo."""
#         n_samp = int(self.ray_max_range / self.ray_step)
#         ang = np.linspace(-np.pi, np.pi, self.n_rays, endpoint=False)
#         r = (np.arange(1, n_samp + 1) * self.ray_step)
#         self.ray_r = r
#         self.ray_dx = np.cos(ang)[:, None] * r[None, :]      # (n_rays, n_samp)
#         self.ray_dy = np.sin(ang)[:, None] * r[None, :]

#     def _build_window_offsets(self):
#         k = self.window_layers
#         g = np.arange(-k, k + 1) * self.window_res
#         gx, gy = np.meshgrid(g, g[::-1])                     # frente = topo
#         self.win_dx = gx.ravel()
#         self.win_dy = gy.ravel()

#     def _sample_known(self, wx, wy):
#         """Amostra known_map em coordenadas de mundo (arrays). Vetorizado."""
#         col = ((wx - self.xlim[0]) * self.mx).astype(np.int32)
#         lin = (self.nrow - (wy - self.ylim[0]) * self.my).astype(np.int32)
#         fora = (lin < 0) | (lin >= self.nrow) | (col < 0) | (col >= self.ncol)
#         lin = np.clip(lin, 0, self.nrow - 1)
#         col = np.clip(col, 0, self.ncol - 1)
#         v = self.known_map[lin, col]
#         v = np.where(fora, 1, v)                             # fora = obstaculo
#         return v

#     def get_observation(self):
#         x, y, theta = self.pose
#         c, s = np.cos(theta), np.sin(theta)

#         # ---------- raios tipo lidar, egocentricos ----------
#         wx = x + c * self.ray_dx - s * self.ray_dy
#         wy = y + s * self.ray_dx + c * self.ray_dy
#         v = self._sample_known(wx.ravel(), wy.ravel()).reshape(self.ray_dx.shape)

#         bloqueado = (v == 1)
#         tem = bloqueado.any(axis=1)
#         idx = np.argmax(bloqueado, axis=1)
#         dist_raio = np.where(tem, self.ray_r[idx], self.ray_max_range)
#         rays = (dist_raio / self.ray_max_range).astype(np.float32)   # [0,1]

#         # ---------- janela egocentrica grossa (info de exploracao) ----------
#         wwx = x + c * self.win_dx - s * self.win_dy
#         wwy = y + s * self.win_dx + c * self.win_dy
#         wv = self._sample_known(wwx, wwy).astype(np.float32)
#         # -1 desconhecido, 0 livre, 1 obstaculo, 2 alvo -> 0.5
#         win = np.where(wv == 2, 0.5, wv).astype(np.float32)

#         # ---------- alvo no referencial do robo ----------
#         dx_w = self.alvo[0] - x
#         dy_w = self.alvo[1] - y
#         dx_r = c * dx_w + s * dy_w
#         dy_r = -s * dx_w + c * dy_w
#         erro = np.arctan2(dy_r, dx_r)

#         D = float(np.hypot(self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]))
#         dist_e = float(np.hypot(dx_w, dy_w))
#         dist_g = self.dist_geodesica(self.p)

#         escalares = np.array([
#             np.clip(dx_r / D, -1, 1),
#             np.clip(dy_r / D, -1, 1),
#             np.clip(dist_e / D, 0, 1),
#             np.clip(dist_g / max(self.geo_max, 1e-6), 0, 1),
#             np.sin(erro), np.cos(erro),
#             np.sin(theta), np.cos(theta),
#             np.clip(self.steps / MAX_STEPS, 0, 1),
#             np.clip(self.n_known / self.n_cells, 0, 1),
#             np.clip(self.get_clearance(self.p) / 2.0, 0, 1),
#             self.last_steering_deg / self.max_steering_deg,
#             float(self.last_direction),
#         ], dtype=np.float32)

#         return np.concatenate([rays, win, escalares]).astype(np.float32)

#     # ==================================================================
#     # RESET / STEP
#     # ==================================================================
#     def reset(self, initial_pose=None):
#         if initial_pose is None:
#             pose = self.getRand()
#         else:
#             pose = np.array(initial_pose, dtype=np.float64, copy=True)
#             if pose.shape != (3,) or not np.all(np.isfinite(pose)):
#                 raise ValueError("initial_pose deve conter [x, y, theta]")
#             pose[2] = self.wrap_angle(pose[2])
#             if self.collision(pose):
#                 raise ValueError("a pose inicial esta em colisao")
#             if np.linalg.norm(pose[:2] - self.alvo) <= self.goal_tol:
#                 raise ValueError("a pose inicial esta na regiao do alvo")

#         self.steps = 0
#         self.sub_steps = 0
#         self.pose = pose.copy()
#         self.p = self.pose[:2].copy()
#         self.traj = [self.p.copy()]

#         self.last_collision = False
#         self.last_steering_deg = 0.0
#         self.prev_steering_deg = 0.0
#         self.last_direction = +1
#         self.info_gain = 0

#         if self.reset_known_map_each_episode:
#             self.known_map = -np.ones_like(self.mapa, dtype=np.int8)
#             self.n_known = 0
#         elif not hasattr(self, 'n_known'):
#             self.n_known = 0

#         self.update_known_map()
#         self.info_gain = 0

#         if self.continuous_obs:
#             return self.get_observation()
#         return self.get_robot_cell()

#     # ------------------------------------------------------------------
#     def step(self, action):
#         """
#         Devolve: obs, reward, terminated, truncated, info
#         (API nova do gymnasium; so 'terminated' deve ir para o replay buffer)
#         """
#         action = int(action)
#         steering_deg, direction = self.actions[action]

#         self.prev_steering_deg = self.last_steering_deg
#         self.last_steering_deg = float(steering_deg)
#         self.last_direction = int(direction)

#         self.steps += 1
#         reward = 0.0
#         terminated = False

#         # ---- action repeat ----
#         for _ in range(self.action_repeat):
#             self.sub_steps += 1
#             next_pose = self.ackermann_step(steering_deg, direction)
#             collided = self.collision(next_pose)
#             self.last_collision = collided

#             if not collided:
#                 self.pose = next_pose
#                 self.p = self.pose[:2].copy()

#             self.traj.append(self.p.copy())
#             self.update_known_map()

#             reward += self.getReward(action)

#             if collided or self.reached_goal():
#                 terminated = True
#                 break

#         truncated = (self.steps >= MAX_STEPS) and not terminated

#         obs = self.get_observation() if self.continuous_obs else self.get_robot_cell()

#         info = {
#             "collision": self.last_collision,
#             "steering_deg": float(steering_deg),
#             "direction": int(direction),
#             "theta": float(self.pose[2]),
#             "truncated": truncated,
#             "success": self.reached_goal(),
#             "dist_geo": self.dist_geodesica(self.p),
#         }
#         return obs, float(reward), terminated, truncated, info

#     # ==================================================================
#     # RECOMPENSA
#     # ==================================================================
#     def getReward(self, action):
#         """
#         ATENCAO AO ORCAMENTO: um episodio tem ate
#         MAX_STEPS * action_repeat = 2000 sub-passos. Todo termo DENSO e
#         positivo acumula 2000x, entao precisa ser <= ~0.01, ou ser uma
#         penalidade pura (que o agente quer zerar, nao farmar).

#         Orcamento alvo:
#             sucesso (caminho de ~16 m)  ->  +30 (alvo) + ~32 (progresso) = ~+60
#             timeout vagando             ->  ~-10 (tempo) + <=+5 (info)   = ~ -5
#             colisao                     ->  -30                          = ~-25
#         """
#         # ---- progresso geodesico (NAO usar a forma (d - gamma*d'):
#         #      o termo (1-gamma)*d vira um bonus por ficar longe do alvo) ----
#         d_ant = self.dist_geodesica(self.traj[-2])
#         d_atual = self.dist_geodesica(self.p)
#         progresso = np.clip((d_ant - d_atual) / self.passo_max, -1.0, 1.0)
#         reward = self.W_PROG * float(progresso)          # ~2.0 por metro

#         # ---- custo de tempo ----
#         reward -= self.W_TEMPO

#         # ---- desalinhamento: PENALIDADE PURA (max 0), nunca um bonus ----
#         dx = self.alvo[0] - self.p[0]
#         dy = self.alvo[1] - self.p[1]
#         erro = self.wrap_angle(np.arctan2(dy, dx) - self.pose[2])
#         reward += self.W_ALINHA * (float(np.cos(erro)) - 1.0)

#         # ---- ganho de informacao: orcamento FIXO para o mapa inteiro ----
#         # explorar 100% do mapa rende no maximo W_INFO no episodio todo
#         reward += self.W_INFO * (self.info_gain / self.n_cells)

#         # ---- margem de seguranca (penalidade pura) ----
#         clr = self.get_clearance(self.p)
#         if clr < 0.6:
#             reward -= self.W_CLEAR * (1.0 - clr / 0.6)

#         # ---- suavidade do esterçamento e leve custo da re ----
#         reward -= self.W_SUAVE * abs(self.last_steering_deg - self.prev_steering_deg) \
#             / self.max_steering_deg
#         if self.last_direction < 0:
#             reward -= self.W_RE

#         # ---- terminais ----
#         if self.last_collision:
#             reward -= 30.0
#         elif self.reached_goal():
#             reward += 30.0
#         # timeout: NENHUMA penalidade -> vai como truncated, com bootstrap

#         return float(reward)

#     # ------------------------------------------------------------------
#     def reached_goal(self):
#         return bool(np.linalg.norm(self.p - self.alvo) <= self.goal_tol)

#     def terminal(self):
#         return self.last_collision or self.reached_goal()

#     # ------------------------------------------------------------------
#     def getRand(self, dist_min=None, dist_max=None, max_tent=10000):
#         """Sorteia uma pose livre. Opcionalmente restrita a uma faixa de
#         distancia GEODESICA do alvo (usado pelo curriculo)."""
#         for _ in range(max_tent):
#             qx = np.random.uniform(self.xlim[0], self.xlim[1])
#             qy = np.random.uniform(self.ylim[0], self.ylim[1])
#             th = np.random.uniform(-np.pi, np.pi)
#             pose = np.array([qx, qy, th], dtype=np.float64)

#             if self.collision(pose):
#                 continue
#             if np.linalg.norm(pose[:2] - self.alvo) <= self.goal_tol:
#                 continue
#             dg = self.dist_geodesica(pose[:2])
#             if dg > self.geo_max + 1e-6:    # regiao nao conectada ao alvo
#                 continue
#             if dist_min is not None and dg < dist_min:
#                 continue
#             if dist_max is not None and dg > dist_max:
#                 continue
#             return pose
#         raise RuntimeError("nao consegui sortear pose inicial valida")

#     def seed(self, rnd_seed=None):
#         np.random.seed(rnd_seed)
#         return [rnd_seed]

#     # ==================================================================
#     # RENDER
#     # ==================================================================
#     def _init_render(self):
#         import pygame
#         pygame.init()
#         self.screen_size = (SCREEN_SIZE, SCREEN_SIZE)
#         self.screen = pygame.display.set_mode(self.screen_size)
#         self.clock = pygame.time.Clock()
#         pygame.display.set_caption("Labirinto")
#         rgb = np.stack([self.mapa] * 3, axis=-1)
#         surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
#         self.map_surface = pygame.transform.scale(surf, self.screen_size)

#     def world_to_screen(self, pos):
#         x = int((pos[0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.screen_size[0])
#         y = int(self.screen_size[1] - (pos[1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.screen_size[1])
#         return (x, y)

#     def render(self):
#         if not self.render_env or self.screen is None:
#             return
#         import pygame
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 self.screen = None
#                 return

#         self.screen.fill((180, 180, 180))
#         self.screen.blit(self.map_surface, (0, 0))

#         ap = self.world_to_screen(self.alvo)
#         t = 6
#         pygame.draw.line(self.screen, (0, 200, 0), (ap[0]-t, ap[1]-t), (ap[0]+t, ap[1]+t), 3)
#         pygame.draw.line(self.screen, (0, 200, 0), (ap[0]-t, ap[1]+t), (ap[0]+t, ap[1]-t), 3)

#         for p in self.traj[::2]:
#             px, py = self.world_to_screen(p)
#             pygame.draw.rect(self.screen, (155, 0, 200), (px, py, 3, 3))

#         hl, hw = self.robot_length/2, self.robot_width/2
#         corners = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])
#         th = self.pose[2]
#         R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
#         pts = [self.world_to_screen(q) for q in (corners @ R.T + self.p)]
#         pygame.draw.polygon(self.screen, (0, 0, 255), pts, 3)

#         front = self.p + 0.4 * np.array([np.cos(th), np.sin(th)])
#         pygame.draw.line(self.screen, (255, 0, 0),
#                          self.world_to_screen(self.p),
#                          self.world_to_screen(front), 3)
#         pygame.display.flip()
#         self.clock.tick(self.render_fps)

#     def get_percentage_explored(self, only_free=True):
#         if only_free:
#             total = np.count_nonzero(self.free)
#             if total == 0:
#                 return 0.0
#             return 100.0 * np.count_nonzero((self.known_map != -1) & self.free) / total
#         return 100.0 * np.count_nonzero(self.known_map != -1) / self.known_map.size

#     def save_known_map_image(self, filename="results/known_map.png"):
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

#         vis = np.copy(self.known_map)
#         for p in self.traj:
#             px, py = self.mts2px(p)
#             vis[int(np.clip(py, 0, self.nrow-1)), int(np.clip(px, 0, self.ncol-1))] = 3
#         px, py = self.mts2px(self.p)
#         vis[int(np.clip(py, 0, self.nrow-1)), int(np.clip(px, 0, self.ncol-1))] = 4
#         px, py = self.mts2px(self.alvo)
#         vis[int(np.clip(py, 0, self.nrow-1)), int(np.clip(px, 0, self.ncol-1))] = 2

#         img = np.zeros((*vis.shape, 3), dtype=np.uint8)
#         img[vis == -1] = [60, 60, 60]
#         img[vis == 0] = [255, 255, 255]
#         img[vis == 1] = [0, 0, 0]
#         img[vis == 2] = [0, 255, 0]
#         img[vis == 3] = [180, 0, 255]
#         img[vis == 4] = [0, 0, 255]

#         plt.figure(figsize=(6, 7))
#         plt.imshow(img)
#         plt.title(f"Mapa de informacao - Exploracao: "
#                   f"{self.get_percentage_explored(True):.1f}%")
#         plt.axis("off")
#         plt.savefig(filename, dpi=150, bbox_inches="tight")
#         plt.close()

#     def close(self):
#         if self.render_env and self.screen is not None:
#             import pygame
#             pygame.quit()
#             self.screen = None