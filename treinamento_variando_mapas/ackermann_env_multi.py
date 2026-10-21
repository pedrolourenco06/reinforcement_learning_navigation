# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    import gym
    from gym import spaces
import numpy as np
import os
from functools import partial
import pygame
from pyparsing import col

# Globais
NACTIONS = 8
MAX_STEPS = 500
SCREEN_SIZE = 500

########################################
# classe do mapa
########################################
class AckermannEnv(gym.Env):
    ########################################
    # construtor
    
    def __init__(
            self,
            xlim=np.array([0.0,16.8]),
            ylim=np.array([0.0,19.2]),
            res=0.4,
            img='labirinto6.png',
            alvo=np.array([5.0, 1.8]),
            render=False,
            continuous_obs=False,
            window_layers=5,
            reset_known_map_each_episode=False,
            wheelbase=0.4,
            robot_length=0.545,
            robot_width=0.415,
            max_steering_deg=20.0,
            speed=0.5,
            dt=0.2,
            allow_reverse=True,
            imgs=None,
            min_region_frac=0.05
    ):
        # salva o tamanho geometrico da imagem em metros
        self.xlim = xlim
        self.ylim = ylim

        # resolucao
        self.res = res

        #parâmetros fisicos do veículo
        self.wheelbase = wheelbase
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.max_steering_deg = max_steering_deg
        self.allow_reverse = bool(allow_reverse)
        self.speed = speed
        self.dt = dt

        #descricao das acoes discretas de esterçamento
        self.steering_actions_deg = np.array([
            -self.max_steering_deg,
            -self.max_steering_deg/2,
            0,
            self.max_steering_deg/2,
            self.max_steering_deg
        ], dtype=np.float32)

        # Os índices 0 a 4 continuam com os mesmos movimentos para frente.
        self.action_directions = np.ones(5, dtype=np.float32)
        if self.allow_reverse:
            self.steering_actions_deg = np.concatenate((
                self.steering_actions_deg,
                np.array([-self.max_steering_deg, 0.0, self.max_steering_deg], dtype=np.float32),
            ))
            self.action_directions = np.concatenate((
                self.action_directions, -np.ones(3, dtype=np.float32),
            ))

        # modo de observacao:
        # False -> estado discreto para Q-learning/SARSA tabular
        # True  -> vetor continuo para DQN
        self.continuous_obs = continuous_obs

        # tamanho da janela local usada na observacao continua
        # window_layers = 5 -> janela 11x11 ao redor do robo
        self.window_layers = window_layers

        # se True, reinicia o mapa conhecido a cada episodio
        self.reset_known_map_each_episode = reset_known_map_each_episode

        ns = int(np.max([np.abs(np.diff(self.xlim)), np.abs(np.diff(self.ylim))])/res)
        self.num_states = [ns, ns]
        
        # espaco de atuacao
        self.action_space = spaces.Discrete(len(self.steering_actions_deg))

        # espaco de observacao para DQN
        # janela local + 9 variaveis continuas:
        # x_norm, y_norm, dx_goal, dy_goal, dist_goal, steps_norm, info_norm
        obs_dim = 2* (2 * self.window_layers + 1) ** 2 + 9

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # converte estados continuos em discretos
        lower_bounds = [self.xlim[0], self.ylim[0]]
        upper_bounds = [self.xlim[1], self.ylim[1]]
        self.get_state = partial(self.obs_to_state, self.num_states, lower_bounds, upper_bounds)

        # alvo
        self.alvo = alvo
        
        # renderizar
        self.render_env = render
        
        # cria mapa(s): todos os mapas candidatos sao lidos UMA vez e guardados
        # em self.mapas; depois set_map() apenas troca entre eles (sem ler disco).
        # Os mapas devem cobrir a mesma area xlim x ylim (resolucao de imagem pode variar).
        self.min_region_frac = float(min_region_frac)
        lista_imgs = [img] if imgs is None else list(imgs)
        if img not in lista_imgs:
            lista_imgs.insert(0, img)
        self.mapas = {}
        pygame.init()
        pygame.display.set_mode((1, 1))  # Inicializa com uma janelinha mínima
        for caminho in lista_imgs:
            chave = self._chave_mapa(caminho)
            if chave in self.mapas:
                raise ValueError(f"dois mapas com o mesmo nome de arquivo: {chave}")
            self.mapas[chave] = self._ler_imagem(caminho)
        pygame.quit()
        self.mapa_atual = None
        self._instalar_mapa(self._chave_mapa(img))

        #cria pontos internos no retangulo do robo para detectar colisao    
        self.build_footprint_samples(spacing=0.04)

        #inicia mapa informacao
        self.known_map = -np.ones_like(self.mapa, dtype=np.int8)

        # Inicializa pygame se necessário
        if self.render_env:
            pygame.init()
            self.screen_size = (SCREEN_SIZE, SCREEN_SIZE)
            self.screen = pygame.display.set_mode(self.screen_size)
            self.clock = pygame.time.Clock()
            pygame.display.set_caption("Labirinto")
        
            # Converte o mapa para uma imagem pygame
            self._atualizar_superficie_mapa()

    ########################################
    # ambientes em 2D
    def _chave_mapa(self, caminho):
        # o nome do arquivo identifica o mapa (precisa ser unico entre os mapas)
        return os.path.basename(str(caminho))

    def _ler_imagem(self, image):
        # Mesma leitura/binarizacao do init2D original, mas devolve os dados
        # em vez de gravar em self (para poder guardar varios mapas).
        # Carrega a imagem em escala de cinza
        I_surface = pygame.image.load(image).convert()
        I_array = pygame.surfarray.pixels3d(I_surface)

        # Transpõe para (altura, largura, canais), como no OpenCV
        I_array = I_array.transpose(1, 0, 2)

        # Converte para escala de cinza (média dos canais RGB)
        I_gray = np.mean(I_array, axis=2).astype(np.uint8)

        # Pega o número de linhas e colunas
        nrow = I_gray.shape[0]
        ncol = I_gray.shape[1]

        # Binariza a imagem (limiar 127)
        I_binary = np.where(I_gray > 127, 255, 0).astype(np.uint8)
        mapa = I_binary.copy()

        # Parâmetros de conversão (como no original)
        mx = float(ncol) / float(self.xlim[1] - self.xlim[0])
        my = float(nrow) / float(self.ylim[1] - self.ylim[0])

        rotulos, tamanhos = self._calcular_regioes(mapa, mx, my)
        return dict(mapa=mapa, nrow=nrow, ncol=ncol, mx=mx, my=my,
                    rotulos=rotulos, tamanhos=tamanhos)

    def _instalar_mapa(self, chave):
        d = self.mapas[chave]
        self.mapa = d['mapa']
        self.nrow = d['nrow']
        self.ncol = d['ncol']
        self.mx = d['mx']
        self.my = d['my']
        self.rotulos = d['rotulos']
        self.tamanhos_regioes = d['tamanhos']
        self.mapa_atual = chave
        if self.render_env and hasattr(self, 'screen'):
            self._atualizar_superficie_mapa()

    def _atualizar_superficie_mapa(self):
        mapa_norm = self.mapa.astype(np.uint8)  # inverte e escala pra 0-255
        mapa_rgb = np.stack([mapa_norm]*3, axis=-1)  # gray -> RGB
        mapa_surface = pygame.surfarray.make_surface(np.transpose(mapa_rgb, (1, 0, 2)))
        self.map_surface = pygame.transform.scale(mapa_surface, self.screen_size)

    ########################################
    # regioes livres conectadas (usadas so para validar alvo/pose sorteados)
    def _calcular_regioes(self, mapa, mx, my):
        # Erode o espaco livre pelo raio inscrito do robo (0.9 * largura/2): todo
        # centro de pose sem colisao fica fora de obstaculos por pelo menos esse
        # raio, entao a erosao nunca descarta um centro realmente viavel.
        # Componentes conexas (8 vizinhos) dessa mascara = regioes onde o robo
        # pode estar. Condicao NECESSARIA (nao suficiente) para existir caminho,
        # pois nao considera o raio de curva do Ackermann.
        livre = mapa >= 127
        raio = max(1, int(np.floor(0.9 * (self.robot_width / 2.0) * min(mx, my))))
        h, w = livre.shape
        pad = np.pad(livre, raio, constant_values=False)
        erodido = np.ones_like(livre)
        for dy in range(-raio, raio + 1):
            for dx in range(-raio, raio + 1):
                if dx * dx + dy * dy <= raio * raio:
                    erodido &= pad[raio + dy: raio + dy + h, raio + dx: raio + dx + w]
        try:
            from scipy import ndimage
            rotulos, n = ndimage.label(erodido, structure=np.ones((3, 3), dtype=int))
        except ImportError:
            rotulos, n = self._rotular_bfs(erodido)
        rotulos = rotulos.astype(np.int32)
        tamanhos = np.bincount(rotulos.ravel(), minlength=n + 1)
        tamanhos[0] = 0
        return rotulos, tamanhos

    @staticmethod
    def _rotular_bfs(mascara):
        from collections import deque
        h, w = mascara.shape
        rotulos = np.zeros((h, w), dtype=np.int32)
        n = 0
        for i0, j0 in zip(*np.nonzero(mascara)):
            if rotulos[i0, j0]:
                continue
            n += 1
            rotulos[i0, j0] = n
            fila = deque([(i0, j0)])
            while fila:
                i, j = fila.popleft()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        a, b = i + di, j + dj
                        if 0 <= a < h and 0 <= b < w and mascara[a, b] and not rotulos[a, b]:
                            rotulos[a, b] = n
                            fila.append((a, b))
        return rotulos, n

    def _rotulo_regiao(self, q):
        px, py = self.mts2px(q)
        lin = int(py)
        col = int(px)
        if lin < 0 or lin >= self.nrow or col < 0 or col >= self.ncol:
            return 0
        return int(self.rotulos[lin, col])

    def mesma_regiao(self, p, q):
        # True se os dois pontos estao na mesma regiao livre conexa do mapa atual
        rp = self._rotulo_regiao(p)
        return rp > 0 and rp == self._rotulo_regiao(q)

    def alvo_valido(self, alvo):
        alvo = np.asarray(alvo, dtype=np.float64)
        if alvo.shape != (2,) or not np.all(np.isfinite(alvo)):
            return False
        if not (self.xlim[0] < alvo[0] < self.xlim[1] and self.ylim[0] < alvo[1] < self.ylim[1]):
            return False
        rot = self._rotulo_regiao(alvo)
        if rot == 0:
            return False
        # ignora bolsoes isolados minusculos
        if self.tamanhos_regioes[rot] < self.min_region_frac * self.tamanhos_regioes.sum():
            return False
        # o robo precisa caber no alvo em pelo menos uma de 8 orientacoes
        for theta in np.linspace(0.0, np.pi, 8, endpoint=False):
            if not self.collision(np.array([alvo[0], alvo[1], theta])):
                return True
        return False

    ########################################
    # troca de mapa / alvo
    def set_map(self, img):
        chave = self._chave_mapa(img)
        if chave not in self.mapas:
            raise KeyError(f"mapa '{img}' nao foi carregado no construtor (imgs=...). "
                           f"Disponiveis: {sorted(self.mapas)}")
        if chave != self.mapa_atual:
            self._instalar_mapa(chave)
        # o mapa conhecido tem o formato do mapa atual
        self.known_map = -np.ones_like(self.mapa, dtype=np.int8)

    def set_alvo(self, alvo):
        alvo = np.asarray(alvo, dtype=np.float64).copy()
        if not self.alvo_valido(alvo):
            raise ValueError(f"alvo {alvo} invalido no mapa '{self.mapa_atual}'")
        self.alvo = alvo
        # o alvo faz parte do mapa conhecido
        self.known_map = -np.ones_like(self.mapa, dtype=np.int8)

    def build_footprint_samples(self, spacing=0.04):
        xs = np.arange(
            -self.robot_length/2,
            self.robot_length/2 + spacing,
            spacing
        )

        ys = np.arange(
            -self.robot_width/2,
            self.robot_width/2 + spacing,
            spacing
        )

        xx, yy = np.meshgrid(xs, ys)

        self.footprint_local = np.column_stack([
            xx.ravel(),
            yy.ravel()
        ])

    def wrap_angle(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi
    
    ########################################
    # seed
    ########################################
    def seed(self, rnd_seed = None):
        np.random.seed(rnd_seed)
        return [rnd_seed]

    ########################################
    # reset
    ########################################
    def reset(self, initial_pose = None, alvo = None, img = None):

        # troca opcional de mapa e/ou alvo ANTES de validar a pose inicial
        if img is not None:
            self.set_map(img)
        if alvo is not None:
            self.set_alvo(alvo)
        if (img is not None or alvo is not None) and not self.alvo_valido(self.alvo):
            raise ValueError(f"o alvo atual nao e valido no mapa '{self.mapa_atual}'")

        if initial_pose is None:
            pose = self.getRand()
        else:
            pose = np.array(initial_pose, dtype=np.float32, copy=True)

            if pose.shape != (3,) or not np.all(np.isfinite(pose)):
                raise ValueError("initia_pose deve conter [x, y, theta]")

            pose[2] = self.wrap_angle(pose[2])

            if self.collision(pose):
                raise ValueError("a pose inicial esta em colisão")
        if np.linalg.norm(pose[:2] - self.alvo) <= 0.3:
            raise ValueError("a pose inicial esta na regiao do alvo")
        self.steps = 0
        self.pose = pose.copy()
        self.p = self.pose[:2].copy()
        self.traj = [self.p.copy()]

        self.last_collision = False
        self.last_steering_deg = 0.0

        if self.reset_known_map_each_episode:
            self.known_map = -np.ones_like(self.mapa, dtype=np.int8)

        self.update_known_map(layers=6)

        self.info_gain = 0

        if self.continuous_obs:
            return self.get_observation()

        return self.get_state(self.p)

    def action_to_steering(self, action):
        action = int(action)

        return float(self.steering_actions_deg[action])

    def ackermann_step(self, steering_deg, direction=1.0, pose=None):
        # Velocidade assinada também inverte a evolução da orientação na ré.
        x, y, theta = self.pose if pose is None else pose
        velocidade = self.speed * direction

        delta = np.deg2rad(steering_deg)

        lr = self.wheelbase / 2.0

        beta = np.arctan(lr/ self.wheelbase * np.tan(delta))

        x_new = x + velocidade * np.cos(theta + beta) * self.dt
        y_new = y + velocidade * np.sin(theta + beta) * self.dt
        theta_new = (theta + (velocidade/lr) * np.sin(beta) * self.dt)
        theta_new = self.wrap_angle(theta_new)
        return np.array([x_new, y_new, theta_new], dtype=np.float32)

    # step -> new_observation, reward, done, info = env.step(action)
    def step(self, action):

        action = int(action)

        if not 0 <= action < self.action_space.n:
            raise ValueError(f"Ação inválida: {action}")

        self.steps += 1

        steering_deg = self.action_to_steering(action)

        self.last_steering_deg = steering_deg

        direction = float(self.action_directions[action])
        next_pose = self.ackermann_step(steering_deg, direction)

        collided = self.collision(next_pose)
        self.last_collision = collided

        if not collided:
            self.pose = next_pose
            self.p = self.pose[:2].copy()

        self.traj.append(self.p.copy())

        self.update_known_map(layers=6)

        reward = self.getReward(action)

        done = self.terminal()

        if self.continuous_obs:
            obs = self.get_observation()
        else:
            obs = self.get_state(self.p)

        info = {"collision": collided,
                "steering_deg": steering_deg,
                "direction": direction,
                "theta": float(self.pose[2])}

        return obs, reward, done, info

    ########################################
    # função de reforço
    def getReward(self, action):
        reward = -0.1

        # Custo adicional por usar ré.
        if self.action_directions[int(action)] < 0:
            reward -= 0.02

        distancia_anterior = np.linalg.norm(self.traj[-2] - self.alvo)

        distancia_atual = np.linalg.norm(self.p - self.alvo)

        progresso = distancia_anterior - distancia_atual
        reward += 2*progresso

        # bonus_exploracao = min (
        #     0.002 * max(self.info_gain, 0), 0.2
        # )

        # reward += bonus_exploracao

        if self.last_collision:
            reward -= 50.0
        elif self.reached_goal():
            reward += 100.0
        elif self.steps >= MAX_STEPS:
            reward -= 20.0
        return float(reward)
    
    ########################################
    # terminou?
    def terminal(self):
        if self.last_collision:
            return True
        if self.reached_goal():
            return True
        if self.steps >= MAX_STEPS:
            return True
        return False
    
    def reached_goal(self):
        return np.linalg.norm(self.p - self.alvo) <= 0.30

    ########################################
    # pega ponto aleatorio 
    def getRand(self):
        while True:
            qx = np.random.uniform(
                self.xlim[0],
                self.xlim[1]
            )

            qy = np.random.uniform(
                self.ylim[0],
                self.ylim[1]
            )

            theta = np.random.uniform(
                -np.pi,
                np.pi
            )

            pose = np.array([
                qx,
                qy,
                theta
            ], dtype=np.float32)

            if not self.collision(pose):
                break

        return pose
    ########################################
    # verifica colisao com os obstaculos
    def collision(self, q):

        q = np.asarray(q, dtype=np.float32)

        if q.size == 2:
            pose = np.array([q[0], q[1], 0.0])
        else:
            pose = q

        x, y, theta = pose

        c = np.cos(theta)
        s = np.sin(theta)

        #pontos do robo no referencial local
        lx = self.footprint_local[:, 0]
        ly = self.footprint_local[:, 1]

        #rotaciona e translada para o mundo
        wx = x + c * lx - s * ly
        wy = y + s * lx + c * ly

        #verifica os limites do ambiente
        if np.any(wx <= self.xlim[0]):
            return True
        if np.any(wx >= self.xlim[1]):
            return True
        if np.any(wy <= self.ylim[0]):
            return True
        if np.any(wy >= self.ylim[1]):
            return True

        #verifica cada ponto q compoe o robo
        for px_world, py_world in zip(wx, wy):
            px, py = self.mts2px([px_world, py_world])

            col = int(px)
            lin = int(py)

            if(lin < 0 or lin >= self.nrow or col < 0 or col >= self.ncol):
                return True

            if self.mapa[lin, col] < 127:
                return True

        return False
            
        

       


    ########################################
    # transforma pontos no mundo real para pixels na imagem
    def mts2px(self, q):
        qx, qy = q
        # conversao
        px = (qx - self.xlim[0])*self.mx
        py = self.nrow - (qy - self.ylim[0])*self.my

        return px, py

    ##########################################
    # converte estados continuos em discretos
    def obs_to_state(self, num_states, lower_bounds, upper_bounds, obs):
        state_idx = []
        for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
            state_idx.append(self.discretize_val(ob, lower, upper, num))

        return np.ravel_multi_index(state_idx, num_states)

    ##########################################
    # discretiza um valor
    def discretize_val(self, val, min_val, max_val, num_states):
        state = int(num_states * (val - min_val) / (max_val - min_val))
        if state >= num_states:
            state = num_states - 1
        if state < 0:
            state = 0
        return state
    
    ########################################
    # Mapeamento de coordenadas reais para pixels
    def world_to_screen(self, pos):
        x = int((pos[0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.screen_size[0])
        y = int(self.screen_size[1] - (pos[1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.screen_size[1])
        return (x, y)
    
    
    def get_robot_cell(self):
        px, py = self.mts2px(self.p)
        lin = max(0, min(int(py), self.nrow - 1))
        col = max(0, min(int(px), self.ncol - 1))

        return lin, col
        
    
    def update_known_map(self, layers=2):

        lin, col = self.get_robot_cell()

        radius_m = layers * self.res

        radius_px_x = int(np.ceil(radius_m * self.mx))
        radius_px_y = int(np.ceil(radius_m * self.my))

        # janela ao redor do robô, recortada nos limites do mapa
        i0 = max(0, lin - radius_px_y)
        i1 = min(self.nrow, lin + radius_px_y + 1)
        j0 = max(0, col - radius_px_x)
        j1 = min(self.ncol, col + radius_px_x + 1)

        # ganho de informação: células da janela que ainda eram desconhecidas
        self.info_gain = int(np.count_nonzero(self.known_map[i0:i1, j0:j1] == -1))

        # 1 = obstáculo, 0 = livre (mesmo critério do original: mapa < 127)
        janela = (self.mapa[i0:i1, j0:j1] < 127).astype(np.int8)

        # 2 = alvo (tem prioridade sobre obstáculo/livre, como no original)
        alvo_px, alvo_py = self.mts2px(self.alvo)
        alvo_lin = max(0, min(int(alvo_py), self.nrow - 1))
        alvo_col = max(0, min(int(alvo_px), self.ncol - 1))
        if i0 <= alvo_lin < i1 and j0 <= alvo_col < j1:
            janela[alvo_lin - i0, alvo_col - j0] = 2

        self.known_map[i0:i1, j0:j1] = janela

    def get_known_value_at_world(self, q):

        px, py = self.mts2px(q)

        lin = int(py)
        col = int(px)

        if lin < 0 or lin >= self.nrow or col < 0 or col >= self.ncol:
            return 1.0  # fora do mapa tratado como obstaculo

        value = self.known_map[lin, col]

        if value == -1:
            return -1.0  # desconhecido
        if value == 0:
            return 0.0   # livre
        if value == 1:
            return 1.0   # obstaculo
        if value == 2:
            return 0.5   # alvo

        return -1.0
    
    # def get_observation(self):

    #     obs = []

    #     x = self.pose[0]
    #     y = self.pose[1]
    #     theta = self.pose[2]

    #     for dy in range(self.window_layers, -self.window_layers - 1, -1):
    #         for dx in range(-self.window_layers, self.window_layers + 1):
    #             q = np.array([x + dx * self.res, y + dy * self.res])
    #             obs.append(self.get_known_value_at_world(q))

    #     x_norm = (2 * (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) - 1)

    #     y_norm = (2 * (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) - 1)

    #     dx_goal = (self.alvo[0] - x) / (self.xlim[1] - self.xlim[0])

    #     dy_goal = (self.alvo[1] - y) / (self.ylim[1] - self.ylim[0])

    #     max_dist = np.linalg.norm([self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0]])

    #     dist_goal = (np.linalg.norm(self.alvo - self.p) / max_dist)

    #     steps_norm = np.clip(self.steps / MAX_STEPS, 0, 1)

    #     info_norm = np.clip(self.info_gain / self.known_map.size, 0, 1)

    #     obs.extend([x_norm, y_norm, np.sin(theta), np.cos(theta), dx_goal, dy_goal, dist_goal, steps_norm, info_norm])

    #     return np.array(obs, dtype=np.float32)
    def get_observation(self):
        ocupacao = []
        conhecido = []

        x, y, theta = self.pose

        for dy in range(self.window_layers, -self.window_layers - 1, -1):
            for dx in range(-self.window_layers, self.window_layers + 1):
                q = np.array([
                    x + dx * self.res,
                    y + dy * self.res,
                ])

                valor = self.get_known_value_at_world(q)

                ocupacao.append(1.0 if valor == 1.0 else 0.0)
                conhecido.append(0.0 if valor == -1.0 else 1.0)

        largura = self.xlim[1] - self.xlim[0]
        altura = self.ylim[1] - self.ylim[0]

        x_norm = 2.0 * (x - self.xlim[0]) / largura - 1.0
        y_norm = 2.0 * (y - self.ylim[0]) / altura - 1.0

        dx_goal = (self.alvo[0] - x) / largura
        dy_goal = (self.alvo[1] - y) / altura

        max_dist = np.hypot(largura, altura)
        dist_goal = np.linalg.norm(self.alvo - self.p) / max_dist

        steps_norm = np.clip(self.steps / MAX_STEPS, 0.0, 1.0)
        info_norm = np.clip(
            self.info_gain / self.known_map.size,
            0.0,
            1.0,
        )

        variaveis = [
            x_norm,
            y_norm,
            np.sin(theta),
            np.cos(theta),
            dx_goal,
            dy_goal,
            dist_goal,
            steps_norm,
            info_norm,
        ]

        return np.asarray(
            ocupacao + conhecido + variaveis,
            dtype=np.float32,
        )

    def render(self, Q=None, arrow_size=0.5, target_size=5, robot_size=10):

        if not self.render_env:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((180, 180, 180))

        self.screen.blit(self.map_surface, (0, 0))

        # alvo no mapa real
        alvo_pos = self.world_to_screen(self.alvo)
        pygame.draw.line(self.screen, (0, 200, 0),
                        (alvo_pos[0] - target_size, alvo_pos[1] - target_size),
                        (alvo_pos[0] + target_size, alvo_pos[1] + target_size), 3)
        pygame.draw.line(self.screen, (0, 200, 0),
                        (alvo_pos[0] - target_size, alvo_pos[1] + target_size),
                        (alvo_pos[0] + target_size, alvo_pos[1] - target_size), 3)

        # trajetória no mapa real
        for p in self.traj:
            px, py = self.world_to_screen(p)
            pygame.draw.rect(self.screen, (155, 0, 200), (px, py, 4, 4))

        half_l = self.robot_length / 2
        half_w = self.robot_width / 2

        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w]
        ])

        theta = self.pose[2]

        rotation = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        corners_to_world = (corners @ rotation.T + self.p)
        corners_to_screen = [self.world_to_screen(p) for p in corners_to_world]

        pygame.draw.polygon(self.screen, (0, 0 , 255), corners_to_screen, 3)

        front = np.array([self.p[0] + 0.4 * np.cos(theta), self.p[1] + 0.4 * np.sin(theta)])

        pygame.draw.line(self.screen, (255, 0, 0), self.world_to_screen(self.p), self.world_to_screen(front), 3)

        pygame.display.flip()
        self.clock.tick(120)


    def render_known_map(self):
        import matplotlib.pyplot as plt
        import numpy as np

        img = np.copy(self.known_map)

        vis = np.copy(img)

        for p in self.traj:
            px, py = self.mts2px(p)
            lin = max(0, min(int(py), self.nrow - 1))
            col = max(0, min(int(px), self.ncol - 1))
            vis[lin, col] = 3

        px, py = self.mts2px(self.p)
        lin = max(0, min(int(py), self.nrow - 1))
        col = max(0, min(int(px), self.ncol - 1))
        vis[lin, col] = 4

        px, py = self.mts2px(self.alvo)
        lin = max(0, min(int(py), self.nrow - 1))
        col = max(0, min(int(px), self.ncol - 1))
        vis[lin, col] = 2

        cmap_data = np.zeros((vis.shape[0], vis.shape[1], 3), dtype=np.uint8)
        cmap_data[vis == -1] = [60, 60, 60]       # desconhecido
        cmap_data[vis == 0]  = [255, 255, 255]    # livre
        cmap_data[vis == 1]  = [0, 0, 0]          # obstáculo
        cmap_data[vis == 2]  = [0, 255, 0]        # alvo            cmap_data[vis == 3]  = [180, 0, 255]      # trajetória
        cmap_data[vis == 4]  = [0, 0, 255]    

        plt.figure(2)
        plt.clf()
        plt.imshow(cmap_data)
        plt.title("Mapa de informação")
        plt.axis("off")
        plt.pause(0.001)


     
    ########################################
    def draw_arrow(self, surface, color, start, end, width=2, head_size=3):
        # Linha principal
        pygame.draw.line(surface, color, start, end, width)

        # Vetor da seta
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.arctan2(dy, dx)

        # Cálculo da cabeça da seta (duas linhas formando um "V")
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)

        left = (
            end[0] - head_size * cos_a + head_size * sin_a,
            end[1] - head_size * sin_a - head_size * cos_a
        )
        right = (
            end[0] - head_size * cos_a - head_size * sin_a,
            end[1] - head_size * sin_a + head_size * cos_a
        )

        pygame.draw.line(surface, color, end, left, width)
        pygame.draw.line(surface, color, end, right, width)

    def get_percentage_explored(self, only_free=True):
        if only_free:
            # células livres no mapa real
            free_cells = self.mapa >= 127

            # células livres que já foram conhecidas pelo robô
            known_free_cells = (self.known_map != -1) & free_cells

            total_free = np.sum(free_cells)

            if total_free == 0:
                return 0.0

            return 100.0 * np.sum(known_free_cells) / total_free

        else:
            # considera o mapa inteiro, incluindo obstáculos
            known_cells = np.sum(self.known_map != -1)
            total_cells = self.known_map.size

            return 100.0 * known_cells / total_cells    

    def save_known_map_image(self, filename="results/known_map.png"):
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        vis = np.copy(self.known_map)

        for p in self.traj:
            px, py = self.mts2px(p)
            lin = max(0, min(int(py), self.nrow - 1))
            col = max(0, min(int(px), self.ncol - 1))
            vis[lin, col] = 3

        px, py = self.mts2px(self.p)
        lin = max(0, min(int(py), self.nrow - 1))
        col = max(0, min(int(px), self.ncol - 1))
        vis[lin, col] = 4

        px, py = self.mts2px(self.alvo)
        lin = max(0, min(int(py), self.nrow - 1))
        col = max(0, min(int(px), self.ncol - 1))
        vis[lin, col] = 2

        img = np.zeros((vis.shape[0], vis.shape[1], 3), dtype=np.uint8)
        img[vis == -1] = [60, 60, 60]       # desconhecido
        img[vis == 0]  = [255, 255, 255]    # livre
        img[vis == 1]  = [0, 0, 0]          # obstáculo
        img[vis == 2]  = [0, 255, 0]        # alvo
        img[vis == 3]  = [180, 0, 255]      # trajetória
        img[vis == 4]  = [0, 0, 255]        # robô

        plt.figure()
        plt.imshow(img)
        explored = self.get_percentage_explored(only_free=True)
        plt.title(f"Mapa de informação - Exploração: {explored:.2f}%")
        plt.axis("off")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

    def render_map_with_target(self, filename=None):
        import matplotlib.pyplot as plt

        # mostra o mapa binário
        plt.figure(figsize=(6, 6))
        plt.imshow(self.mapa, cmap='gray', origin='lower')

        # converte alvo do mundo para pixel
        px, py = self.mts2px(self.alvo)

        # desenha o alvo
        plt.scatter(px, py, c='red', s=120, marker='x', linewidths=3, label='Alvo')

        plt.title("Mapa com alvo destacado")
        plt.legend()
        plt.axis("off")

        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')

        plt.show()

    ########################################

    def close(self):
        if self.render_env:
            pygame.quit()

    def __del__(self):
        self.close()