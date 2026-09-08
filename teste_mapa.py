import numpy as np
import maze_information_map.ackermann_env as cm

env = cm.Maze(
    render=False,
    continuous_obs=True,
    img='labirinto2.png',
    alvo=np.array([5.0, 1.8])
)

env.render_map_with_target("results/mapa_com_alvo.png")