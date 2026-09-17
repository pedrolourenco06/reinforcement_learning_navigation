from pathlib import Path
from itertools import product

import numpy as np
import torch

import ackermann_env as cm
from labirinto_dqn import QNetwork


PASTA = Path(__file__).resolve().parent

# Aponte para o modelo do treinamento com inícios aleatórios.
MODELO = PASTA / "results" / "dqn_best.pt"

# True para visualizar os percursos.
RENDER = True


if __name__ == "__main__":
    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cpu")

    print(f"Modelo: {MODELO}")
    print(f"Dispositivo: {device}")

    checkpoint = torch.load(
        MODELO,
        map_location="cpu",
        weights_only=False,
    )

    rede = QNetwork(
        checkpoint["obs_dim"],
        checkpoint["num_actions"],
    )

    rede.load_state_dict(checkpoint["q_net_state_dict"])
    rede.to(device)
    rede.eval()
    rede.requires_grad_(False)

    env = cm.AckermannEnv(
        img=str(PASTA / "labirinto6.png"),
        xlim=np.array([0.0, 19.2]),
        ylim=np.array([0.0, 24.0]),
        alvo=np.array([13.2, 12.2]),
        render=RENDER,
        continuous_obs=True,
        window_layers=5,
        reset_known_map_each_episode=True,
        wheelbase=0.4,
        robot_length=0.545,
        robot_width=0.415,
        max_steering_deg=20.0,
        speed=0.5,
        dt=0.2,
    )
    rng_avaliacao = np.random.default_rng(2026)

    poses = [(3.6, 20.4, -90)]

    sucessos = 0

    try:
        for teste, (x, y, angulo) in enumerate(poses, start=1):
            inicio = np.array(
                [x, y, np.deg2rad(angulo)],
                dtype=np.float32,
            )

            state = env.reset(initial_pose=inicio)
            menor_distancia = np.linalg.norm(env.p - env.alvo)
            total_reward = 0.0

            if RENDER:
                env.render()

            for passo in range(1, cm.MAX_STEPS + 1):
                # Escolhe sempre a ação de maior valor Q: epsilon zero.
                with torch.no_grad():
                    entrada = torch.as_tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    action = int(rede(entrada).argmax(dim=1).item())

                state, reward, done, info = env.step(action)
                total_reward += reward

                distancia = np.linalg.norm(env.p - env.alvo)
                menor_distancia = min(menor_distancia, distancia)

                if RENDER:
                    env.render()

                if done:
                    break

            sucesso = env.reached_goal()
            sucessos += int(sucesso)

            print(
                f"Teste {teste:02d}/{len(poses)} | "
                f"Início: ({x}, {y}, {angulo}°) | "
                f"Passos: {passo} | "
                f"Sucesso: {sucesso} | "
                f"Colisão: {info['collision']} | "
                f"Menor distância: {menor_distancia:.3f} m | "
                f"Posição final: ({env.p[0]:.2f}, {env.p[1]:.2f}) | "
                f"Reward: {total_reward:.2f}"
            )

        print(
            f"\nSucesso: {sucessos}/{len(poses)} "
            f"({100 * sucessos / len(poses):.2f}%)"
        )

    finally:
        env.close()