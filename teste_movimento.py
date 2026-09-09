from ackermann_env import AckermannEnv

env = AckermannEnv(
    img="labirinto6.png",
    render=True,
    continuous_obs=True,
    wheelbase=0.40,
    robot_length=0.545,
    robot_width=0.415,
    max_steering_deg=20.0,
    speed=0.50,
    dt=0.20
)

# Testa movimento reto e curvas
for action in [2, 4, 0]:

    state = env.reset()

    print(f"\nTestando ação {action}")

    for step in range(100):

        state, reward, done, info = env.step(action)

        env.render()

        print(
            f"Passo: {step} | "
            f"Pose: {env.pose} | "
            f"Colisão: {info['collision']}"
        )

        if done:
            print("Episódio encerrado.")
            break

env.close()