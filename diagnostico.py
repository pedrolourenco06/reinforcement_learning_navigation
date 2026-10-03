# -*- coding: utf-8 -*-
"""
Rode ISTO antes de qualquer treino. Ele responde, em ~1 minuto:

  1. O sensor enxerga mais longe que o raio de giro?  (se nao, a tarefa e
     impossivel: o robo descobre a parede depois do ponto de nao-retorno)
  2. Que fracao do mapa livre tem espaco para o robo manobrar?
  3. Um oraculo guloso sobre o campo geodesico resolve o ambiente?
     (se o ORACULO nao resolve, nenhum DQN vai resolver)
  4. A ordem das recompensas esta certa?   goal > timeout > colisao
  5. Quantas decisoes por segundo o ambiente entrega?

Uso:  python diagnostico.py
"""
from collections import defaultdict
import time

import numpy as np

import ackermann_env2 as cm


ALVO = np.array([13.2, 12.2])
IMG = "labirinto6.png"


# ----------------------------------------------------------------------
def oraculo_guloso(env):
    """Melhor acao por lookahead de uma acao completa, descendo o campo geodesico."""
    melhor, melhor_a = -1e9, 0
    pose0, p0 = env.pose.copy(), env.p.copy()
    for a in range(env.action_space.n):
        sd, d = env.actions[a]
        env.pose = pose0.copy()
        ok = True
        for _ in range(env.action_repeat):
            q = env.ackermann_step(sd, d)
            if env.collision(q):
                ok = False
                break
            env.pose = q
        if not ok:
            continue
        sc = -env.dist_geodesica(env.pose[:2]) - (0.5 if d < 0 else 0.0)
        if sc > melhor:
            melhor, melhor_a = sc, a
    env.pose, env.p = pose0, p0
    return melhor_a


def politica_evasiva(env):
    """Nunca bate, mas nao busca o alvo -> gera episodios de timeout."""
    pose0, p0 = env.pose.copy(), env.p.copy()
    cand = []
    for a in range(env.action_space.n):
        sd, d = env.actions[a]
        env.pose = pose0.copy()
        ok = True
        for _ in range(env.action_repeat * 3):
            q = env.ackermann_step(sd, d)
            if env.collision(q):
                ok = False
                break
            env.pose = q
        if ok:
            cand.append(a)
    env.pose, env.p = pose0, p0
    return int(np.random.choice(cand)) if cand else 2


def politica_suicida(env):
    return 2                      # sempre em frente


# ----------------------------------------------------------------------
def rodar(env, politica, n, dist_max, seed):
    np.random.seed(seed)
    out = defaultdict(list)
    passos = defaultdict(list)
    for _ in range(n):
        env.reset(initial_pose=env.getRand(dist_max=dist_max))
        R = 0.0
        for k in range(cm.MAX_STEPS):
            o, r, term, trunc, info = env.step(politica(env))
            R += r
            if term or trunc:
                break
        fim = ("goal" if env.reached_goal()
               else ("colisao" if info["collision"] else "timeout"))
        out[fim].append(R)
        passos[fim].append(k + 1)
    return out, passos


# ----------------------------------------------------------------------
def main():
    print("=" * 72)
    print("1) GEOMETRIA")
    print("=" * 72)
    env = cm.AckermannEnv(img=IMG, alvo=ALVO, render=False)

    rg = env.raio_giro_min
    rs = env.sensor_radius_m
    print(f"  raio de giro minimo      : {rg:.2f} m")
    print(f"  raio do sensor           : {rs:.2f} m")
    print(f"  passo por sub-passo      : {env.passo_max:.2f} m")
    print(f"  action_repeat            : {env.action_repeat} "
          f"(-> {env.action_repeat*env.passo_max:.2f} m por decisao)")
    print(f"  acoes                    : {env.action_space.n} "
          f"({'com re' if env.allow_reverse else 'so frente'})")
    print(f"  obs_dim                  : {env.obs_dim}")
    if rs < 1.2 * rg:
        print("  >> FALHA: sensor curto demais. Desvio e impossivel. "
              "Aumente sensor_layers.")
    else:
        print(f"  >> OK: sensor / raio de giro = {rs/rg:.2f}x")

    print()
    print("=" * 72)
    print("2) ESPACO LIVRE")
    print("=" * 72)
    clr = env.clearance[env.free]
    meia_diag = np.hypot(env.robot_length, env.robot_width) / 2.0
    for q in [10, 25, 50, 75, 90]:
        print(f"  clearance p{q:<2d}            : {np.percentile(clr, q):.2f} m")
    print(f"  meia-diagonal do robo    : {meia_diag:.2f} m")
    print(f"  celulas livres onde o robo cabe em qualquer angulo : "
          f"{100*(clr > meia_diag).mean():.1f}%")
    print(f"  celulas livres com espaco para girar (> raio giro) : "
          f"{100*(clr > rg).mean():.1f}%")
    print(f"  distancia geodesica maxima ao alvo                 : "
          f"{env.geo_max:.1f} m")

    print()
    print("=" * 72)
    print("3) ORACULO  (se isto falhar, o DQN nao tem chance)")
    print("=" * 72)
    for dmax in [5.0, 12.0, env.geo_max]:
        out, pas = rodar(env, oraculo_guloso, 25, dmax, seed=0)
        ng = len(out["goal"])
        print(f"  dist_max={dmax:5.1f} m | sucesso {ng:2d}/25 | "
              f"colisao {len(out['colisao']):2d} | timeout {len(out['timeout']):2d} | "
              f"decisoes medias {np.mean(pas['goal']) if ng else -1:5.1f}")
    print("  >> esperado: >=20/25 de sucesso e ~0 colisoes")

    print()
    print("=" * 72)
    print("4) ORDEM DAS RECOMPENSAS")
    print("=" * 72)
    tudo = defaultdict(list)
    for pol, n, seed in [(oraculo_guloso, 25, 0),
                         (politica_evasiva, 25, 2),
                         (politica_suicida, 20, 3)]:
        out, _ = rodar(env, pol, n, 12.0, seed)
        for k, v in out.items():
            tudo[k].extend(v)
    print(f"  {'fim':<10s} {'n':>4s} {'R medio':>9s} {'R min':>9s} {'R max':>9s}")
    for k in ["goal", "timeout", "colisao"]:
        if tudo[k]:
            print(f"  {k:<10s} {len(tudo[k]):4d} {np.mean(tudo[k]):9.1f} "
                  f"{min(tudo[k]):9.1f} {max(tudo[k]):9.1f}")
    m = {k: np.mean(v) for k, v in tudo.items() if v}
    if "goal" in m and "timeout" in m and "colisao" in m:
        ok = m["goal"] > m["timeout"] > m["colisao"]
        print(f"  >> {'OK' if ok else 'FALHA'}: "
              f"goal > timeout > colisao  {'(ordem correta)' if ok else '(ORDEM ERRADA!)'}")

    print()
    print("=" * 72)
    print("5) DESEMPENHO")
    print("=" * 72)
    np.random.seed(0)
    t0 = time.time()
    n = 0
    for _ in range(15):
        env.reset(initial_pose=env.getRand())
        for k in range(cm.MAX_STEPS):
            o, r, t, tr, i = env.step(np.random.randint(env.action_space.n))
            n += 1
            if t or tr:
                break
    dt = time.time() - t0
    print(f"  {n/dt:.0f} decisoes/s  ({n*env.action_repeat/dt:.0f} sub-passos/s), "
          f"politica aleatoria, sem render")
    print(f"  600k decisoes levariam ~{600000/(n/dt)/3600:.1f} h so de ambiente")
    env.close()


if __name__ == "__main__":
    main()