#!/usr/bin/env python3
"""
run_trained_model.py - run a trained policy in Gazebo (PID + feed-forward + RL lean offset).
    python3 run_trained_model.py --name stage2_gazebo                 # random speed + arm targets
    python3 run_trained_model.py --name stage2_gazebo --v 0.4         # constant 0.4 m/s
    python3 run_trained_model.py --pid-only --v 0.4                   # no policy, only PID + FF (baseline)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  must be imported BEFORE numpy: pip torch ships its own libomp, conda numpy/openblas
              # uses llvm-openmp -> importing numpy first crashes on macOS (OMP Error #179)
import argparse
import numpy as np
from robot_balance_env import GazeboBalanceEnv

ap = argparse.ArgumentParser()
ap.add_argument('--name', default=None); ap.add_argument('--v', type=float, default=None)
ap.add_argument('--pid-only', action='store_true'); ap.add_argument('--no-arm', action='store_true')
a = ap.parse_args()

env = GazeboBalanceEnv(fixed_v=a.v, arm_random=not a.no_arm)
if a.pid_only or a.name is None:
    pol = lambda o: 0.0
else:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    norm = VecNormalize.load(a.name + '_vecnorm.pkl', DummyVecEnv([lambda: env]))
    norm.training = False; norm.norm_reward = False
    model = PPO.load(a.name)
    pol = lambda o: float(model.predict(norm.normalize_obs(o[None, :]), deterministic=True)[0][0][0])
try:
    ep = 0
    while True:
        o, _ = env.reset(); ret = 0.0; n = 0
        while True:
            o, r, te, tr, _ = env.step([pol(o)]); ret += r; n += 1
            if n % 25 == 0:
                print('t=%.0fs  v=%.2f (cmd %.2f)  pitch err=%.1f deg' % (n * 0.04, o[2], o[3], np.degrees(o[0])))
            if te or tr:
                break
        ep += 1
        print('episode %d: %s after %d steps, return %.1f' % (ep, 'FELL' if te else 'ok', n, ret))
except KeyboardInterrupt:
    pass
finally:
    env.close()
