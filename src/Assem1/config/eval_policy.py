#!/usr/bin/env python3
"""
eval_policy.py - compare PID-only (policy off) with your trained policy in the FAST simulator.
    python3 eval_policy.py                         # PID only baseline
    python3 eval_policy.py --name stage1_fast      # baseline AND policy
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  must be imported BEFORE numpy: pip torch ships its own libomp, conda numpy/openblas
              # uses llvm-openmp -> importing numpy first crashes on macOS (OMP Error #179)
import argparse
import numpy as np
from fast_sim import FastBalanceEnv


def run(policy, episodes, seed0=1000):
    env = FastBalanceEnv(seed=seed0)
    tot = falls = 0; err = []; 
    for ep in range(episodes):
        o, _ = env.reset(seed=seed0 + ep)
        while True:
            o, r, te, tr, _ = env.step([policy(o)])
            tot += r; err.append(abs(o[4])); falls += int(te)
            if te or tr:
                break
    return tot / episodes, falls, float(np.mean(err))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--name', default=None); ap.add_argument('--episodes', type=int, default=30)
    a = ap.parse_args()
    ret, falls, e = run(lambda o: 0.0, a.episodes)
    print('PID + arm FF only : return/episode %.1f   falls %d/%d   mean |speed error| %.3f m/s' % (ret, falls, a.episodes, e))
    if a.name:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        venv = DummyVecEnv([lambda: FastBalanceEnv(seed=0)])
        norm = VecNormalize.load(a.name + '_vecnorm.pkl', venv); norm.training = False; norm.norm_reward = False
        model = PPO.load(a.name)
        pol = lambda o: float(model.predict(norm.normalize_obs(o[None, :]), deterministic=True)[0][0][0])
        ret, falls, e = run(pol, a.episodes)
        print('PID + RL policy   : return/episode %.1f   falls %d/%d   mean |speed error| %.3f m/s' % (ret, falls, a.episodes, e))
