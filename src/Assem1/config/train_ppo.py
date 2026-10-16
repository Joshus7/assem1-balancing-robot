#!/usr/bin/env python3
"""
train_ppo.py - PPO training of the high-level policy (lean offset on top of PID + arm feed-forward).

STAGE 1 (fast simulator, minutes, no Gazebo):
    python3 train_ppo.py --env fast --steps 1500000 --n-envs 8 --name stage1_fast
STAGE 2 (Gazebo with GUI, watch it learn). Terminal 1:  ros2 launch Assem1 robot_gazebo.launch.py balance:=false
    python3 train_ppo.py --env gazebo --iterations 1000 --load stage1_fast --lr 1e-4 --name stage2_gazebo
    (1 iteration = 1 PPO update = --n-steps steps of 0.04 s sim time; Ctrl+C saves the model)

Outputs  <name>.zip  and  <name>_vecnorm.pkl   (ALWAYS keep the two together).
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: F401  must be imported BEFORE numpy: pip torch ships its own libomp, conda numpy/openblas
              # uses llvm-openmp -> importing numpy first crashes on macOS (OMP Error #179)
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


class EpisodePrinter(BaseCallback):
    """prints one line per finished episode so you can follow training next to the Gazebo window"""
    def __init__(self, n_steps):
        super().__init__(); self.n_steps = n_steps; self.ep = 0; self.falls = 0; self.ret = None; self.len = None

    def _on_step(self):
        rew = self.locals['rewards']; dones = self.locals['dones']; infos = self.locals['infos']
        if self.ret is None:
            self.ret = [0.0] * len(rew); self.len = [0] * len(rew)
        for i in range(len(rew)):
            self.ret[i] += float(self.training_env.get_original_reward()[i])     # un-normalised reward
            self.len[i] += 1
            if dones[i]:
                self.ep += 1
                fell = not infos[i].get('TimeLimit.truncated', False)
                self.falls += fell
                print('[episode %4d | iteration %4d] %s after %5.1f s   return %7.1f   falls so far %d' % (
                    self.ep, self.num_timesteps // self.n_steps + 1, 'FELL' if fell else 'ok  ',
                    self.len[i] * 0.04, self.ret[i], self.falls), flush=True)
                self.ret[i] = 0.0; self.len[i] = 0
        return True


def make_fast(seed):
    def _f():
        from fast_sim import FastBalanceEnv
        return FastBalanceEnv(seed=seed)
    return _f


def make_gazebo(seed):
    def _f():
        from robot_balance_env import GazeboBalanceEnv
        return GazeboBalanceEnv(seed=seed)
    return _f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env', choices=['fast', 'gazebo'], default='fast')
    ap.add_argument('--steps', type=int, default=1_500_000, help='total environment steps')
    ap.add_argument('--iterations', type=int, default=None, help='PPO iterations (overrides --steps): steps = iterations * n_steps')
    ap.add_argument('--n-steps', type=int, default=None, help='steps per env per iteration (default: 256 fast, 128 gazebo)')
    ap.add_argument('--n-envs', type=int, default=8, help='parallel envs (fast only; gazebo always 1)')
    ap.add_argument('--name', type=str, default='stage1_fast')
    ap.add_argument('--load', type=str, default=None, help='continue from <load>.zip and <load>_vecnorm.pkl')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.env == 'fast':
        n = max(1, args.n_envs)
        fns = [make_fast(args.seed + i) for i in range(n)]
        venv = SubprocVecEnv(fns) if n > 1 else DummyVecEnv(fns)
    else:
        n = 1
        venv = DummyVecEnv([make_gazebo(args.seed)])
    n_steps = args.n_steps or (max(256, 2048 // n) if args.env == 'fast' else 128)
    if args.iterations:
        args.steps = args.iterations * n_steps * n
    print('[*] %d steps = %d iterations of %d steps x %d envs' % (args.steps, args.steps // (n_steps * n), n_steps, n))

    if args.load and os.path.exists(args.load + '_vecnorm.pkl'):
        print('[*] loading normalisation stats from', args.load + '_vecnorm.pkl')
        env = VecNormalize.load(args.load + '_vecnorm.pkl', venv)
        env.training = True; env.norm_reward = True
    else:
        env = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    if args.load and os.path.exists(args.load + '.zip'):
        print('[*] loading model', args.load)
        model = PPO.load(args.load, env=env, custom_objects={'learning_rate': args.lr, 'lr_schedule': (lambda _: args.lr),
                                                             'n_steps': n_steps, 'batch_size': min(256, n_steps * n)})
    else:
        print('[*] new PPO model')
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=args.lr,
                    n_steps=n_steps, batch_size=min(256, n_steps * n), n_epochs=10,
                    gamma=0.98, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                    policy_kwargs=dict(net_arch=[64, 64]), seed=args.seed)

    ckpt = CheckpointCallback(save_freq=max((50_000 if args.env == 'fast' else 5_000) // n, 1), save_path='./checkpoints', name_prefix=args.name)
    try:
        cbs = [ckpt] + ([EpisodePrinter(n_steps * n)] if args.env == 'gazebo' else [])
        model.learn(total_timesteps=args.steps, callback=CallbackList(cbs), reset_num_timesteps=True)
    except KeyboardInterrupt:
        print('\n[!] stopped by Ctrl+C - saving what was learned so far')
    finally:
        model.save(args.name)
        env.save(args.name + '_vecnorm.pkl')
        print('[done] saved %s.zip and %s_vecnorm.pkl' % (args.name, args.name))
        env.close()


if __name__ == '__main__':
    main()
