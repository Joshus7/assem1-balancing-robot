#!/usr/bin/env python3
"""
fast_sim.py - Gymnasium environment with a fast physics model of the robot (no Gazebo, ~1000+ steps/s).
Same observation / action / reward as the Gazebo env (task_common.py).  Use it for stage-1 training.
Domain randomisation: mass +-15%, unknown COM error, motor lag, command delay, sensor noise, random pushes,
random hills (0, +-3, +-5 deg, like the road in worlds/road_hills.world).
"""
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from robot_model import equilibrium_pitch
from balance_core import BalanceController
from fast_sim_core import WIPSim
from task_common import *

ARM_RATE = 1.2     # rad/s, same as BalanceNode


class FastBalanceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None, randomize=True, v_max=0.8, fixed_v=None, arm_random=True, pushes=True, use_ff=True,
                 hills=True):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.rng = random.Random(seed)
        self.sim = WIPSim(rng=self.rng)
        self.randomize, self.pushes, self.hills = randomize, pushes, hills
        self.cmds = CommandGenerator(self.rng, v_max, fixed_v, arm_random)
        self.ctrl = BalanceController(use_ff=use_ff)

    def _state(self):
        p, pr, v = self.sim.measure()
        return dict(pitch=p, pitch_rate=pr, v=v, arm_q=self.arm_q.copy(), arm_dq=self.arm_dq.copy(),
                    theta_ff=self.ctrl.theta_ff if self.ctrl.theta_ff is not None else 0.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        self.sim.randomize(self.randomize)
        self.cmds.reset()
        self.arm_q = ARM_START.copy(); self.arm_dq = np.zeros(3)
        self.sim.reset(pitch=equilibrium_pitch(self.arm_q) + self.rng.uniform(-0.06, 0.06), q=self.arm_q)
        self.ctrl.reset()
        self.t = 0.0; self.steps = 0; self.prev_a = 0.0
        self.t_push = self.rng.uniform(2, 6)
        self.t_slope = self.rng.uniform(2, 5)
        # let the controller settle its filters (no RL)
        for _ in range(10):
            p, pr, v = self.sim.measure()
            self.sim.step_control_tick(self.ctrl.update(p, pr, v, 0.0, self.arm_q, 0.01))
        return build_obs(self._state(), self.cmds.v_cmd, 0.0), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        lean = MAX_LEAN_ACTION * a
        v_cmd, arm_t = self.cmds.step(RL_DT)
        for _ in range(4):                                   # 4 x 10 ms control ticks
            dq = np.clip(arm_t - self.arm_q, -ARM_RATE * 0.01, ARM_RATE * 0.01)
            self.arm_q = self.arm_q + dq; self.arm_dq = dq / 0.01
            p, pr, v = self.sim.measure()
            vc = self.ctrl.update(p, pr, v, v_cmd, self.arm_q, 0.01, lean_cmd=lean)
            self.sim.step_control_tick(vc)
        self.sim.set_arm(self.arm_q)
        self.t += RL_DT; self.steps += 1
        if self.hills and self.t > self.t_slope:            # road changes: flat / uphill / downhill
            self.sim.slope = math.radians(self.rng.choice([0, 0, 3, -3, 5, -5]))
            self.t_slope = self.t + self.rng.uniform(3, 7)
        if self.pushes and self.t > self.t_push:
            self.sim.phid += self.rng.uniform(-0.6, 0.6)
            self.t_push = self.t + self.rng.uniform(3, 8)
        s = self._state()
        terminated = fell(s)
        r = -10.0 if terminated else compute_reward(s, v_cmd, a, self.prev_a)
        self.prev_a = a
        truncated = self.steps >= MAX_STEPS
        return build_obs(s, v_cmd, a), r, terminated, truncated, {}
