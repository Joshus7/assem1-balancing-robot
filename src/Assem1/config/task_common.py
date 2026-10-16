#!/usr/bin/env python3
"""
task_common.py - the RL task, shared by the fast simulator (fast_sim.py) and Gazebo (robot_balance_env.py).
Because both use EXACTLY the same observation / action / reward, a policy trained in the fast sim can be
loaded in Gazebo and fine-tuned there.

What does the RL policy do?
  The PID+feed-forward controller (balance_core.py) always keeps the robot upright (100 Hz).
  The policy (25 Hz) adds one thing: a small extra LEAN ANGLE  (action in [-1,1] -> +-0.15 rad).
  It learns to use it to follow the commanded speed faster and with less overshoot while the arm moves.
  The ARM is moved by the task (random targets while training, or by you) - the policy only sees it.
"""
import numpy as np
from robot_model import ARM_HOME

OBS_DIM = 12
ACT_DIM = 1
MAX_LEAN_ACTION = 0.15          # rad
RL_DT = 0.04                    # s   (25 Hz)
MAX_STEPS = 500                 # 20 s episodes
ARM_LO = np.array([-0.4, -1.0, 0.5])       # random arm targets stay in this box around the camera pose
ARM_HI = np.array([0.3, -0.2, 1.4])
ARM_START = np.array(ARM_HOME)
FALL_PITCH_ERR = 0.6            # rad away from the balance point -> "fell"


class CommandGenerator:
    """random speed commands and random arm targets"""
    def __init__(self, rng, v_max=0.8, fixed_v=None, arm_random=True):
        self.rng, self.v_max, self.fixed_v, self.arm_random = rng, v_max, fixed_v, arm_random
        self.reset()

    def reset(self):
        self.v_cmd = 0.0 if self.fixed_v is None else self.fixed_v
        self.arm_target = ARM_START.copy()
        self.t_v = self.rng.uniform(1.0, 3.0)      # first command after 1-3 s
        self.t_arm = self.rng.uniform(1.0, 3.0)

    def step(self, dt):
        self.t_v -= dt; self.t_arm -= dt
        if self.fixed_v is None and self.t_v <= 0:
            self.v_cmd = 0.0 if self.rng.random() < 0.3 else self.rng.uniform(-self.v_max, self.v_max)
            self.t_v = self.rng.uniform(2.0, 5.0)
        if self.arm_random and self.t_arm <= 0:
            self.arm_target = ARM_LO + (ARM_HI - ARM_LO) * np.array([self.rng.random() for _ in range(3)])
            self.t_arm = self.rng.uniform(1.0, 4.0)
        return self.v_cmd, self.arm_target


def build_obs(s, v_cmd, prev_action):
    """s: dict(pitch, pitch_rate, v, arm_q, arm_dq, theta_ff)  -> 12 numbers"""
    return np.array([s['pitch'] - s['theta_ff'], s['pitch_rate'], s['v'], v_cmd, s['v'] - v_cmd,
                     *s['arm_q'], *s['arm_dq'], prev_action], dtype=np.float32)


def fell(s):
    return abs(s['pitch'] - s['theta_ff']) > FALL_PITCH_ERR or abs(s['pitch']) > 1.0


def compute_reward(s, v_cmd, action, prev_action):
    pe = s['pitch'] - s['theta_ff']
    r = 1.0 * np.exp(-((s['v'] - v_cmd) / 0.15) ** 2)          # follow the speed command
    r += 0.5 * np.exp(-(pe / 0.15) ** 2)                       # stay near the balance point
    r -= 0.1 * abs(action - prev_action)                       # smooth
    r -= 0.01 * s['pitch_rate'] ** 2                           # calm
    return float(r)
