#!/usr/bin/env python3
"""
robot_balance_env.py - Gymnasium environment for the Gazebo simulation (stage-2 training / evaluation).

Differences to your old env (all on purpose):
  * robot is reset with the /reset_world service (std_srvs/Empty, provided by libgazebo_ros_init that you
    already load).  Your old env used /set_entity_state + entity 'bot' - but the entity is called 'Assem1'
    and the state plugin is not loaded, so the robot was NEVER reset after a fall.
  * fixed time step: every step waits exactly 0.04 s of SIMULATION time (old: random 10-30 ms).
  * the 100 Hz balance controller (PID + arm feed-forward) lives in BalanceNode, RL only adds a lean offset.
  * observation has measured speed, speed command, arm angles + velocities (old: none of them).
  * reward rewards following the speed command (old: nothing rewarded moving).
"""
import time
import threading
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import rclpy
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty

from balance_node import BalanceNode
from task_common import *


class GazeboBalanceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None, v_max=0.6, fixed_v=None, arm_random=True, settle_timeout=4.0):
        super().__init__()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
        self.rng = random.Random(seed)
        self.cmds = CommandGenerator(self.rng, v_max, fixed_v, arm_random)
        self.settle_timeout = settle_timeout

        if not rclpy.ok():
            rclpy.init()
        self.node = BalanceNode('rl_balance_env', use_sim_time=True, use_ff=True)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)
        self._thread = threading.Thread(target=self.executor.spin, daemon=True)
        self._thread.start()
        self.reset_client = self.node.create_client(Empty, '/reset_world')
        self._wait_for_sim()
        self.steps = 0; self.prev_a = 0.0

    # ------------------------------------------------------------------ helpers
    def _now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _wait_for_sim(self, timeout=60.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            s = self.node.get_state()
            if s['imu_count'] > 5 and s['odom_count'] > 5 and self.reset_client.service_is_ready():
                if self.node.count_publishers('/cmd_vel') > 1:
                    raise RuntimeError('Another node is already balancing the robot (publishing /cmd_vel). '
                                       'Launch Gazebo with:  ros2 launch Assem1 robot_gazebo.launch.py balance:=false')
                return
            time.sleep(0.2)
        raise RuntimeError('No /imu/data, /odom or /reset_world.  Is Gazebo running (ros2 launch Assem1 robot_gazebo.launch.py) '
                           'and is the robot spawned?  Run check_setup.py.')

    def _sim_sleep(self, dt):
        """wait until `dt` seconds of SIMULATION time passed"""
        t_end = self._now() + dt
        wall0 = time.time()
        while self._now() < t_end:
            time.sleep(0.0005)
            if time.time() - wall0 > max(5.0, 50 * dt):
                raise RuntimeError('Simulation time is not advancing (Gazebo paused / crashed?)')

    def _call_reset_world(self):
        fut = self.reset_client.call_async(Empty.Request())
        t0 = time.time()
        while not fut.done():
            time.sleep(0.01)
            if time.time() - t0 > 10:
                raise RuntimeError('/reset_world did not answer')

    def _state(self):
        return self.node.get_state()

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng.seed(seed)
        for attempt in range(4):
            self.node.set_speed(0.0, 0.0); self.node.set_lean(0.0)
            self._call_reset_world()
            self._sim_sleep(0.1)
            self.node.reset_controller()
            self.node.snap_arm(ARM_START)
            # let the PID catch the robot and settle (RL not active yet)
            t_ok = None; t_start = self._now()
            while self._now() - t_start < self.settle_timeout:
                self._sim_sleep(0.05)
                s = self._state()
                if abs(s['pitch'] - s['theta_ff']) < 0.12 and abs(s['v']) < 0.2:
                    t_ok = t_ok or self._now()
                    if self._now() - t_ok > 0.3:
                        break
                else:
                    t_ok = None
                if abs(s['pitch']) > 1.0:
                    break
            s = self._state()
            if abs(s['pitch'] - s['theta_ff']) < 0.25:
                break
            print('[env] robot did not stabilise after reset (attempt %d): pitch=%.2f ff=%.2f' % (attempt + 1, s['pitch'], s['theta_ff']))
        self.cmds.reset()
        self.node.set_arm_target(ARM_START)
        self.steps = 0; self.prev_a = 0.0
        return build_obs(self._state(), self.cmds.v_cmd, 0.0), {}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1, 1))
        v_cmd, arm_t = self.cmds.step(RL_DT)
        self.node.set_lean(MAX_LEAN_ACTION * a)
        self.node.set_speed(v_cmd, 0.0)
        self.node.set_arm_target(arm_t)
        self._sim_sleep(RL_DT)
        s = self._state()
        self.steps += 1
        terminated = fell(s)
        r = -10.0 if terminated else compute_reward(s, v_cmd, a, self.prev_a)
        self.prev_a = a
        truncated = self.steps >= MAX_STEPS
        if terminated or truncated:
            self.node.set_speed(0.0, 0.0)
        return build_obs(s, v_cmd, a), r, terminated, truncated, {}

    def close(self):
        try:
            self.node.stop_robot()
            self.executor.shutdown()
            self.node.destroy_node()
        except Exception:
            pass
