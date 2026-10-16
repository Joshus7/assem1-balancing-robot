#!/usr/bin/env python3
"""
balance_core.py - the balance controller (pure python, no ROS).

Cascade controller, 3 loops, runs at 100 Hz (IMU rate):

 1. ARM FEED-FORWARD  : theta_ff = pitch at which the centre of mass of (body+arm+battery)
                        is exactly above the wheel axle, computed from the arm joint angles.
                        When the arm moves, the target pitch moves WITH it -> no fight.
 2. VELOCITY PI (outer): theta_ref = theta_ff + lean_cmd + kv_p*(v_ref - v) + kv_i*integral
                        (if too fast -> lean back, if too slow -> lean forward)
 3. PITCH PD (inner)   : a = kp*(pitch - theta_ref) + kd*pitch_rate    [m/s^2]
                        v_cmd += a*dt          (wheel speed command, m/s -> /cmd_vel linear.x)

Why integrate acceleration?  The wheels are VELOCITY controlled, but a falling pendulum needs an
ACCELERATION of the axle proportional to the tilt.  (Your old PID sent a velocity proportional to the
tilt, which only worked by accident through the plugin's lag.)

Gains: LQR design of the linearised model gave the structure (kp > w0^2/b ~ 2.3); real gains were then
tuned in test_fast_sim.py with noise, 20-40 ms delay+motor lag and +-15% mass error (high LQR gains FAIL with that lag).
Outer loop + speed ramp re-tuned for the upright robot on 5 deg hills (fast_sim_core slope).
"""
import math
import numpy as np
from robot_model import equilibrium_pitch


class BalanceController:
    def __init__(self,
                 kp=6.0, kd=0.8,              # inner pitch loop      [m/s^2 per rad], [m/s^2 per rad/s]
                 kv_p=0.30, kv_i=0.40,       # outer velocity loop   [rad per m/s], [rad per m]  (strong enough for 5 deg hills)
                 max_lean=0.45,               # rad, limit of the velocity loop lean (5 deg hill needs ~0.3 rad)
                 max_v=1.5,                   # m/s, wheel speed limit
                 ff_filter=0.08,              # s, low-pass on the feed-forward angle
                 v_ref_acc=0.2,               # m/s^2, ramp limit on the speed command (None = no ramp)
                 use_ff=True):
        self.kp, self.kd, self.kv_p, self.kv_i = kp, kd, kv_p, kv_i
        self.max_lean, self.max_v, self.ff_filter, self.use_ff = max_lean, max_v, ff_filter, use_ff
        self.v_ref_acc = v_ref_acc
        self.reset()

    def reset(self, v_cmd=0.0):
        self.v_cmd = v_cmd
        self.int_v = 0.0
        self.v_ref_f = 0.0
        self.theta_ff = None
        self._q_cache = None

    # feed-forward angle for given arm angles (cached; arm moves slowly)
    def feedforward(self, arm_q):
        if not self.use_ff:
            return 0.0
        key = tuple(round(float(a), 2) for a in arm_q)
        if key != self._q_cache:
            self._q_cache = key
            self._ff_target = equilibrium_pitch(key)
        return self._ff_target

    def update(self, pitch, pitch_rate, v_meas, v_ref, arm_q, dt, lean_cmd=0.0, ff_bias=0.0):
        """Returns wheel speed command v_cmd (m/s).
        pitch [rad,+forward], pitch_rate [rad/s], v_meas [m/s wheel speed], v_ref [m/s],
        arm_q (3 joint angles), lean_cmd = extra lean from RL/user [rad]."""
        ff = self.feedforward(arm_q)
        if self.theta_ff is None:
            self.theta_ff = ff
        a = dt / (self.ff_filter + dt)
        self.theta_ff += a * (ff - self.theta_ff)

        if self.v_ref_acc:                     # smooth speed steps (a step makes the robot lunge)
            dv = self.v_ref_acc * dt
            self.v_ref_f += max(-dv, min(dv, v_ref - self.v_ref_f))
            v_ref = self.v_ref_f
        verr = v_ref - v_meas
        self.int_v += verr * dt
        lim = self.max_lean / max(self.kv_i, 1e-6)
        self.int_v = max(-lim, min(lim, self.int_v))
        lean = self.kv_p * verr + self.kv_i * self.int_v
        lean = max(-self.max_lean, min(self.max_lean, lean))

        theta_ref = self.theta_ff + ff_bias + lean_cmd + lean
        acc = self.kp * (pitch - theta_ref) + self.kd * pitch_rate
        self.v_cmd += acc * dt
        # stay close to the real wheel speed (anti wind-up)
        self.v_cmd = max(v_meas - 1.0, min(v_meas + 1.0, self.v_cmd))
        self.v_cmd = max(-self.max_v, min(self.max_v, self.v_cmd))
        self.theta_ref = theta_ref
        return self.v_cmd
