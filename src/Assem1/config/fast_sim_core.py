#!/usr/bin/env python3
"""
fast_sim_core.py - fast planar physics of the robot (wheel-inverted-pendulum with moving arm).
Equations (Lagrange, x = axle position, phi = pitch, + forward):
   (Mx+m) xdd + m Z phidd - m X phid^2 = tau/r
   m Z xdd + I_a phidd              = m g X - tau
   X = x_c cos(phi) + h sin(phi),  Z = -x_c sin(phi) + h cos(phi)   (COM relative to the axle)
The wheel motor is a velocity servo:  tau = clip(Ks*(v_cmd - xd), +-tau_max).
Hills: self.slope [rad, + = uphill in +x] adds the gravity pull  -(M_wheels+m) g sin(slope)  along the road
(phi stays measured from the true vertical, like the IMU).
"""
import math, random
from robot_model import (ARM_HOME, G, WHEEL_RADIUS as R, M_WHEEL, I_WHEEL, sprung_params, equilibrium_pitch)

MX = 2 * M_WHEEL + 2 * I_WHEEL / R**2


class WIPSim:
    def __init__(self, dt=0.002, rng=None):
        self.dt = dt
        self.rng = rng or random.Random(0)
        self.randomize(False)
        self.reset()

    def randomize(self, on=True, mass_err=0.15, servo_tau=(0.012, 0.05), delay=(1, 3)):
        r = self.rng
        self.mass_scale = 1 + r.uniform(-mass_err, mass_err) if on else 1.0
        self.com_dx = r.uniform(-0.004, 0.004) if on else 0.0     # unknown COM error (m)
        self.com_dh = r.uniform(-0.006, 0.006) if on else 0.0
        tau = r.uniform(*servo_tau) if on else 0.02
        self.Ks = R * (2 * M_WHEEL + 1.0) / tau                    # velocity-servo stiffness
        self.tau_max = 20.0
        self.delay_ticks = r.randint(*delay) if on else 2          # command delay in 10 ms ticks
        self.noise = on

    def set_arm(self, q):
        p = sprung_params(q)
        s = self.mass_scale
        self.m = p['m'] * s
        self.x_c = p['x_c'] + self.com_dx
        self.h = p['h'] + self.com_dh
        l2 = self.x_c**2 + self.h**2
        self.I_a = p['I_c'] * s + self.m * l2
        self.q = tuple(q)

    def reset(self, pitch=0.0, pitch_rate=0.0, v=0.0, q=ARM_HOME):
        self.set_arm(q)
        self.x = 0.0; self.xd = v; self.phi = pitch; self.phid = pitch_rate
        self.slope = 0.0
        self.cmd_queue = [v] * (self.delay_ticks + 1)
        self.t = 0.0

    def step_control_tick(self, v_cmd, tick=0.01):
        """advance one 10 ms control tick with command v_cmd (applied after delay)."""
        self.cmd_queue.append(v_cmd)
        v_applied = self.cmd_queue.pop(0)
        n = int(round(tick / self.dt))
        m, I_a, x_c, h = self.m, self.I_a, self.x_c, self.h
        for _ in range(n):
            c, s = math.cos(self.phi), math.sin(self.phi)
            X = x_c * c + h * s
            Z = -x_c * s + h * c
            tau = self.Ks * (v_applied - self.xd)
            tau = max(-self.tau_max, min(self.tau_max, tau))
            a11 = MX + m; a12 = m * Z; a22 = I_a
            b1 = tau / R + m * X * self.phid**2 - (2 * M_WHEEL + m) * G * math.sin(self.slope)
            b2 = m * G * X - tau
            det = a11 * a22 - a12 * a12
            xdd = (b1 * a22 - a12 * b2) / det
            phidd = (a11 * b2 - a12 * b1) / det
            self.xd += xdd * self.dt
            self.phid += phidd * self.dt
            self.x += self.xd * self.dt
            self.phi += self.phid * self.dt
        self.t += tick
        self.last_acc = xdd

    def measure(self):
        r = self.rng
        if self.noise:
            return (self.phi + r.gauss(0, 0.002), self.phid + r.gauss(0, 0.02), self.xd + r.gauss(0, 0.01))
        return self.phi, self.phid, self.xd
