#!/usr/bin/env python3
"""Offline test of PID + arm feed-forward in the fast physics model. Run:  python3 test_fast_sim.py"""
import math, random, sys
import numpy as np
from robot_model import equilibrium_pitch, ARM_HOME
from balance_core import BalanceController
from fast_sim_core import WIPSim


def run(use_ff, arm_traj, v_prof, T=20.0, seed=0, randomize=True, init_pitch=0.08):
    sim = WIPSim(rng=random.Random(seed)); sim.randomize(randomize)
    q0 = arm_traj(0.0)
    sim.reset(pitch=equilibrium_pitch(q0) + init_pitch, q=q0)
    ctrl = BalanceController(use_ff=use_ff)
    max_pitch_err = 0; fell = False; vs = []
    t = 0.0
    while t < T:
        q = arm_traj(t)
        if int(round(t / 0.01)) % 4 == 0:
            sim.set_arm(q)            # arm updates at 25 Hz like in the env
        p, pr, v = sim.measure()
        vcmd = ctrl.update(p, pr, v, v_prof(t), q, 0.01)
        sim.step_control_tick(vcmd)
        t += 0.01
        err = sim.phi - equilibrium_pitch(q)
        if t > 3: max_pitch_err = max(max_pitch_err, abs(err))
        if abs(sim.phi) > 0.8:
            fell = True; break
        vs.append(sim.xd)
    return fell, t, max_pitch_err, np.array(vs)


def sweep(t):      # arm moves through large range, ~1 rad/s
    return (0.35 * math.sin(0.8 * t), -0.6 + 0.4 * math.sin(1.3 * t), 0.95 + 0.45 * math.sin(0.7 * t))

if __name__ == '__main__':
    still = lambda t: ARM_HOME
    v0 = lambda t: 0.0
    vstep = lambda t: 0.0 if t < 5 else (0.6 if t < 12 else (-0.4 if t < 17 else 0.0))
    ok = True
    for name, traj, vp in [('arm still, stand', still, v0), ('arm sweeping, stand', sweep, v0),
                           ('arm sweeping + drive', sweep, vstep)]:
        for use_ff in (True, False):
            falls = 0; worst = 0; N = 20
            for s in range(N):
                f, t, e, vs = run(use_ff, traj, vp, seed=s)
                falls += f; worst = max(worst, e)
            print('%-24s FF=%-5s  falls %2d/%d   worst pitch error %.1f deg' % (name, use_ff, falls, N, math.degrees(worst)))
            if use_ff and falls: ok = False
    f, t, e, vs = run(True, still, vstep, seed=1, randomize=False)
    print('speed tracking (nominal model): v at t=10s = %.2f (ref 0.6), t=15s = %.2f (ref -0.4)' % (vs[999], vs[1499]))
    print('RESULT:', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)
