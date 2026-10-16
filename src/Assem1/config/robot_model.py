#!/usr/bin/env python3
"""
robot_model.py  -  physical model of the Assem1 two-wheel balancing robot with arm.

Pure numpy (no ROS).  Used by:
  * balance_core.py   (arm feed-forward lean angle)
  * fast_sim.py       (fast physics for RL pre-training)
  * robot_balance_env.py / balance_controller.py

All numbers come from your Assem1.urdf / STL files.
Frames: base_link: x forward, y left, z up.  Axle is at z = AXLE_Z above base_link origin.
Arm joints rotate about the body-frame axis (0,-1,0) (= base y axis), so the whole
problem is planar (pitch plane).  Positive pitch = leaning FORWARD (+x).
"""
import math
import numpy as np

G = 9.81
WHEEL_RADIUS = 0.08          # from left_wheel.STL (diameter 0.16 m)   <-- your old plugin said 0.10 (wrong)
WHEEL_SEPARATION = 0.2732       # = 2 x 0.1366 (wheel joint origins in Assem1.urdf, same as diff-drive plugin)
BODY_Z = 0.05                # body origin height above base_link origin (base_to_body joint, body UPRIGHT: rpy 0 0 0)
AXLE_Z = 0.0815              # wheel centre height above base_link origin (= BODY_Z + wheel joint z 0.0315)
AXLE_X = 0.0                 # wheel centre straight below the body origin
M_WHEEL = 0.2257             # each wheel
I_WHEEL = 0.000702           # each wheel, about its spin axis

# ---------------------------------------------------------------------------
# ballast (battery).  A balancing robot needs its centre of mass ABOVE the axle.
# In your original model it was only ~2 cm above the axle -> pitch equilibrium swings +-40 deg with arm motion.
# This is added to the URDF as link "ballast".  Put your REAL battery here.
# position is in base_link frame (x forward, z up)
BALLAST_MASS = 0.50              # 2 packs x 0.25 kg (ballast_left / ballast_right in the URDF, y = +-0.07)
BALLAST_POS = (-0.009, 0.0, 0.20)         # pack centre 20 cm up (invisible), 0.9 cm behind the axle -> balanced in camera pose,
                                          # COM 5.2 cm above the axle: a 5 deg hill needs only ~11 deg lean
BALLAST_SIZE = (0.08, 0.10, 0.076)         # box x,y,z  (only for inertia / visual)

def _rot(axis, a):
    axis = np.array(axis, float); axis /= np.linalg.norm(axis)
    x, y, z = axis; c, s = math.cos(a), math.sin(a); C = 1 - c
    return np.array([[c+x*x*C, x*y*C-z*s, x*z*C+y*s],
                     [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
                     [z*x*C-y*s, z*y*C+x*s, c+z*z*C]])

def _T(R, p):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = p; return M

AX1 = np.array([0.0065584, -0.99997, -0.0030051])
# name: (mass, com in link frame, inertia tensor about com in link frame)
LINKS = {
 'body': (0.5231, [-0.00063, -3.188e-05, 0.00754],
          [[0.002215, 8.565e-06, -8.353e-06], [8.565e-06, 0.000817, -6.06e-06], [-8.353e-06, -6.06e-06, 0.002583]]),
 'arm1': (0.0549, [0.00043, -0.0028, 0.04227],
          [[2.346e-05, 0, 0], [0, 2.167e-05, -1.423e-06], [0, -1.423e-06, 6.846e-06]]),
 'arm2': (0.0549, [0.01828, -0.00142, 0.03751],
          [[2.012e-05, -6.781e-07, -6.665e-06], [-6.781e-07, 2.166e-05, -1.377e-06], [-6.665e-06, -1.377e-06, 1.02e-05]]),
 'arm3': (0.0375, [0.00815, 0.00521, 0.02824],
          [[1.893e-05, 4.182e-07, -4.053e-06], [4.182e-07, 2.032e-05, 1.537e-06], [-4.053e-06, 1.537e-06, 5.789e-06]]),
}

ARM_LIMITS = np.array([[-1.57, 1.57]] * 3)
# body upright: q = (0,0,0) is the arm straight up.  Joint axes are ~ -y, so negative q1/q2 tilt FORWARD and
# positive q3 bends the last link forward.  CAMERA pose: link1 vertical (base to the sky), last link looking forward (+x)
ARM_HOME = (0.0, -0.6, 0.97)

def _fk(q):
    q1, q2, q3 = q
    Tb = _T(np.eye(3), [0, 0, BODY_Z])
    T1 = Tb @ _T(np.eye(3), [-0.0047285, 0.00080999, 0.049815]) @ _T(_rot(AX1, q1), [0, 0, 0])
    T2 = T1 @ _T(np.eye(3), [0.0016571, -0.020702, 0.070511]) @ _T(_rot(AX1, q2), [0, 0, 0])
    T3 = T2 @ _T(np.eye(3), [0.031019, 0.022517, 0.062295]) @ _T(_rot(-AX1, q3), [0, 0, 0])
    return {'body': Tb, 'arm1': T1, 'arm2': T2, 'arm3': T3}

def sprung_params(q, ballast=True):
    """Everything above the axle (body + arm + ballast; wheels excluded).
    Returns dict: m, x_c, h (COM relative to axle, base frame, pitch=0), I_c (pitch inertia about COM), l, I_a."""
    Ts = _fk(q)
    parts = []   # (mass, world pos, I_yy about own com)
    for n, (m, c, I) in LINKS.items():
        R = Ts[n][:3, :3]
        p = (Ts[n] @ np.array([*c, 1.0]))[:3]
        Iw = R @ np.array(I) @ R.T
        parts.append((m, p, Iw[1, 1]))
    if ballast and BALLAST_MASS > 0:
        sx, sy, sz = BALLAST_SIZE
        parts.append((BALLAST_MASS, np.array(BALLAST_POS), BALLAST_MASS * (sx*sx + sz*sz) / 12.0))
    m = sum(p[0] for p in parts)
    com = sum(p[0] * p[1] for p in parts) / m
    x_c = com[0] - AXLE_X; h = com[2] - AXLE_Z
    I_c = sum(p[2] + p[0] * ((p[1][0]-com[0])**2 + (p[1][2]-com[2])**2) for p in parts)
    l2 = x_c**2 + h**2
    return dict(m=m, x_c=x_c, h=h, I_c=I_c, l=math.sqrt(l2), I_a=I_c + m*l2)

def equilibrium_pitch(q, ballast=True):
    """Pitch (rad, + = forward lean) at which the COM is exactly above the axle."""
    p = sprung_params(q, ballast)
    return math.atan2(-p['x_c'], p['h'])

if __name__ == '__main__':
    for q in [(0, 0, 0), ARM_HOME, (0.3, -1.0, 1.4), (-0.4, -0.2, 0.5)]:
        for b in (False, True):
            p = sprung_params(q, b)
            print(q, 'ballast' if b else 'original', 'm=%.3f x_c=%.4f h=%.4f  eq_pitch=%.1f deg' %
                  (p['m'], p['x_c'], p['h'], math.degrees(equilibrium_pitch(q, b))))
