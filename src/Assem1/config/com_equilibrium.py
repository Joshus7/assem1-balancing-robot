"""
Equilibrium (balance) pitch of the Assem1 robot as a function of arm angles.

Thin wrapper around robot_model.py (the single source of truth: current Assem1.urdf incl. ballast).
The old version of this file had its own copy of the URDF numbers (old wheel joints, no ballast)
and gave different answers - do not copy numbers here again.

    from com_equilibrium import equilibrium_pitch
    pitch_ref = equilibrium_pitch(arm1, arm2, arm3)      # radians, + = forward lean, same sign as the IMU pitch
"""
import math
from robot_model import G, sprung_params, equilibrium_pitch as _eq_pitch


def system_com(q1, q2, q3=0.0):
    """Mass and CoM of everything above the axle (body + arm + ballast), relative to the axle,
    base frame at pitch 0: returns (m, x_c, h)."""
    p = sprung_params((q1, q2, q3))
    return p['m'], p['x_c'], p['h']


def equilibrium_pitch(q1, q2, q3=0.0):
    """Body pitch (rad) at which the CoM is exactly above the wheel axle."""
    return _eq_pitch((q1, q2, q3))


def gravity_torque_at_pitch0(q1, q2, q3=0.0):
    """Torque (N.m) the wheels must supply to hold pitch = 0 with this arm pose."""
    m, x_c, _ = system_com(q1, q2, q3)
    return m * G * x_c


if __name__ == "__main__":
    print("arm1   arm2  | CoM-axle dx,dz (mm) | equilibrium pitch (deg) | torque @ pitch 0 (N.m)")
    for q1 in (0.0, -0.4, -0.75, -1.2):
        for q2 in (-0.5, 0.0, 0.65):
            m, x_c, h = system_com(q1, q2)
            print(f"{q1:+.2f} {q2:+.2f} | {x_c*1000:+6.1f} {h*1000:+6.1f}      | {math.degrees(equilibrium_pitch(q1, q2)):+8.1f}"
                  f"                | {gravity_torque_at_pitch0(q1, q2):+.3f}")
