#!/usr/bin/env python3
"""
balance_controller.py - drive the robot with the keyboard while it balances (PID + arm feed-forward).
Terminal 1: ros2 launch Assem1 robot_gazebo.launch.py
Terminal 2: python3 balance_controller.py            (click in this terminal, then press keys)

  W / S : faster forward / backward (steps of 0.1 m/s)      SPACE : stop
  A / D : turn left / right                                 X     : turn off turning
  Arm   : 1/2 = link1 -/+   3/4 = link2 -/+   5/6 = link3 -/+   (steps of 0.15 rad, rate limited)
  P     : arm to camera pose (looks forward)     O : arm straight up     Q : quit
"""
import sys, select, termios, tty, threading
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from balance_node import BalanceNode, ARM_HOME

LIM_V = 0.8
KEYS_ARM = {'1': (0, -1), '2': (0, 1), '3': (1, -1), '4': (1, 1), '5': (2, -1), '6': (2, 1)}


def main():
    rclpy.init()
    node = BalanceNode('balance_controller', use_sim_time=True, use_ff=True)
    ex = MultiThreadedExecutor(num_threads=3); ex.add_node(node)
    th = threading.Thread(target=ex.spin, daemon=True); th.start()

    old = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    v = 0.0; w = 0.0
    arm = ARM_HOME.copy()
    node.set_arm_target(arm)
    print(__doc__)
    try:
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1).lower()
                if k == 'w': v = min(LIM_V, v + 0.1)
                elif k == 's': v = max(-LIM_V, v - 0.1)
                elif k == 'a': w = min(1.5, w + 0.3)
                elif k == 'd': w = max(-1.5, w - 0.3)
                elif k == 'x': w = 0.0
                elif k == ' ': v = 0.0; w = 0.0
                elif k in KEYS_ARM:
                    j, sgn = KEYS_ARM[k]; arm[j] += 0.15 * sgn; arm = np.clip(arm, -1.57, 1.57)
                elif k == 'p': arm = ARM_HOME.copy()
                elif k == 'o': arm = np.array([0.0, 0.0, 0.0])
                elif k == 'q' or k == '\x03': break
                node.set_speed(v, w); node.set_arm_target(arm)
                s = node.get_state()
                print('v_ref=%.1f yaw=%.1f arm=%s | pitch=%.1f deg  v=%.2f  ff=%.1f deg' % (
                    v, w, np.round(arm, 2), np.degrees(s['pitch']), s['v'], np.degrees(s['theta_ff'])))
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.set_speed(0, 0)
        ex.shutdown(); rclpy.shutdown()

if __name__ == '__main__':
    main()
