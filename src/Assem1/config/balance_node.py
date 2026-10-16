#!/usr/bin/env python3
"""
balance_node.py - ROS 2 node that KEEPS THE ROBOT UPRIGHT (100 Hz, runs inside IMU callback).

Used by:   balance_controller.py (keyboard driving)   robot_balance_env.py (RL)   run_trained_model.py

Subscribes : /imu/data (sensor_msgs/Imu)     /odom (nav_msgs/Odometry)
Publishes  : /cmd_vel (geometry_msgs/Twist)  /arm_controller/commands (std_msgs/Float64MultiArray, ros2_control position)

High level inputs (thread safe setters):  set_speed(v_ref, yaw_rate)   set_arm_target(q)   set_lean(rad)
The arm is moved with a rate limit (so the centre of mass never jumps) and the controller gets the
CURRENT arm angles for the feed-forward lean (see balance_core.py).
"""
import math
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from balance_core import BalanceController
from robot_model import ARM_LIMITS, ARM_HOME as _ARM_HOME

ARM_JOINTS = ['link1', 'link2', 'link3']
ARM_HOME = np.array(_ARM_HOME)                   # camera pose: arm base to the sky, last link looking forward
ARM_MAX_RATE = 1.2                               # rad/s


def quat_to_pitch(x, y, z, w):
    s = 2.0 * (w * y - z * x)
    return math.copysign(math.pi / 2, s) if abs(s) >= 1.0 else math.asin(s)


class BalanceNode(Node):
    def __init__(self, name='balance_node', use_sim_time=True, use_ff=True, ctrl_kwargs=None):
        super().__init__(name, parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, use_sim_time)])
        self.lock = threading.RLock()
        self.ctrl = BalanceController(use_ff=use_ff, **(ctrl_kwargs or {}))

        self.pitch = 0.0; self.pitch_rate = 0.0; self.v = 0.0; self.yaw_rate = 0.0
        self.v_ref = 0.0; self.yaw_cmd = 0.0; self.lean_cmd = 0.0
        self.arm_q = ARM_HOME.copy(); self.arm_target = ARM_HOME.copy(); self.arm_dq = np.zeros(3)
        self.enabled = True
        self.imu_count = 0; self.odom_count = 0
        self._last_imu_t = None
        self.last_vcmd = 0.0

        self.create_subscription(Imu, '/imu/data', self._imu_cb, 50)
        self.create_subscription(Odometry, '/odom', self._odom_cb, 50)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_arm = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self._arm_dt = 0.02
        self.create_timer(self._arm_dt, self._arm_timer)   # sim-time timer

    # ------------------------------------------------------------------ setters (thread safe)
    def set_speed(self, v_ref, yaw_rate=0.0):
        with self.lock:
            self.v_ref = float(v_ref); self.yaw_cmd = float(yaw_rate)

    def set_arm_target(self, q):
        with self.lock:
            self.arm_target = np.clip(np.asarray(q, float), ARM_LIMITS[:, 0], ARM_LIMITS[:, 1])

    def set_lean(self, lean):
        with self.lock:
            self.lean_cmd = float(lean)

    def snap_arm(self, q):
        """teleport arm state to q (used on reset)"""
        with self.lock:
            self.arm_q = np.asarray(q, float).copy(); self.arm_target = self.arm_q.copy(); self.arm_dq[:] = 0
        self._publish_arm()

    def reset_controller(self):
        with self.lock:
            self.ctrl.reset(); self.v_ref = 0.0; self.yaw_cmd = 0.0; self.lean_cmd = 0.0
            self.ctrl.theta_ff = None

    def get_state(self):
        with self.lock:
            return dict(pitch=self.pitch, pitch_rate=self.pitch_rate, v=self.v, yaw_rate=self.yaw_rate,
                        arm_q=self.arm_q.copy(), arm_dq=self.arm_dq.copy(),
                        theta_ff=(self.ctrl.theta_ff if self.ctrl.theta_ff is not None else 0.0),
                        v_cmd=self.last_vcmd, imu_count=self.imu_count, odom_count=self.odom_count)

    # ------------------------------------------------------------------ callbacks
    def _odom_cb(self, msg):
        with self.lock:
            self.v = msg.twist.twist.linear.x
            self.yaw_rate = msg.twist.twist.angular.z
            self.odom_count += 1

    def _imu_cb(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        with self.lock:
            dt = 0.01 if self._last_imu_t is None else t - self._last_imu_t
            self._last_imu_t = t
            if dt <= 0.0:                     # sim was reset (time went back)
                dt = 0.01
            dt = min(max(dt, 0.002), 0.05)
            q = msg.orientation
            self.pitch = quat_to_pitch(q.x, q.y, q.z, q.w)
            self.pitch_rate = msg.angular_velocity.y
            self.imu_count += 1
            if not self.enabled:
                return
            v_cmd = self.ctrl.update(self.pitch, self.pitch_rate, self.v, self.v_ref, self.arm_q, dt,
                                     lean_cmd=self.lean_cmd)
            self.last_vcmd = v_cmd
            cmd = Twist()
            cmd.linear.x = float(v_cmd)
            cmd.angular.z = float(self.yaw_cmd)
            self.pub_cmd.publish(cmd)

    def _arm_timer(self):
        with self.lock:
            step = ARM_MAX_RATE * self._arm_dt
            new = self.arm_q + np.clip(self.arm_target - self.arm_q, -step, step)
            self.arm_dq = (new - self.arm_q) / self._arm_dt
            self.arm_q = new
        self._publish_arm()

    def _publish_arm(self):
        msg = Float64MultiArray()
        with self.lock:
            msg.data = [float(a) for a in self.arm_q]      # order = ARM_JOINTS (arm_controllers.yaml)
        self.pub_arm.publish(msg)

    def stop_robot(self):
        self.enabled = False
        self.pub_cmd.publish(Twist())


def main():
    """Headless: just keep the robot upright (started by robot_gazebo.launch.py)."""
    rclpy.init()
    node = BalanceNode('balance_node', use_sim_time=True, use_ff=True)
    node.snap_arm(ARM_HOME)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
