#!/usr/bin/env python3
"""
check_setup.py - 1-minute test that Gazebo + URDF are wired correctly.  Run with Gazebo already running:
     python3 check_setup.py
Checks: IMU frame (pitch of the balanced robot ~ feed-forward angle), odom, both wheels turn the SAME way,
reset service.  Prints PASS / FAIL for each.
"""
import time, math, sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_srvs.srv import Empty
from balance_node import BalanceNode, quat_to_pitch, ARM_HOME
from robot_model import sprung_params, equilibrium_pitch, ARM_LIMITS
import threading
from rclpy.executors import MultiThreadedExecutor

rclpy.init()
node = BalanceNode('check_setup', use_sim_time=True)
ex = MultiThreadedExecutor(num_threads=3); ex.add_node(node)
threading.Thread(target=ex.spin, daemon=True).start()
ok = True
def report(name, passed, info=''):
    global ok; ok &= bool(passed); print('[%s] %s %s' % ('PASS' if passed else 'FAIL', name, info))

t0 = time.time()
while time.time() - t0 < 15 and (node.get_state()['imu_count'] < 20 or node.get_state()['odom_count'] < 20):
    time.sleep(0.2)
s = node.get_state()
report('IMU messages on /imu/data', s['imu_count'] >= 20, '(%d msgs)' % s['imu_count'])
report('odometry messages on /odom', s['odom_count'] >= 20, '(%d msgs)' % s['odom_count'])
if s['imu_count'] < 20: sys.exit(1)

cli = node.create_client(Empty, '/reset_world')
report('/reset_world service', cli.wait_for_service(timeout_sec=5.0))
if cli.service_is_ready():
    f = cli.call_async(Empty.Request()); 
    while not f.done(): time.sleep(0.05)
time.sleep(0.5)
node.reset_controller(); node.snap_arm(ARM_HOME)
time.sleep(4.0)       # PID balances the robot
s = node.get_state()
err = abs(s['pitch'] - s['theta_ff'])
report('robot balances by itself (PID + FF)', err < 0.12 and abs(s['pitch']) < 0.6,
       'pitch=%.1f deg, expected balance angle (feed-forward)=%.1f deg, v=%.2f' % (math.degrees(s['pitch']), math.degrees(s['theta_ff']), s['v']))
if abs(s['pitch']) > 0.9:
    print('   -> pitch near 90 deg means the IMU frame is rotated. Check that imu_link sensor is in the URDF you spawned.')

# wheel direction test: drive 0.3 m/s ; yaw rate must stay near zero and speed positive
node.set_speed(0.3, 0.0); time.sleep(3.0)
s = node.get_state()
report('drives straight forward (both wheels same direction)', s['v'] > 0.1 and abs(s['yaw_rate']) < 0.3,
       'v=%.2f m/s, yaw rate=%.2f rad/s' % (s['v'], s['yaw_rate']))
node.set_speed(-0.3, 0.0); time.sleep(3.0)
s = node.get_state()
report('drives backward', s['v'] < -0.1, 'v=%.2f' % s['v'])
node.set_speed(0.0, 0.0)

# arm test: move arm through its range, robot must stay up
fell = False
for q in ([0.3, -1.0, 1.4], [-0.4, -0.2, 0.5], list(ARM_HOME)):
    node.set_arm_target(q); time.sleep(2.0)
    if abs(node.get_state()['pitch']) > 0.8: fell = True
report('robot stays up while arm moves (look at Gazebo: arm should move)', not fell,
       'pitch now %.1f deg, ff %.1f deg' % (math.degrees(node.get_state()['pitch']), math.degrees(node.get_state()['theta_ff'])))
print('\nOVERALL:', 'ALL PASS - go on with training' if ok else 'SOMETHING FAILED - read the FAIL lines above')
node.stop_robot(); ex.shutdown(); rclpy.shutdown()
