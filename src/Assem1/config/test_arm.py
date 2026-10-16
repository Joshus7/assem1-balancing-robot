#!/usr/bin/env python3
"""test_arm.py - move the arm back and forth (Gazebo must be running).  python3 test_arm.py"""
import time
import rclpy
from std_msgs.msg import Float64MultiArray

rclpy.init()
node = rclpy.create_node('test_arm_movement')
pub = node.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
time.sleep(1.0)
for q in ([0.0, 0.0, 0.0], [0.0, -0.6, 0.97], [0.3, -1.0, 1.4], [0.0, -0.6, 0.97]):
    print('[*] arm ->', q)
    pub.publish(Float64MultiArray(data=q))
    time.sleep(2.0)
node.destroy_node(); rclpy.shutdown()
