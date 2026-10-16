#!/usr/bin/env python3
"""Show Assem1 in RViz2 with sliders for the joints (ROS 2 replacement of the old ROS 1 display.launch).
   ros2 launch Assem1 display.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('Assem1')
    with open(os.path.join(pkg, 'urdf', 'Assem1.urdf'), 'r') as f:
        urdf = f.read()

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
               parameters=[{'robot_description': urdf}])
    jsp_gui = Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui', output='screen')
    rviz = Node(package='rviz2', executable='rviz2', output='screen')

    return LaunchDescription([rsp, jsp_gui, rviz])
