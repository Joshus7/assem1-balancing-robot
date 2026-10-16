#!/usr/bin/env python3
"""Launch Gazebo (classic, with GUI) + robot_state_publisher + spawn Assem1 + arm controller + balance controller.
   ros2 launch Assem1 robot_gazebo.launch.py                    robot balances by itself
   ros2 launch Assem1 robot_gazebo.launch.py balance:=false     for RL training / balance_controller.py
   ros2 launch Assem1 robot_gazebo.launch.py gui:=false         no Gazebo window
   ros2 launch Assem1 robot_gazebo.launch.py world:=empty       flat empty world (default: road with hills)"""
import os
import re
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, RegisterEventHandler,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('Assem1')
    config_dir = os.path.join(pkg, 'config')
    urdf_path = os.path.join(pkg, 'urdf', 'Assem1.urdf')
    with open(urdf_path, 'r') as f:
        urdf = f.read()
    # Gazebo classic cannot resolve package:// by itself -> use absolute file:// paths
    urdf_abs = urdf.replace('package://Assem1/', 'file://' + pkg + '/')
    urdf_abs = urdf_abs.replace('ASSEM1_CONFIG_DIR', config_dir)
    # gazebo_ros2_control passes the URDF on a command line and cannot parse XML comments -> strip them
    urdf_abs = re.sub(r'<!--.*?-->', '', urdf_abs, flags=re.S)
    spawn_file = '/tmp/Assem1_spawned.urdf'
    with open(spawn_file, 'w') as f:
        f.write(urdf_abs)

    worlds = {'road': os.path.join(pkg, 'worlds', 'road_hills.world'),
              'empty': os.path.join(get_package_share_directory('gazebo_ros'), 'worlds', 'empty.world')}
    world_arg = LaunchConfiguration('world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')),
        launch_arguments={'verbose': 'true', 'pause': 'false', 'gui': LaunchConfiguration('gui'),
                          'world': PythonExpression(["{'road': '", worlds['road'], "', 'empty': '", worlds['empty'], "'}.get('", world_arg, "', '", world_arg, "')"])}.items())

    rsp = Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
               parameters=[{'robot_description': urdf_abs, 'use_sim_time': True}])

    # entity name MUST stay 'Assem1' (your old env used 'bot' -> reset never worked)
    spawn = Node(package='gazebo_ros', executable='spawn_entity.py', output='screen',
                 arguments=['-entity', 'Assem1', '-file', spawn_file, '-x', '0', '-y', '0', '-z', '0.03'])

    jsb = Node(package='controller_manager', executable='spawner', output='screen',
               arguments=['joint_state_broadcaster', '--controller-manager-timeout', '30'])
    arm = Node(package='controller_manager', executable='spawner', output='screen',
               arguments=['arm_controller', '--controller-manager-timeout', '30'])

    # without a controller a two-wheel robot ALWAYS falls over -> start the balance node once the arm is ready
    balance = ExecuteProcess(cmd=[sys.executable, os.path.join(config_dir, 'balance_node.py')],
                             cwd=config_dir, output='screen', condition=IfCondition(LaunchConfiguration('balance')),
                             additional_env={'KMP_DUPLICATE_LIB_OK': 'TRUE'})

    return LaunchDescription([
        DeclareLaunchArgument('balance', default_value='true', description='start the balance controller'),
        DeclareLaunchArgument('gui', default_value='true', description='open the Gazebo window'),
        DeclareLaunchArgument('world', default_value='road', description='road (hills) | empty | path to a .world file'),
        gazebo, rsp,
        TimerAction(period=4.0, actions=[spawn]),
        RegisterEventHandler(OnProcessExit(target_action=spawn, on_exit=[jsb, arm])),
        RegisterEventHandler(OnProcessExit(target_action=arm, on_exit=[balance])),
    ])
