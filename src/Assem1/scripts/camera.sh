#!/bin/zsh
# Show the end-effector camera (Gazebo must be running: scripts/gazebo.sh).   Topic: /arm_camera/image_raw
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_env
source ~/new_ws/install/setup.zsh 2>/dev/null
echo "=== End-effector camera view (close the window to stop) ==="
ros2 run rqt_image_view rqt_image_view /arm_camera/image_raw
