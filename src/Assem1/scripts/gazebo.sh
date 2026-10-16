#!/bin/zsh
# Terminal 1: Gazebo (with GUI) + road with hills + Assem1, balance controller off (RL drives it)
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_env
source $CONDA_PREFIX/share/gazebo/setup.sh
source ~/new_ws/install/setup.zsh 2>/dev/null
pkill -9 gzserver gzclient 2>/dev/null
echo "=== Gazebo: Assem1 on the hill road (balance:=false, ready for RL training) ==="
echo "    NOTE: do not click Gazebo's camera/screenshot button - it crashes gzclient on this build."
ros2 launch Assem1 robot_gazebo.launch.py balance:=false 2>&1 | tee /tmp/assem1_gazebo.log
