#!/bin/zsh
# Watch a trained policy in Gazebo (Gazebo must be running: scripts/gazebo.sh).   Usage: run_model.sh [model_name] [speed]
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_env
source ~/new_ws/install/setup.zsh 2>/dev/null
cd ~/new_ws/src/Assem1/config
export KMP_DUPLICATE_LIB_OK=TRUE
NAME=${1:-stage2_gazebo}
echo "=== Running trained model '$NAME' on the hill road (Ctrl+C to stop) ==="
if [ -n "$2" ]; then python3 -u run_trained_model.py --name $NAME --v $2; else python3 -u run_trained_model.py --name $NAME; fi
