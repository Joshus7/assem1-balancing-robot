#!/bin/zsh
# Terminal 2: PPO training in Gazebo, 1000 iterations, starting from the fast-sim policy
source ~/miniforge3/etc/profile.d/conda.sh && conda activate ros_env
source ~/new_ws/install/setup.zsh 2>/dev/null
cd ~/new_ws/src/Assem1/config
export KMP_DUPLICATE_LIB_OK=TRUE
echo "=== RL training on the hill road: 1000 iterations (Ctrl+C = stop and save) ==="
python3 -u train_ppo.py --env gazebo --iterations 1000 --load stage1_fast --lr 1e-4 --name stage2_gazebo 2>&1 | tee ~/new_ws/src/Assem1/config/stage2_train.log
