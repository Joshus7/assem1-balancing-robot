#!/bin/zsh
# retrain.sh - retrain the RL lean policy for the current task (12 obs / 1 action).
#
#   ./retrain.sh stage1     fast simulator, no Gazebo (~3 min on an M3), then compares PID-only vs policy
#   ./retrain.sh stage2     Gazebo fine-tuning. Start Gazebo FIRST in another terminal:
#                               conda activate ros_env && source ~/new_ws/install/setup.zsh
#                               ros2 launch Assem1 robot_gazebo.launch.py
#                           then check the wiring once:  python3 check_setup.py   (must say ALL PASS)
#   ./retrain.sh run        drive the trained stage-2 policy in Gazebo
#
# Outputs (keep each .zip together with its _vecnorm.pkl): stage1_fast.*  stage2_gazebo.*  checkpoints/
set -e
cd "${0:A:h}"                                   # always run from this config/ folder (scripts import each other)
export KMP_DUPLICATE_LIB_OK=TRUE                # pip torch + conda numpy both ship libomp (macOS / RoboStack)

if [[ "$CONDA_DEFAULT_ENV" != "ros_env" ]]; then
  echo "Activate the RoboStack env first:  conda activate ros_env"; exit 1
fi

case "$1" in
  stage1)
    python3 train_ppo.py --env fast --steps ${STEPS:-1500000} --n-envs 8 --name stage1_fast
    python3 eval_policy.py --name stage1_fast --episodes 30
    ;;
  stage2)
    [[ -f stage1_fast.zip && -f stage1_fast_vecnorm.pkl ]] || { echo "Run ./retrain.sh stage1 first"; exit 1; }
    python3 train_ppo.py --env gazebo --steps ${STEPS:-150000} --load stage1_fast --lr 1e-4 --name stage2_gazebo
    ;;
  run)
    python3 run_trained_model.py --name ${NAME:-stage2_gazebo} ${V:+--v $V}
    ;;
  *)
    sed -n 2,13p "$0"; exit 1
    ;;
esac
