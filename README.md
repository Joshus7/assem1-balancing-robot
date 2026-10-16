# Assem1: Self-Balancing Two-Wheel Robot with a 3-Link Arm (ROS 2 + Gazebo + Reinforcement Learning)

Assem1 is a two-wheeled self-balancing robot (an inverted pendulum on wheels) with a 3-joint arm and a camera on the arm's tip. I designed it in SolidWorks, exported it to URDF, and simulated it in Gazebo with ROS 2. A classical controller keeps the robot upright, and a PPO reinforcement-learning policy runs on top of it. The policy helps the robot follow speed commands on a road with hills while the arm moves around.

Everything runs natively on an Apple Silicon Mac (M3) through RoboStack/conda. There's no Docker and no Linux VM.

---

## What it does

- **Balances on two wheels.** It stays upright on flat ground and on 3° and 5° hills.
- **Moves its arm while balancing.** When the arm moves, the robot's centre of mass shifts. The controller works out the new balance angle from the arm joint angles and leans to match it, so the robot doesn't fall.
- **Learns to drive better.** A PPO policy adds a small extra lean (±0.15 rad) on top of the controller. With it, the robot reaches the commanded speed faster and overshoots less.
- **Sees with its arm.** A camera on the end of the arm publishes images on `/arm_camera/image_raw`.

## Architecture

```
            25 Hz                                  100 Hz
 ┌─────────────────────┐   lean offset   ┌──────────────────────────────────────┐   /cmd_vel   ┌──────────────┐
 │ PPO policy (SB3)    │ ──────────────▶ │ Arm feed-forward  (CoM over axle)    │ ───────────▶ │ Gazebo       │
 │ 12 obs → 1 action   │                 │ Velocity PI       (outer loop)       │              │ diff-drive   │
 └─────────────────────┘                 │ Pitch PD          (inner loop)       │ ◀─────────── │ IMU, odom,   │
            ▲                            └──────────────────────────────────────┘  /imu /odom  │ arm, camera  │
            └──────────── pitch, rate, speed, speed cmd, arm angles/velocities ◀───────────────└──────────────┘
```

1. **Arm feed-forward.** From the arm joint angles, it computes the pitch at which the centre of mass of the body, arm and ballast sits exactly over the wheel axle.
2. **Velocity PI (outer loop).** If the robot is going too fast, it leans back. If it's going too slow, it leans forward.
3. **Pitch PD (inner loop).** It outputs a wheel *acceleration*, and that acceleration is integrated into the wheel velocity command.
4. **PPO policy.** It adds a learned lean correction on top of the three loops above.

### Two-stage training

| Stage | Simulator | Speed | Purpose |
|---|---|---|---|
| 1 | Custom fast physics model (`fast_sim.py`), 8 parallel envs | ~1000+ steps/s, a few minutes | Learn the basics with domain randomisation: mass ±15%, CoM error, motor lag, command delay, sensor noise, pushes, random hills |
| 2 | Gazebo with GUI | real-time | Fine-tune the stage-1 policy in the full physics simulator |

Both environments share the same observation, action and reward (`task_common.py`). That lets a policy trained in the fast simulator load straight into Gazebo for fine-tuning.

## Repository layout

```
new_ws/
├── README.md
└── src/
    ├── requirements.txt            # pip packages for RL (gymnasium, stable-baselines3, torch)
    └── Assem1/                     # ROS 2 ament_cmake package
        ├── urdf/Assem1.urdf        # robot: body, wheels, ballast, 3-link arm, IMU, camera, Gazebo plugins
        ├── meshes/*.STL            # SolidWorks meshes
        ├── launch/
        │   ├── robot_gazebo.launch.py   # Gazebo + spawn + controllers (+ balance node)
        │   └── display.launch.py        # RViz2 with joint sliders
        ├── worlds/
        │   ├── make_road_world.py  # generates the road with hills
        │   └── road_hills.world
        ├── config/
        │   ├── robot_model.py      # mass/CoM model, single source of truth from the URDF
        │   ├── balance_core.py     # cascade controller (pure Python)
        │   ├── balance_node.py     # ROS 2 node running the controller at 100 Hz
        │   ├── fast_sim*.py        # fast physics simulator + Gym env (stage 1)
        │   ├── robot_balance_env.py# Gazebo Gym env (stage 2)
        │   ├── task_common.py      # shared obs / action / reward
        │   ├── train_ppo.py        # PPO training
        │   ├── eval_policy.py      # PID-only vs. policy comparison
        │   ├── run_trained_model.py# drive a trained policy in Gazebo
        │   ├── check_setup.py      # 1-minute wiring test (IMU, odom, wheels, reset)
        │   ├── arm_controllers.yaml
        │   └── stage1_fast.zip / stage1_long.zip / stage2_gazebo.zip  (+ *_vecnorm.pkl)
        └── scripts/                # one-command helpers
            ├── gazebo.sh   train.sh   run_model.sh   camera.sh
```

## Setup (macOS Apple Silicon; Linux works the same with apt or RoboStack)

**Tested with:** macOS (M3), Miniforge, RoboStack ROS 2 Humble, Gazebo Classic 11.15, Python 3.12.

### 1. Install ROS 2 Humble and Gazebo with RoboStack

```bash
# Miniforge: https://github.com/conda-forge/miniforge
conda create -n ros_env -c conda-forge -c robostack-staging python=3.12 ros-humble-desktop
conda activate ros_env
conda install -c conda-forge -c robostack-staging \
    compilers cmake pkg-config make ninja colcon-common-extensions \
    ros-humble-gazebo-ros-pkgs ros-humble-gazebo-ros2-control \
    ros-humble-ros2-control ros-humble-ros2-controllers \
    ros-humble-joint-state-publisher-gui ros-humble-rqt-image-view
```

### 2. Install the RL packages

```bash
pip install -r src/requirements.txt
```

### 3. Clone and build

```bash
mkdir -p ~/new_ws && cd ~/new_ws
git clone https://github.com/Joshus7/assem1-balancing-robot.git .
colcon build
source install/setup.zsh        # or setup.bash
```

> The install step **copies** files, so run `colcon build` again after you edit the URDF, launch or config files.

The helper scripts assume the workspace is at `~/new_ws` and Miniforge is at `~/miniforge3`. If yours are somewhere else, change the paths at the top of each `scripts/*.sh` file.

## Running it

Open two terminals.

**Terminal 1: Gazebo + robot on the hill road**
```bash
src/Assem1/scripts/gazebo.sh
```

**Terminal 2: pick one**
```bash
src/Assem1/scripts/run_model.sh                    # watch the trained policy drive (default: stage2_gazebo)
src/Assem1/scripts/run_model.sh stage1_long 0.4    # another model, fixed speed 0.4 m/s
src/Assem1/scripts/train.sh                        # fine-tune in Gazebo (Ctrl+C saves the model)
src/Assem1/scripts/camera.sh                       # view the arm camera
```

Other useful commands:
```bash
ros2 launch Assem1 robot_gazebo.launch.py                   # robot balances by itself (PID only)
ros2 launch Assem1 robot_gazebo.launch.py world:=empty      # flat world
ros2 launch Assem1 display.launch.py                        # RViz2 with joint sliders
cd src/Assem1/config && python3 check_setup.py              # wiring test, must print ALL PASS
```

### Retraining from scratch
```bash
cd src/Assem1/config
./retrain.sh stage1      # fast simulator, about 3 minutes on an M3, then prints PID-only vs. policy
./retrain.sh stage2      # Gazebo fine-tuning (start Gazebo with balance:=false first)
```
Always keep each `<name>.zip` together with its `<name>_vecnorm.pkl`. The policy needs both files.

## Problems I hit and how I fixed them

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | The robot froze at 0° pitch, then the physics went NaN | The old `joint_pose_trajectory` plugin **teleported the whole model** (`SetWorldPose`) on every arm command | Moved the arm to `gazebo_ros2_control` with a position interface, so it now moves only the arm joints |
| 2 | The robot was never reset after a fall during training | The env reset an entity called `bot`, but the robot was spawned as `Assem1`, and the set-state plugin wasn't loaded | Reset with the `/reset_world` service and kept the entity name fixed as `Assem1` |
| 3 | Balance was sluggish and the odometry was wrong | The diff-drive plugin had a wheel diameter of 0.10 m (the real wheel is 0.16 m), and its acceleration limit was far too low to catch a fall | Measured the wheel from the STL, fixed the diameter, and removed the acceleration limit |
| 4 | The PID only worked "by accident" | It sent a wheel *velocity* proportional to tilt, but an inverted pendulum needs axle *acceleration* | Rebuilt it as a cascade: pitch PD → acceleration → integrated into the velocity command. Got the gain structure from LQR, then tuned the gains with delay, noise and ±15% mass error |
| 5 | The robot fell whenever the arm moved | Moving the arm shifts the centre of mass | Added arm feed-forward to compute the balance angle from the joint angles, and rate-limited the arm |
| 6 | Python crashed on macOS with `OMP Error #179` | pip `torch` and conda `numpy` each ship their own OpenMP runtime | Import `torch` before `numpy` and set `KMP_DUPLICATE_LIB_OK=TRUE` |
| 7 | Gazebo couldn't find the meshes, and `gazebo_ros2_control` failed to parse the URDF | Gazebo Classic doesn't resolve `package://`, and comments in the URDF break the plugin's command-line parsing | The launch file rewrites the paths to absolute `file://` paths and strips XML comments before spawning |
| 8 | RL training in Gazebo was unstable and slow | Step time varied randomly (10–30 ms), and training from scratch in Gazebo is slow | Fixed the step at 0.04 s of *simulation* time. Pre-trained in a fast custom simulator with domain randomisation, then fine-tuned in Gazebo |
| 9 | Two controllers fought each other | The balance node and the RL env both published `/cmd_vel` | Added a `balance:=false` launch flag, and the env refuses to start if another node is already publishing `/cmd_vel` |
| 10 | Two copies of the robot model gave different equilibrium angles | Mass and CoM numbers were copied into several files | Made `robot_model.py` the single source of truth that every script imports |
| 11 | Gazebo's GUI crashed | Clicking the screenshot/camera button crashes `gzclient` on this macOS build | Don't use that button. Use `camera.sh` (rqt_image_view) for camera images instead |

## Tech stack
ROS 2 Humble · Gazebo Classic 11 · gazebo_ros2_control · Python · Gymnasium · Stable-Baselines3 (PPO) · PyTorch · SolidWorks → URDF · RoboStack/conda on macOS

## Author
Sayala Joshua
