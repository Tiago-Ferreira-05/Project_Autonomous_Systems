# EKF-SLAM — Extended Kalman Filter Simultaneous Localization and Mapping

> **Autonomous Systems 2025/26** | Instituto Superior Técnico, DEEC  
> Project Code: `SKP` (Pioneer 3DX) / `SKT` (TurtleBot3)

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Requirements](#requirements)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Running the Project](#running-the-project)
- [TurtleBot3 Command Reference](#turtlebot3-command-reference)
- [Experimental Results](#experimental-results)
- [Team](#team)

---

## Overview

This project implements **EKF-SLAM** — a method for simultaneously estimating the trajectory of a mobile robot and the positions of landmarks in the environment, using an Extended Kalman Filter (EKF).

The robot maintains a joint state vector containing its own pose (position + orientation) and the estimated positions of all observed landmarks. As it moves and perceives the environment, both the robot pose and map are updated in real-time.

**Key capabilities:**
- Real-time trajectory and landmark estimation
- Fusion of wheel odometry (relative sensing) with fiducial marker observations (absolute sensing)
- Evaluated on rosbag data collected from Pioneer 3DX or TurtleBot3 robots
- Quantitative accuracy evaluation for both robot trajectory and landmark positions

---

## Algorithm

EKF-SLAM maintains and recursively updates a joint Gaussian belief:

```
State vector:   x = [x_r, y_r, θ_r, x_L1, y_L1, ..., x_Ln, y_Ln]^T
Covariance:     Σ (full joint covariance matrix)
```

The filter alternates between two steps:

**1. Prediction** — propagate state using the motion model (wheel odometry):
```
x̂ = f(x, u)        (nonlinear motion model)
Σ̂ = F·Σ·Fᵀ + Q    (covariance propagation with process noise Q)
```

**2. Update** — correct state using landmark observations (ARuCO / AprilTag markers):
```
z = h(x̂)           (expected observation from predicted state)
K = Σ̂·Hᵀ·(H·Σ̂·Hᵀ + R)⁻¹   (Kalman gain)
x = x̂ + K·(z_obs - z)        (state correction)
Σ = (I - K·H)·Σ̂              (covariance update)
```

New landmarks are initialized on first observation and added to the joint state.

---

## Requirements

### Hardware (Lab)
- **Robot:** Pioneer 3DX with Hokuyo URG-04LX-UG01 laser **or** TurtleBot3 Waffle Pi with RPLIDAR
- **Sensing:** Camera for ARuCO / AprilTag fiducial marker detection
- **Lab Network:** `deec-robots` (WiFi password: `shakeytherobot`)

### Software
- Ubuntu 22.04 LTS
- ROS 2 Humble Hawksbill
- Python 3.10+
- Python packages: `numpy`, `scipy`, `matplotlib`, `opencv-python`

### ROS 2 Dependencies

```bash
sudo apt update

# TurtleBot3 packages
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
sudo apt install ros-humble-teleop-twist-keyboard

# ArUco and vision packages
sudo apt install ros-humble-aruco-opencv
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-image-transport-plugins
```

---

## Repository Structure

```
ekf_slam/
├── README.md
├── ekf_slam/
│   ├── __init__.py
│   ├── ekf_slam.py         # Core EKF-SLAM algorithm
│   ├── motion_model.py     # Odometry-based motion model
│   ├── observation_model.py # Landmark observation model
│   └── landmark_init.py    # New landmark initialization
├── simulator/
│   ├── micro_simulator.py  # Synthetic data generator for validation
│   └── visualizer.py       # Real-time state visualization
├── ros2_nodes/
│   ├── ekf_slam_node.py    # ROS 2 node wrapping the EKF-SLAM
│   └── marker_detector.py  # ARuCO/AprilTag detection node
├── evaluation/
│   ├── trajectory_error.py # RMSE and ATE computation
│   └── landmark_error.py   # Landmark position error analysis
├── bags/                   # Place rosbag recordings here (gitignored)
├── results/                # Output plots and metrics
└── tests/
    └── test_ekf_slam.py
```

---

## Setup & Installation

### 1. Clone the repository

```bash
mkdir -p ~/ekf_slam_ws/src
cd ~/ekf_slam_ws/src
git clone <your-repo-url> ekf_slam
cd ~/ekf_slam_ws
```

### 2. Install Python dependencies

```bash
pip install numpy scipy matplotlib opencv-python
```

### 3. Build the ROS 2 workspace

```bash
cd ~/ekf_slam_ws
colcon build --symlink-install
source install/setup.bash
```

> Add `source ~/ekf_slam_ws/install/setup.bash` to your `~/.bashrc` to avoid sourcing manually on every terminal.

**Full recommended `~/.bashrc` additions:**

```bash
# Source ROS 2 Humble
source /opt/ros/humble/setup.bash

# Source project workspace
source ~/ekf_slam_ws/install/setup.bash

# Set TurtleBot3 model
export TURTLEBOT3_MODEL=waffle_pi

# Set ROS Domain ID — must match your specific robot's ID
export ROS_DOMAIN_ID=14
```

### 4. Camera Calibration (ArduCam)

The camera must be calibrated before running marker detection. The calibration file for the **ArduCam** (`mmal_service_16.1`) is provided below.

**Create the calibration file:**

```bash
mkdir -p ~/.ros/camera_info
nano ~/.ros/camera_info/mmal_service_16.1.yaml
```

**Paste the following content:**

```yaml
image_width: 320
image_height: 240
camera_name: arducam
camera_matrix:
  rows: 3
  cols: 3
  data: [245.57422, 0.       , 164.42725,
         0.       , 246.33530,  99.80013,
         0.       , 0.       ,   1.     ]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.036978, 0.008770, -0.041104, 0.012788, 0.000000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1., 0., 0.,
         0., 1., 0.,
         0., 0., 1.]
projection_matrix:
  rows: 3
  cols: 4
  data: [247.30991, 0.       , 168.40876, 0.,
         0.       , 239.61870,  89.40307, 0.,
         0.       , 0.       ,   1.     , 0.]
```

**Save and exit:** `Ctrl+X` → `Y` → `Enter`

> **Key parameters:**
> - **Resolution:** 320 × 240 px
> - **Focal lengths:** fx = 245.57, fy = 246.34 (pixels)
> - **Principal point:** cx = 164.43, cy = 99.80
> - **Distortion model:** Plumb Bob (radial + tangential)

---

## Running the Project

### Option A — Micro-Simulator (recommended for development)

Validate the algorithm on synthetic data before running on real hardware:

```bash
python3 simulator/micro_simulator.py
```

This generates a synthetic robot trajectory with known landmark positions and sensor noise, then runs EKF-SLAM and visualizes the estimated vs. ground-truth trajectory.

### Option B — Rosbag replay (recommended for reproducibility)

Record or obtain a rosbag from the Pioneer/TurtleBot3 robot, then replay offline:

```bash
# Replay the bag
ros2 bag play bags/<your_bag>

# In a separate terminal, run the EKF-SLAM node
ros2 run ekf_slam ekf_slam_node
```

### Option C — Live robot

> **Pioneer 3DX**

```bash
# On the robot (SSH in first)
ssh deec@10.16.140.[17..23]
ros2 launch p2os_bringup p2os_driver_launch.py

# On your laptop — set domain ID
export ROS_DOMAIN_ID=<robot_id>

# Launch EKF-SLAM
ros2 run ekf_slam ekf_slam_node
```

> **TurtleBot3**

```bash
# 1. SSH into the robot
ssh deec@10.16.140.[11..15]

# 2. Launch robot drivers (on the robot)
ros2 launch turtlebot3_bringup robot.launch.py

# 3. Start camera node at low-bandwidth resolution for SLAM stability (on the robot)
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"

# 4. On your laptop — set environment
export TURTLEBOT3_MODEL=waffle_pi
export ROS_DOMAIN_ID=<robot_id>

# 5. Launch EKF-SLAM
ros2 run ekf_slam ekf_slam_node
```

### Recording data from the robot

```bash
# Record all relevant topics for EKF-SLAM
ros2 bag record /aruco_detections /image_raw /odom /tf /tf_static /camera_info

# Copy bag from robot to laptop
scp -r deec@10.16.140.<id>:/<bag_dir> .
```

### Inspecting files with `cat`

```bash
# View a Python source file
cat ekf_slam/ekf_slam.py

# View the contents of a rosbag metadata file
cat bags/<your_bag>/metadata.yaml

# View evaluation results
cat results/trajectory_error.txt

# Concatenate multiple result files into one
cat results/exp_01.txt results/exp_02.txt > results/all_experiments.txt

# Preview the first N lines of a large log file
cat logs/run.log | head -50

# Preview the last N lines of a log file
cat logs/run.log | tail -50

# Count lines in a results file
cat results/traj_est.csv | wc -l

# Search for a keyword inside a file
cat ekf_slam/ekf_slam.py | grep "def "
```

---

## Evaluation

Run the evaluation scripts after an experiment to compute quantitative metrics:

```bash
# Trajectory accuracy (RMSE, Absolute Trajectory Error)
python3 evaluation/trajectory_error.py --estimated results/traj_est.csv --groundtruth results/traj_gt.csv

# Landmark position accuracy
python3 evaluation/landmark_error.py --estimated results/lm_est.csv --groundtruth results/lm_gt.csv
```

Metrics reported:
- **RMSE** of robot (x, y, θ) over time
- **Absolute Trajectory Error (ATE)**
- **Mean landmark position error** per landmark

---

## TurtleBot3 Command Reference

> Quick-reference for environment setup, robot operations, SLAM execution, and data analysis on **TurtleBot3 with ROS 2 Humble**.

### Environment Setup (Local Workstation)

```bash
# Always run before any installation
sudo apt update

# TurtleBot3 and control packages
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
sudo apt install ros-humble-teleop-twist-keyboard

# ArUco and vision packages
sudo apt install ros-humble-aruco-opencv
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-image-transport-plugins
```

### Robot Operations (SSH — on the Raspberry Pi)

```bash
# Remote login (replace with your robot's IP)
ssh deec@10.16.140.14

# Hardware bringup
ros2 launch turtlebot3_bringup robot.launch.py

# Camera node — low-bandwidth resolution for SLAM stability
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"
```

### SLAM Execution (Workstation)

```bash
# Movement control
ros2 run turtlebot3_teleop teleop_keyboard

# ArUco landmark tracker — with internal topic remapping
ros2 run aruco_opencv aruco_tracker_autostart --ros-args \
  -r /camera/image_raw:=/image_raw \
  -r /camera/camera_info:=/camera_info

# Manual camera_info publication (if calibration file is not loaded automatically)
# Matrix format: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
ros2 topic pub /camera_info sensor_msgs/msg/CameraInfo "{
  header: {frame_id: 'camera_link'},
  height: 240, width: 320, distortion_model: 'plumb_bob',
  d: [0.0, 0.0, 0.0, 0.0, 0.0],
  k: [300.0, 0.0, 160.0, 0.0, 300.0, 120.0, 0.0, 0.0, 1.0],
  r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  p: [300.0, 0.0, 160.0, 0.0, 0.0, 300.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}"

# Data recording
ros2 bag record /aruco_detections /image_raw /odom /tf /tf_static /camera_info
```

### Data Analysis & Visualization

```bash
# Bag file inspection
ros2 bag info <bag_folder_name>

# Standard playback
ros2 bag play <bag_folder_name>

# Advanced playback — looping with rate control
ros2 bag play <bag_folder_name> --loop --rate 1.0

# Visualization (set Fixed Frame to: odom)
rviz2
ros2 run rqt_image_view rqt_image_view

# Topic monitoring
ros2 topic echo /aruco_detections
ros2 topic info /aruco_detections
```

---

## Experimental Results

Results and plots are saved in `results/`. Each experiment should be documented with:

| Experiment | Robot | Scenario | # Landmarks | Trajectory RMSE | Landmark RMSE |
|------------|-------|----------|-------------|-----------------|---------------|
| exp_01     | Pioneer 3DX | Lab corridor | 5 | — | — |
| exp_02     | TurtleBot3  | Open room    | 8 | — | — |

> Fill in this table as experiments are completed.

---

## Project Assessment

Per course guidelines (Autonomous Systems 2025/26):

| Milestone | Date |
|-----------|------|
| Weekly progress presentations (×5) | From 27 April 2026 |
| Project report & code submission | 5 June 2026 |
| Project discussions | 8–9 June 2026 |

**Report format:** 6-page IEEE paper template.

---

## Team

| Name | IST ID |
|------|--------|
| — | — |
| — | — |
| — | — |
| — | — |

---

## References

1. Thrun, S., Burgard, W., Fox, D. — *Probabilistic Robotics*, MIT Press, 2005
2. Durrant-Whyte, H., Bailey, T. — "Simultaneous localisation and mapping (SLAM): Part I", *IEEE Robotics & Automation Magazine*, 2006
3. ROS 2 Humble Documentation — https://docs.ros.org/en/humble/
4. ARuCO Marker Detection — OpenCV Documentation

---

*Instituto Superior Técnico — Departamento de Engenharia Electrotécnica e de Computadores*
