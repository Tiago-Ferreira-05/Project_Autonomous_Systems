# TurtleBot3 ROS 2 Humble — Command Reference

> Quick-reference guide for environment setup, robot operations, SLAM execution, and data analysis.

---

## Table of Contents

- [1. Environment Setup \& Installation](#1-environment-setup--installation)
- [2. Robot Operations (SSH)](#2-robot-operations-ssh)
- [3. SLAM Execution \& Data Acquisition](#3-slam-execution--data-acquisition)
- [4. Data Analysis \& Visualization](#4-data-analysis--visualization)

---

## 1. Environment Setup & Installation

> Run on the **local workstation** (e.g., `Tiago-HP-Laptop`).

### Update Package Lists

Always execute before any installation.

```bash
sudo apt update
```

### Install TurtleBot3 and Control Packages

```bash
sudo apt install ros-humble-dynamixel-sdk
sudo apt install ros-humble-turtlebot3-msgs
sudo apt install ros-humble-turtlebot3
sudo apt install ros-humble-teleop-twist-keyboard
```

### Install ArUco and Vision Packages

```bash
sudo apt install ros-humble-aruco-opencv
sudo apt install ros-humble-image-transport
sudo apt install ros-humble-image-transport-plugins
```

### `~/.bashrc` Configuration

Add the following lines to your `~/.bashrc`:

```bash
# Set TurtleBot model
export TURTLEBOT3_MODEL=waffle_pi

# Set ROS Domain ID (must match your specific robot ID)
export ROS_DOMAIN_ID=14

# Source ROS 2 Humble setup
source /opt/ros/humble/setup.bash
```

---

## 2. Robot Operations (SSH)

> Execute these commands on the **Raspberry Pi**.

### Remote Login

Replace the IP address with your target robot's IP.

```bash
ssh deec@10.16.140.14
```

### Hardware Bringup

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

### Sensor Initialization

Uses low-bandwidth resolution for SLAM stability.

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"
```

---

## 3. SLAM Execution & Data Acquisition

> Workstation-side operations.

### Movement Control

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

### Landmark Tracker Initialization

Internal topic remapping for the ArUco tracker.

```bash
ros2 run aruco_opencv aruco_tracker_autostart --ros-args \
  -r /camera/image_raw:=/image_raw \
  -r /camera/camera_info:=/camera_info
```

### Manual Intrinsic Matrix (K) Publication

**Matrix format:** `[fx, 0, cx, 0, fy, cy, 0, 0, 1]`

```bash
ros2 topic pub /camera_info sensor_msgs/msg/CameraInfo "{
  header: {frame_id: 'camera_link'},
  height: 240, width: 320, distortion_model: 'plumb_bob',
  d: [0.0, 0.0, 0.0, 0.0, 0.0],
  k: [300.0, 0.0, 160.0, 0.0, 300.0, 120.0, 0.0, 0.0, 1.0],
  r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
  p: [300.0, 0.0, 160.0, 0.0, 0.0, 300.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}"
```

### Data Recording

```bash
ros2 bag record /aruco_detections /image_raw /odom /tf /tf_static /camera_info
```

---

## 4. Data Analysis & Visualization

> Offline processing and verification.

### Bag File Inspection

```bash
# View bag metadata
ros2 bag info <bag_folder_name>

# Standard playback
ros2 bag play <bag_folder_name>

# Advanced playback — looping with rate control
ros2 bag play <bag_folder_name> --loop --rate 1.0
```

### Visualization

```bash
rviz2
ros2 run rqt_image_view rqt_image_view
```

> **Fixed Frame:** `odom`

### Data Monitoring

```bash
ros2 topic echo /aruco_detections
ros2 topic info /aruco_detections
```
