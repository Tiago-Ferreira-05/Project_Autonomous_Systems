# TurtleBot3 ROS 2 Humble — Command Reference

> Quick-reference guide for environment setup, robot operations, SLAM execution, and data analysis.

---

## Table of Contents

- [1. Environment Setup \& Installation](#1-environment-setup--installation)
- [2. Robot Operations (SSH)](#2-robot-operations-ssh)
- [3. SLAM Execution \& Data Acquisition](#3-slam-execution--data-acquisition)
- [3A. Camera Calibration (ArduCam)](#3a-camera-calibration-arducam)
- [4. Visualization \& Data Monitoring](#4-visualization--data-monitoring)
- [5. Offline Analysis \& Bag Playback](#5-offline-analysis--bag-playback)

---

## 1. Environment Setup & Installation

> Run on the **local workstation** only. These steps are one-time installations.

### 1.1 Update Package Lists

Always run this before installing anything to ensure you get the latest package versions.

```bash
sudo apt update
```

### 1.2 Install TurtleBot3 Packages

The `dynamixel-sdk` drives the servo motors. `turtlebot3-msgs` provides the custom ROS 2 message types. `turtlebot3` is the full driver stack. `teleop-twist-keyboard` enables keyboard teleoperation. All four can be installed in sequence.

```bash
sudo apt install ros-humble-dynamixel-sdk && sudo apt install ros-humble-turtlebot3-msgs && sudo apt install ros-humble-turtlebot3 && sudo apt install ros-humble-teleop-twist-keyboard
```

### 1.3 Install ArUco and Vision Packages

`aruco-opencv` handles fiducial marker detection. `image-transport` and its plugins manage efficient compressed image streaming over ROS 2 topics. All three can be installed in sequence.

```bash
sudo apt install ros-humble-aruco-opencv && sudo apt install ros-humble-image-transport && sudo apt install ros-humble-image-transport-plugins
```

### 1.4 `~/.bashrc` Configuration

These environment variables must be set in every terminal session. Adding them to `~/.bashrc` ensures they load automatically on startup. Open the file for editing:

```bash
nano ~/.bashrc
```

Add the following three lines at the bottom of the file, then save and exit with `Ctrl+X` → `Y` → `Enter`.

```bash
source /opt/ros/humble/setup.bash
export TURTLEBOT3_MODEL=waffle_pi
export ROS_DOMAIN_ID=14
```

> `ROS_DOMAIN_ID` isolates your ROS 2 network traffic from other robots in the lab. It **must match** the ID of your specific robot — IDs are assigned per physical unit.

After saving, apply the changes to your current terminal without reopening it:

```bash
source ~/.bashrc
```

Verify the file contents at any time:

```bash
cat ~/.bashrc
```

---

## 2. Robot Operations (SSH)

> Execute these commands **on the Raspberry Pi** aboard the robot. Open two separate SSH terminal tabs — one for bringup, one for the camera — and keep both running throughout the session.

### 2.1 Remote Login

SSH into the robot. Replace the last octet with your robot's assigned number (e.g. `11`–`15`).

```bash
ssh deec@10.16.140.14
```

### 2.2 Hardware Bringup

Launches the full robot driver stack: Dynamixel motor controllers, IMU, and base sensor interfaces. **Must be running before any other robot node.** Run this in the first SSH tab.

```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

### 2.3 Camera Node

Starts the V4L2 camera driver and publishes frames to `/image_raw`. Resolution is fixed at 320×240 px to reduce network bandwidth and keep SLAM latency low. Run this in the second SSH tab.

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"
```

---

## 3. SLAM Execution & Data Acquisition

> Run on the **local workstation**. The robot must already have bringup and camera running (Section 2) before proceeding.

### 3.1 Keyboard Teleoperation

Publishes `geometry_msgs/Twist` velocity commands to `/cmd_vel`. Drive the robot around the environment so that all landmarks are observed from multiple angles and distances — richer observations lead to tighter EKF covariances.

```bash
ros2 run turtlebot3_teleop teleop_keyboard
```

### 3.2 ArUco Landmark Tracker

Subscribes to the camera feed and publishes detected marker poses to `/aruco_detections`. The `--ros-args` remapping bridges the ArUco node's expected internal topic names to the actual ones published by the V4L2 camera driver.

```bash
ros2 run aruco_opencv aruco_tracker_autostart --ros-args \
  -r /camera/image_raw:=/image_raw \
  -r /camera/camera_info:=/camera_info
```

### 3.3 Data Recording

Records all topics needed for offline EKF-SLAM replay and evaluation. Always use this full topic set — omitting `/tf_static` or `/camera_info` will break offline playback.

```bash
ros2 bag record /aruco_detections /image_raw /odom /tf /tf_static /camera_info
```

---

## 3A. Camera Calibration (ArduCam)

> **One-time setup** — only required once per robot unit. Skip if the calibration file already exists at `~/.ros/camera_info/mmal_service_16.1.yaml`.

Camera calibration maps pixel coordinates to real-world angles and distances. Without it, ArUco range and bearing estimates will be systematically biased, degrading SLAM accuracy. The calibration was computed using the ROS `camera_calibration` package with a checkerboard pattern (mono pinhole, OST v5.0, 120 samples).

### Step 1 — Create the directory

```bash
mkdir -p ~/.ros/camera_info
```

### Step 2 — Open the calibration file for editing

```bash
nano ~/.ros/camera_info/mmal_service_16.1.yaml
```

### Step 3 — Paste the calibration data

Copy the block below exactly into the editor:

```yaml
image_width: 320
image_height: 240
camera_name: arducam
camera_matrix:
  rows: 3
  cols: 3
  data: [245.574224, 0.000000,  164.427251,
         0.000000,  246.335295,  99.800133,
         0.000000,    0.000000,   1.000000]
distortion_model: plumb_bob
distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.036978, 0.008770, -0.041104, 0.012788, 0.000000]
rectification_matrix:
  rows: 3
  cols: 3
  data: [1.000000, 0.000000, 0.000000,
         0.000000, 1.000000, 0.000000,
         0.000000, 0.000000, 1.000000]
projection_matrix:
  rows: 3
  cols: 4
  data: [247.309906, 0.000000, 168.408758, 0.000000,
         0.000000, 239.618699,  89.403075, 0.000000,
         0.000000,   0.000000,   1.000000, 0.000000]
```

### Step 4 — Save and exit

`Ctrl+X` → `Y` → `Enter`

### Calibration Parameter Reference

| Symbol | Field | Value | Meaning |
|--------|-------|-------|---------|
| fx | camera_matrix [0,0] | 245.574 px | Horizontal focal length |
| fy | camera_matrix [1,1] | 246.335 px | Vertical focal length |
| cx | camera_matrix [0,2] | 164.427 px | Horizontal principal point (image centre x) |
| cy | camera_matrix [1,2] | 99.800 px | Vertical principal point (image centre y) |
| k1 | distortion [0] | −0.036978 | Radial distortion (barrel/pincushion) |
| k2 | distortion [1] | +0.008770 | Radial distortion (higher order) |
| p1 | distortion [2] | −0.041104 | Tangential distortion |
| p2 | distortion [3] | +0.012788 | Tangential distortion |

> The **rectification matrix** is identity — no stereo rectification is needed for a monocular camera.
> The **projection matrix** (P) encodes a small shift to the principal point after undistortion.

### Manual `camera_info` Publication (fallback only)

Use this only if the calibration file is not being picked up automatically by the camera node. This publishes the calibration values directly as a `CameraInfo` message to the topic, bypassing the file.

```bash
ros2 topic pub /camera_info sensor_msgs/msg/CameraInfo "{header: {frame_id: 'camera_link'}, height: 240, width: 320, distortion_model: 'plumb_bob', d: [-0.036978, 0.008770, -0.041104, 0.012788, 0.000000], k: [245.574224, 0.0, 164.427251, 0.0, 246.335295, 99.800133, 0.0, 0.0, 1.0], r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], p: [247.309906, 0.0, 168.408758, 0.0, 0.0, 239.618699, 89.403075, 0.0, 0.0, 0.0, 1.0, 0.0]}"
```

---

## 4. Visualization & Data Monitoring

> Run on the **local workstation**. RViz2 and the image viewer are complementary — launch both together to get a complete view of the robot state and the camera feed side by side.

### 4.1 RViz2 — Full State Visualization

Opens the RViz2 3D environment. Use it to inspect robot pose, landmark positions, covariance ellipses, and TF frames in real time.

```bash
rviz2
```

> Set the **Fixed Frame** to `odom` under *Global Options*. Recommended displays: `TF`, `Image` (topic: `/image_raw`), and any custom EKF marker publishers.

### 4.2 Image Viewer — Live Camera Feed

Launches a standalone window showing the raw camera stream. Use it alongside RViz2 to confirm that ArUco markers are visible, well-lit, and being detected correctly.

```bash
ros2 run rqt_image_view rqt_image_view
```

### 4.3 ArUco Detection Stream — Live Echo

Prints each incoming detection message to the terminal in real time. Use this to verify that marker IDs, estimated ranges, and bearings look physically plausible before committing to a recording.

```bash
ros2 topic echo /aruco_detections
```

### 4.4 ArUco Topic Metadata

Displays publisher/subscriber count, message type, and QoS settings for the detections topic. Run this to confirm the tracker node is actively publishing.

```bash
ros2 topic info /aruco_detections
```

---

## 5. Offline Analysis & Bag Playback

> Run on the **local workstation** after a recording session — no robot connection required.

### 5.1 Inspect Bag Metadata

Prints a full summary of the recording: duration, total message count, all topics, average publish rates, and file size. Always run this first to verify a recording is complete before starting analysis.

```bash
ros2 bag info <bag_folder_name>
```

### 5.2 Standard Playback

Replays the bag at real-time speed, republishing all recorded topics as originally captured. The EKF-SLAM node subscribes to these exactly as it would on a live robot.

```bash
ros2 bag play <bag_folder_name>
```

### 5.3 Looped Playback with Rate Control

Replays the bag in a continuous loop at a controlled speed multiplier. Use `--rate 0.5` for half speed when debugging fast sequences, or `--rate 2.0` to sweep through a long recording quickly.

```bash
ros2 bag play <bag_folder_name> --loop --rate 1.0
```
