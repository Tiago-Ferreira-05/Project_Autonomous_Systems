# EKF SLAM — ROS 2 Node

Range-bearing EKF SLAM using ArUco markers and wheel odometry on a TurtleBot3 Waffle Pi (ROS 2 Humble).

---

## Prerequisites

- ROS 2 Humble installed. If you haven't added the source to your shell profile, add it once:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

After this, all new terminals will have ROS 2 available automatically. If you already have this line in your `~/.bashrc`, skip this step entirely.

- Packages installed: `aruco_opencv`, `slam_toolbox`, `image_transport`, `rqt_image_view`
- Python dependencies installed:

```bash
pip install -r requirements_ros.txt --break-system-packages
```

---

## How to Run

---

### Terminal 1 — Decompress camera images

Republishes the compressed camera stream as a raw topic that downstream nodes can consume.

```bash
ros2 run image_transport republish compressed \
  --ros-args \
  --remap in/compressed:=/image_raw/compressed \
  --remap out:=/image_raw/decompressed \
  --param reliability:=best_effort
```

---

### Terminal 2 — Camera info publisher

Publishes the camera calibration parameters required by the ArUco tracker.

```bash
python3 camera_info_publisher.py
```

---

### Terminal 3 — ArUco tracker

Detects ArUco markers in the decompressed image stream and publishes their poses on `/aruco_detections`.

```bash
ros2 run aruco_opencv aruco_tracker_autostart \
  --ros-args \
  -p cam_base_topic:=/image_raw \
  --remap /image_raw:=/image_raw/decompressed \
  --remap /camera_info:=/camera_info_fixed \
  -p image_is_rectified:=false \
  -p output_frame:=camera
```

---

### Terminal 4 — Image viewer *(optional)*

Visual sanity check — lets you inspect the raw or annotated camera feed.

```bash
ros2 run rqt_image_view rqt_image_view
```

---

### Terminal 5 — Rosbag playback

Replays the recorded dataset. Adjust `--rate` to slow down or speed up playback.

```bash
ros2 bag play 24arucos_closed_delay/ --rate 30
```

> Start this **after** Terminals 1–4 are running so no messages are missed.

---

### Terminal 6 — EKF SLAM node

Runs the EKF SLAM algorithm. Subscribes to `/odom` and `/aruco_detections`, produces the estimated trajectory and landmark map.

```bash
python3 ekf_slam_node.py
```

Results are saved to `results/` on shutdown (plot + `.npz` file).

---

### Terminal 7 — slam_toolbox *(ground truth reference)*

Runs lidar-based SLAM in parallel. Its `map → base_footprint` TF is read by the EKF node as a ground-truth reference path for comparison.

```bash
ros2 launch slam_toolbox online_async_launch.py
```

---

## Recommended Start Order

```
T1 image_transport  →  T2 camera_info  →  T3 aruco_tracker
→  T4 rqt (optional)  →  T7 slam_toolbox  →  T6 ekf_slam_node
→  T5 rosbag play
```

Start the bag **last**, once all nodes are ready to receive data.

---

## Output

| File | Description |
|---|---|
| `results/ekf_slam_ros_final.png` | Final trajectory + landmark map plot |
| `results/ekf_slam_run.npz` | Raw arrays (paths, `mu`, `Sigma`, `landmark_map`) for offline analysis |

---

## Tuning

All parameters are in `config.py`:

| Parameter | Effect |
|---|---|
| `ALPHA1`–`ALPHA4` | Motion noise — increase if odometry drifts excessively |
| `MEAS_STD_RANGE` / `MEAS_STD_BEARING` | Measurement noise — increase to trust ArUco less |
| `MAHALANOBIS_THRESHOLD` | Outlier rejection gate — lower value = stricter filtering |
| `MAX_RANGE` / `MAX_BEARING` | Sensor FOV limits — detections outside are discarded |
