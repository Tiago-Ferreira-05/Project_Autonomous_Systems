# Useful Commands

## 1. Environment Setup & Installation

Run on a local workstation (For example, a laptop).

### Update package lists (always execute before any installation)

sudo apt update

### Install TurtleBot3 and control packages

sudo apt install ros-humble-dynamixel-sdk  
sudo apt install ros-humble-turtlebot3-msgs  
sudo apt install ros-humble-turtlebot3  
sudo apt install ros-humble-teleop-twist-keyboard

### Install ArUco and vision packages

sudo apt install ros-humble-aruco-opencv  
sudo apt install ros-humble-image-transport  
sudo apt install ros-humble-image-transport-plugins

### ~/.bashrc configuration

#### Set TurtleBot model

export TURTLEBOT3_MODEL=waffle_pi

#### Set ROS Domain ID (Matches your specific robot ID)

export ROS_DOMAIN_ID=14

#### Source ROS 2 Humble setup

source /opt/ros/humble/setup.bash

---

## 2. Robot Operations (SSH)

Execute these on the Raspberry Pi.

### Remote Login (Replace IP with target robot)

ssh deec@10.16.140.14

### Hardware Bringup

ros2 launch turtlebot3_bringup robot.launch.py

### Sensor Initialization (Low-bandwidth resolution for SLAM stability)

ros2 run v4l2_camera v4l2_camera_node --ros-args -p image_size:="[320,240]"

---

## 3. SLAM Execution & Data Acquisition

Workstation-side operations.

### Movement control

ros2 run turtlebot3_teleop teleop_keyboard

### Landmark Tracker Initialization (Internal Remapping)

ros2 run aruco_opencv aruco_tracker_autostart --ros-args \
-r /camera/image_raw:=/image_raw \
-r /camera/camera_info:=/camera_info

### Manual Intrinsic Matrix (K) Publication

Matrix Format: [fx, 0, cx, 0, fy, cy, 0, 0, 1]

ros2 topic pub /camera_info sensor_msgs/msg/CameraInfo "{
header: {frame_id: 'camera_link'},
height: 240, width: 320, distortion_model: 'plumb_bob',
d: [0.0, 0.0, 0.0, 0.0, 0.0],
k: [300.0, 0.0, 160.0, 0.0, 300.0, 120.0, 0.0, 0.0, 1.0],
r: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
p: [300.0, 0.0, 160.0, 0.0, 0.0, 300.0, 120.0, 0.0, 0.0, 0.0, 1.0, 0.0]
}"

### Data Recording

ros2 bag record /aruco_detections /image_raw /odom /tf /tf_static /camera_info

---

## 4. Data Analysis & Visualization

Offline processing and verification.

### View bag metadata

ros2 bag info <bag_folder_name>

### Standard playback

ros2 bag play <bag_folder_name>

### Advanced playback (Looping and Rate control)

ros2 bag play <bag_folder_name> --loop --rate 1.0

### Visualization

rviz2  
ros2 run rqt_image_view rqt_image_view

Fixed Frame: odom

### Data Monitoring

ros2 topic echo /aruco_detections  
ros2 topic info /aruco_detections
