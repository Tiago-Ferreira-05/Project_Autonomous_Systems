"""
ROS 2 Sensor Interface — Real Odometry and ArUco Observations.
==============================================================

This module is the real-data replacement for sensors.py.
It provides two pure conversion functions that transform ROS 2
message types into the exact same data formats expected by ekf_core/:

    odom_to_u(prev_msg, curr_msg)
        Converts two consecutive nav_msgs/Odometry messages into the
        odometry control tuple u = (prev_odom, curr_odom), where each
        element is a [x, y, theta] array.  This is identical in format
        to the output of simulate_odometry() in sensors.py.

    aruco_to_observations(aruco_msg)
        Converts an aruco_opencv_msgs/ArucoDetection message into a
        list of (landmark_id, np.array([r, phi])) tuples — identical
        in format to the output of simulate_observations() in sensors.py.

Design notes
------------
- Both functions are stateless pure converters: they take a ROS message
  and return a Python/numpy structure.  No ROS node, no subscribers, no
  spin loop — those live in ekf_slam_node.py.
- The camera coordinate frame used by aruco_opencv is:
      +x : right
      +y : down
      +z : forward (depth / range)
  Range and bearing are computed from the (x, z) plane only, ignoring
  the vertical (y) component, which is correct for a 2D SLAM problem.
- Bearing sign convention matches sensors.py and meas_model.py:
      phi > 0 : landmark to the LEFT  of the robot heading
      phi < 0 : landmark to the RIGHT of the robot heading
  This is achieved with phi = atan2(-x, z).

Reference
---------
Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 5 (odometry),
Chapter 6 (range-bearing sensor model).
ROS 2 nav_msgs/Odometry message definition.
aruco_opencv_msgs/ArucoDetection message definition.
"""

import numpy as np
from math import atan2, sqrt


# ===========================================================================
# PRIVATE UTILITIES
# ===========================================================================

def _quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    """
    Extracts the yaw angle (rotation around the z-axis) from a quaternion.

    For a ground-based robot moving in the 2D plane, yaw is the only
    rotation component that matters.  The standard formula is derived
    from the rotation matrix element atan2(2*(qw*qz + qx*qy),
    1 - 2*(qy^2 + qz^2)).

    Args:
        qx, qy, qz, qw : Quaternion components from the ROS message.

    Returns:
        yaw : float — robot heading in radians, wrapped to [-pi, pi].
    """
    # Standard quaternion-to-yaw formula (ZYX Euler convention)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return atan2(siny_cosp, cosy_cosp)


def _normalize_angle(angle: float) -> float:
    """Wraps angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ===========================================================================
# PUBLIC API
# ===========================================================================

def odom_to_u(prev_msg, curr_msg) -> tuple:
    """
    Converts two consecutive nav_msgs/Odometry messages into the
    odometry control tuple expected by ekf_predict().

    The EKF motion model needs two consecutive odometry poses so it can
    compute the displacement (delta_rot1, delta_trans, delta_rot2).
    This function extracts [x, y, theta] from each message and packages
    them into the same (prev_odom, curr_odom) tuple format that
    simulate_odometry() produces in sensors.py.

    Args:
        prev_msg : nav_msgs/Odometry — odometry message at time t-1.
        curr_msg : nav_msgs/Odometry — odometry message at time t.

    Returns:
        u : tuple (prev_odom, curr_odom)
            prev_odom : list [x, y, theta] — pose at time t-1
            curr_odom : list [x, y, theta] — pose at time t

    Coordinate convention:
        x, y   : metres, from msg.pose.pose.position
        theta  : radians, yaw extracted from msg.pose.pose.orientation
                 (quaternion → yaw via _quaternion_to_yaw)
    """
    def _extract_pose(msg) -> list:
        """Extracts [x, y, theta] from a single Odometry message."""
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        qx    = msg.pose.pose.orientation.x
        qy    = msg.pose.pose.orientation.y
        qz    = msg.pose.pose.orientation.z
        qw    = msg.pose.pose.orientation.w
        theta = _quaternion_to_yaw(qx, qy, qz, qw)
        return [x, y, theta]

    prev_odom = _extract_pose(prev_msg)
    curr_odom = _extract_pose(curr_msg)

    return (prev_odom, curr_odom)


def aruco_to_observations(aruco_msg) -> list:
    """
    Converts an aruco_opencv_msgs/ArucoDetection message into a list of
    range-bearing observations expected by ekf_update().

    Each detected marker is converted from its 3D camera-frame pose into
    a (range, bearing) pair.  Markers with empty detections (markers: [])
    produce an empty list — the EKF update step is simply skipped.

    Camera coordinate frame (aruco_opencv convention):
        +x : right  (lateral offset)
        +y : down   (vertical offset — ignored for 2D SLAM)
        +z : forward (depth / range to marker)

    Range and bearing computation:
        r   = sqrt(x^2 + z^2)   — Euclidean distance in the ground plane
        phi = atan2(-x, z)      — bearing relative to camera boresight
                                   negative x because rightward offset
                                   means negative bearing (right of heading)

    Args:
        aruco_msg : aruco_opencv_msgs/ArucoDetection
                    Expected fields per marker:
                        marker.marker_id          — int landmark ID
                        marker.pose.position.x    — lateral offset  (m)
                        marker.pose.position.z    — depth / range   (m)

    Returns:
        observations : list of (landmark_id, np.array([r, phi]))
                       Empty list if no markers were detected.
                       Matches the format of simulate_observations() exactly.

    Notes:
        - Markers with range < 1e-6 m are skipped (degenerate case).
        - Negative range values (sensor glitch) are skipped.
        - phi is normalised to [-pi, pi] after computation.
    """
    observations = []

    for marker in aruco_msg.markers:

        lm_id = int(marker.marker_id)

        # --- Extract camera-frame position ---
        cam_x = marker.pose.position.x   # lateral  (+right, -left)
        cam_z = marker.pose.position.z   # depth    (forward = positive)

        # --- Skip degenerate or invalid detections ---
        if cam_z <= 0.0:
            # Marker behind the camera — invalid detection
            continue

        # --- Range: Euclidean distance in the ground plane ---
        r = sqrt(cam_x ** 2 + cam_z ** 2)

        if r < 1e-6:
            # Robot effectively on top of marker — skip
            continue

        # --- Bearing: angle relative to camera boresight (+z axis) ---
        # atan2(-x, z) : rightward offset → negative bearing (right of heading)
        #                leftward offset  → positive bearing (left of heading)
        phi = _normalize_angle(atan2(-cam_x, cam_z))

        observations.append((lm_id, np.array([r, phi])))

    return observations