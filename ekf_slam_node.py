"""
EKF SLAM ROS 2 Node — Real Robot Data Entry Point.
===================================================

This is the Week 4 replacement for run_simulation.py.  Instead of
stepping through a predefined path in a for-loop, this node subscribes
to live (or bag-replayed) ROS 2 topics and runs the EKF SLAM algorithm
in real time.

Topics consumed:
    /odom                — nav_msgs/Odometry
                           Provides the odometry control input u.
                           The EKF prediction step fires on every message.

    /aruco_detections    — aruco_opencv_msgs/ArucoDetection
                           Provides range-bearing landmark observations.
                           The EKF correction step fires on every message
                           that contains at least one detected marker.

Data flow:
    /odom       → odom_to_u()         → ekf_predict()   → mu, Sigma
    /aruco_det  → aruco_to_obs()      → ekf_update()    → mu, Sigma
                                      → update_plot()   (every N callbacks)

Differences from run_simulation.py:
    - No for-loop: motion is driven by /odom callbacks, not waypoints.
    - No simulate_odometry() or simulate_observations(): replaced by
      simulation/sensors_ros.py converters.
    - No evaluation / Procrustes: skipped until ground-truth positions
      are available (Week 5).
    - Initial pose read from cfg.INIT_ROBOT_POSE (no magic numbers here).
    - MAX_RANGE / MAX_BEARING filters applied to ArUco observations before
      the EKF update, matching the behaviour of simulate_observations().
    - Final plot is shown when the node is shut down (Ctrl+C or bag ends).

Usage
-----
With a rosbag (recommended for Week 4 validation):

    # Terminal 1 — start the EKF node
    cd my_ekf_slam/
    python3 ekf_slam_node.py

    # Terminal 2 — replay the bag
    ros2 bag play <bag_folder>

Live robot:
    # Start bringup + camera on the robot (see ROS2_HUMBLE_TURTLEBOT3_REFERENCE.md)
    # Then on the workstation:
    python3 ekf_slam_node.py

Shutdown:
    Ctrl+C — triggers the shutdown hook, which shows the final static plot.

ROS 2 environment:
    Requires ROS 2 Humble sourced:
        source /opt/ros/humble/setup.bash
    Requires aruco_opencv_msgs installed:
        sudo apt install ros-humble-aruco-opencv

Reference
---------
Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 10.
"""

import sys
import numpy as np
import rclpy
from rclpy.node            import Node
from nav_msgs.msg          import Odometry
from aruco_opencv_msgs.msg import ArucoDetection

# All tunable parameters (R, Q, INIT_ROBOT_POSE, MAX_RANGE, MAX_BEARING,
# ANIMATE, ANIMATION_STEP, …) come from config.py — no magic numbers here.
import config as cfg

from ekf_core                import ekf_predict, ekf_update
from simulation.sensors_ros  import odom_to_u, aruco_to_observations
from visualization           import update_plot, plot_final


# ===========================================================================
# EKF SLAM NODE
# ===========================================================================

class EkfSlamNode(Node):
    """
    ROS 2 node that runs EKF SLAM on live or bag-replayed sensor data.

    Attributes
    ----------
    _mu              : np.ndarray  — EKF state vector  [x, y, theta, lms...]
    _Sigma           : np.ndarray  — EKF covariance    (n x n)
    _landmark_map    : dict        — {marker_id -> start_index_in_mu}
    _prev_odom       : msg | None  — last Odometry message (needed to form u)
    _true_path       : list        — ground-truth path (empty for Week 4 data)
    _robot_path      : list        — EKF-estimated [x, y] positions over time
    _landmarks_display: np.ndarray — current EKF landmark estimates (N, 2)
    _plot_step       : int         — aruco callback counter for throttling
    """

    def __init__(self) -> None:
        super().__init__('ekf_slam_node')

        # ------------------------------------------------------------------
        # EKF state initialisation
        # ------------------------------------------------------------------
        # Initial pose is read from config.py (INIT_ROBOT_POSE) rather than
        # hardcoded here.  Change it in config.py to match the real starting
        # position of the robot in the lab coordinate frame.
        #
        # Initial robot-pose uncertainty is near-zero: we treat the starting
        # pose as our reference frame.  Heading variance is 0 — we trust the
        # robot's initial orientation exactly.
        x0, y0, theta0 = cfg.INIT_ROBOT_POSE

        self._mu           = np.array([x0, y0, theta0], dtype=float)
        self._Sigma        = np.diag([0.01, 0.01, 0.0])
        self._landmark_map = {}

        # ------------------------------------------------------------------
        # Odometry state — we need two consecutive messages to form u
        # ------------------------------------------------------------------
        self._prev_odom = None   # stores the previous Odometry message

        # ------------------------------------------------------------------
        # Path recording buffers (for visualisation)
        # ------------------------------------------------------------------
        self._robot_path = []    # list of [x, y] EKF estimates
        self._true_path  = []    # empty — no ground truth available in Week 4

        # Placeholder landmark array for visualisation.
        # plot_final() and update_plot() always receive a landmarks_true array.
        # Since we have no ground truth, we pass the current EKF estimates as
        # both arguments — the visualiser draws them on top of each other,
        # effectively showing only the EKF estimates.
        # Grows dynamically as new landmarks are added to the map.
        self._landmarks_display = np.empty((0, 2), dtype=float)

        # ArUco callback counter — used to throttle live animation redraws.
        self._plot_step = 0

        # ------------------------------------------------------------------
        # ROS 2 subscribers
        # ------------------------------------------------------------------
        self._odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            qos_profile=10,
        )

        self._aruco_sub = self.create_subscription(
            ArucoDetection,
            '/aruco_detections',
            self._aruco_callback,
            qos_profile=10,
        )

        self.get_logger().info(
            f'EKF SLAM node started — initial pose '
            f'({x0:.2f}, {y0:.2f}, {np.rad2deg(theta0):.1f}°) — '
            f'waiting for sensor data.'
        )

    # =======================================================================
    # ODOMETRY CALLBACK — EKF PREDICTION STEP
    # =======================================================================

    def _odom_callback(self, msg: Odometry) -> None:
        """
        Fires on every /odom message.

        Runs the EKF prediction step using the odometry motion model.
        Skips the very first message (no previous pose available yet).

        Args:
            msg : nav_msgs/Odometry — current odometry reading.
        """
        # First message — store as reference and wait for the next one.
        if self._prev_odom is None:
            self._prev_odom = msg
            return

        # --- Convert two consecutive odom messages → control tuple u ---
        # u = (prev_odom, curr_odom), each a [x, y, theta] list.
        u = odom_to_u(self._prev_odom, msg)

        # --- EKF prediction step ---
        # Only the robot-pose block of mu/Sigma is affected; landmarks stay.
        self._mu, self._Sigma = ekf_predict(
            mu    = self._mu,
            Sigma = self._Sigma,
            u     = u,
            R     = cfg.R,
        )

        # --- Record estimated robot position for the path plot ---
        self._robot_path.append(self._mu[:2].copy())

        # --- Advance the odometry reference to the current message ---
        self._prev_odom = msg

    # =======================================================================
    # ARUCO CALLBACK — EKF CORRECTION STEP
    # =======================================================================

    def _aruco_callback(self, msg: ArucoDetection) -> None:
        """
        Fires on every /aruco_detections message.

        Converts detected markers to range-bearing observations, applies
        sensor FOV and range filters (matching simulate_observations()),
        and runs the EKF correction step.
        Empty frames and frames with no valid observations after filtering
        are skipped.  Triggers the live animation every ANIMATION_STEP calls.

        Args:
            msg : aruco_opencv_msgs/ArucoDetection — detection message.
        """
        # --- Convert ArUco message → observations list ---
        # Returns list of (landmark_id, np.array([r, phi])).
        observations = aruco_to_observations(msg)

        # --- Skip EKF update if no markers were detected this frame ---
        if not observations:
            return

        # --- [FIX #2] Apply sensor constraints before the EKF update ---
        # Filter out observations outside the camera's physical FOV and max
        # detection range, matching the behaviour of simulate_observations()
        # in sensors.py.  This prevents far/edge detections from corrupting
        # the map with poorly conditioned measurements.
        observations = [
            (lm_id, z)
            for lm_id, z in observations
            if z[0] <= cfg.MAX_RANGE and abs(z[1]) <= cfg.MAX_BEARING
        ]

        # After filtering, there may be nothing left — check again.
        if not observations:
            return

        # --- EKF correction step ---
        # New landmarks are initialised inside ekf_update() on first sight.
        self._mu, self._Sigma, self._landmark_map = ekf_update(
            mu_bar       = self._mu,
            Sigma_bar    = self._Sigma,
            observations = observations,
            landmark_map = self._landmark_map,
            Q            = cfg.Q,
        )

        # --- Rebuild landmark display array from current EKF estimates ---
        self._landmarks_display = self._build_landmark_display()

        # --- Live animation (throttled by cfg.ANIMATION_STEP) ---
        self._plot_step += 1
        if cfg.ANIMATE and (self._plot_step % cfg.ANIMATION_STEP == 0):
            update_plot(
                true_path      = self._true_path,
                robot_path     = self._robot_path,
                landmarks_true = self._landmarks_display,
                mu             = self._mu,
                Sigma          = self._Sigma,
                landmark_map   = self._landmark_map,
                cfg            = cfg,
                step           = self._plot_step,
            )

        # --- Console logging (every 50 aruco callbacks) ---
        if self._plot_step % 50 == 0:
            n_lm = len(self._landmark_map)
            self.get_logger().info(
                f'step {self._plot_step:>4d}  |  '
                f'robot ({self._mu[0]:6.2f}, {self._mu[1]:6.2f})  |  '
                f'landmarks in map: {n_lm}'
            )

    # =======================================================================
    # HELPERS
    # =======================================================================

    def _build_landmark_display(self) -> np.ndarray:
        """
        Builds a (N, 2) array of current EKF landmark estimates for
        the visualiser.

        Since update_plot() and plot_final() expect a landmarks_true array
        of the same shape as the EKF estimates, we pass the current EKF
        estimates as both the 'true' and 'estimated' arguments — the
        visualiser draws them on top of each other, effectively showing
        only the EKF estimates until ground truth is available (Week 5).

        Returns:
            lm_array : np.ndarray shape (N, 2) — [x, y] for each landmark
                       in landmark_map, sorted by landmark ID.
        """
        if not self._landmark_map:
            return np.empty((0, 2), dtype=float)

        rows = []
        for lm_id in sorted(self._landmark_map.keys()):
            j = self._landmark_map[lm_id]
            if j + 1 < len(self._mu):
                rows.append([self._mu[j], self._mu[j + 1]])

        return np.array(rows, dtype=float) if rows else np.empty((0, 2), dtype=float)

    def show_final_plot(self) -> None:
        """
        Renders the final static summary plot.

        Called by the shutdown hook when the node is stopped (Ctrl+C or
        bag playback ends).  Since no ground truth is available in Week 4,
        the EKF path is passed as both true_path and robot_path, and
        aligned_landmarks is None (no Procrustes evaluation).
        """
        if not self._robot_path:
            self.get_logger().warn('No data recorded — skipping final plot.')
            return

        self.get_logger().info('Rendering final plot ...')

        plot_final(
            true_path         = self._robot_path,   # no ground truth: use EKF path
            robot_path        = self._robot_path,
            landmarks_true    = self._landmarks_display,
            mu                = self._mu,
            Sigma             = self._Sigma,
            landmark_map      = self._landmark_map,
            cfg               = cfg,
            aligned_landmarks = None,               # Week 5: replace with eval result
        )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def main() -> None:
    """
    Initialises ROS 2, spins the EKF SLAM node, and shows the final
    plot on shutdown.

    The try/finally block guarantees that show_final_plot() and
    rclpy.shutdown() are always called, even if the node crashes mid-run.
    """
    rclpy.init(args=sys.argv)
    node = EkfSlamNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n  [ekf_slam_node] Interrupted — showing final plot.')
    finally:
        # Always attempt to show the final plot, even on unexpected crash.
        node.show_final_plot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()