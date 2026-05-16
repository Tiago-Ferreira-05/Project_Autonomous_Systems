"""
Configuration — EKF SLAM Simulator
====================================
All tunable parameters live here.

Changing a value in this file propagates automatically to every module
that imports Config, so there is no need to hunt for magic numbers scattered
across the codebase.

Sections:
    1. Motion noise          (R matrix)
    2. Measurement noise     (Q matrix)
    3. Sensor constraints
    4. EKF numerical settings
    5. Simulation settings
    6. Visualisation settings
    7. Output / logging

Reference for noise model:
    Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 5 (R) and
    Chapter 6 (Q).  Both matrices are diagonal; off-diagonal terms are
    zero under the standard independence assumption.
"""

import numpy as np


# =============================================================================
# SECTION 1 — MOTION NOISE  (R matrix, 3x3)
# =============================================================================
# R models the uncertainty injected by one step of the odometry motion model.
# Diagonal entries are variances (standard deviation squared).
#
# Physical meaning:
#   R_x     : how uncertain is the x displacement per step (metres²)
#   R_y     : how uncertain is the y displacement per step (metres²)
#   R_THETA : how uncertain is the heading change per step (radians²)
#
# Tuning guide:
#   Increase R → filter trusts odometry less → landmark corrections have
#                 more influence → map stays more consistent but trajectory
#                 may lag real motion.
#   Decrease R → filter trusts odometry more → trajectory tracks motion
#                 closely but accumulated drift is not corrected quickly.
#
# Starting values are conservative estimates for a Pioneer/TurtleBot3 on a
# flat lab floor.  Refine empirically from rosbag data (Week 3 workplan).

MOTION_STD_X     = 0.05          # metres — std dev of x displacement noise
MOTION_STD_Y     = 0.05          # metres — std dev of y displacement noise
MOTION_STD_THETA = np.deg2rad(2) # radians — std dev of heading noise  (~2°)

# Build R as a 3x3 diagonal covariance matrix
R = np.diag([
    MOTION_STD_X     ** 2,
    MOTION_STD_Y     ** 2,
    MOTION_STD_THETA ** 2,
])


# =============================================================================
# SECTION 2 — MEASUREMENT NOISE  (Q matrix, 2x2)
# =============================================================================
# Q models the uncertainty in each range-bearing observation.
#
# Physical meaning:
#   Q_RANGE   : how uncertain is the estimated range from ArUco size (metres²)
#   Q_BEARING : how uncertain is the measured bearing angle (radians²)
#
# Tuning guide:
#   Increase Q → filter trusts measurements less → landmark estimates move
#                 slowly, useful when ArUco detections are noisy or flickery.
#   Decrease Q → filter trusts measurements more → fast convergence but
#                 susceptible to outliers and false detections.
#
# Range uncertainty is typically larger than bearing uncertainty because
# estimating distance from marker pixel size is less precise than measuring
# the horizontal pixel offset (bearing).

MEAS_STD_RANGE   = 0.10          # metres — std dev of range measurement noise
MEAS_STD_BEARING = np.deg2rad(3) # radians — std dev of bearing noise  (~3°)

# Build Q as a 2x2 diagonal covariance matrix
Q = np.diag([
    MEAS_STD_RANGE   ** 2,
    MEAS_STD_BEARING ** 2,
])


# =============================================================================
# SECTION 3 — SENSOR CONSTRAINTS
# =============================================================================
# Defines the physical field-of-view and maximum detection range of the
# camera used for ArUco marker detection.
#
# Observations outside these limits are discarded by simulate_observations()
# (simulator) or filtered at the ROS node level (real data).

MAX_RANGE         = 2.0              # metres — beyond this, markers too small
MAX_BEARING       = np.deg2rad(22.5)  # radians — ±22.5° half-FOV (total 45°)

# Camera intrinsics (used for real ArUco range estimation from marker size).
# Not used by the simulator, but kept here for when the ROS node is built.
ARUCO_MARKER_SIZE = 0.20             # metres — physical side length of markers
CAMERA_FX         = 530.0            # pixels — focal length x  (placeholder)
CAMERA_FY         = 530.0            # pixels — focal length y  (placeholder)


# =============================================================================
# SECTION 4 — EKF NUMERICAL SETTINGS
# =============================================================================

# Mahalanobis distance threshold for outlier rejection in ekf_update().
# An observation is rejected if its Mahalanobis distance to the predicted
# measurement exceeds this value.
#
# chi-squared distribution, 2 DOF (range + bearing):
#   chi2(2, 0.95) = 5.99   →  reject 5% of measurements as outliers
#   chi2(2, 0.99) = 9.21   →  reject 1% of measurements as outliers
#
# Use 9.21 (conservative) for real data; can be relaxed in simulation where
# noise is perfectly Gaussian.
MAHALANOBIS_THRESHOLD = 9.21

# Minimum squared range to a landmark before the Jacobian is considered
# numerically degenerate.  See compute_H() in meas_model.py.
MIN_RANGE_SQ = 1e-9               # metres²


# =============================================================================
# SECTION 5 — SIMULATION SETTINGS
# =============================================================================

# Initial robot pose  [x, y, theta]  in metres and radians.
# Used as the starting pose in both path mode and keyboard mode.
INIT_ROBOT_POSE = np.array([5.0, 20.0, 0.0], dtype=float)

# Map grid dimensions (cells).  Each cell = 1 metre in the simulator.
MAP_WIDTH  = 25                   # cells
MAP_HEIGHT = 25                   # cells

# Number of complete laps the robot makes around the map (path mode only).
# More laps → more re-observations → tighter landmark covariances.
NUM_LAPS = 2

# ---------------------------------------------------------------------------
# Control mode
# ---------------------------------------------------------------------------
# "path"     — robot follows the hardcoded rectangular waypoint path
#              (default, fully automated, reproducible).
# "keyboard" — robot is driven interactively via W/A/S/D keys in the live
#              matplotlib window.  Press Q to finish and run evaluation.
CONTROL_MODE = "path"    # "path" | "keyboard"

# ---------------------------------------------------------------------------
# Keyboard control parameters  (used only when CONTROL_MODE = "keyboard")
# ---------------------------------------------------------------------------
# KEYBOARD_STEP_M   : forward / backward translation per keypress (metres).
# KEYBOARD_ROT_DEG  : rotation per A / D keypress (degrees).
#
# Smaller steps give finer control but more keypresses per metre travelled.
# Larger steps feel faster but increase per-step odometry noise accumulation.
KEYBOARD_STEP_M  = 0.5             # metres per W / S keypress
KEYBOARD_ROT_DEG = 15.0            # degrees per A / D keypress


# =============================================================================
# SECTION 6 — VISUALISATION SETTINGS
# =============================================================================

# Whether to show the live animation during the simulation run.
# Set to False for batch experiments (much faster, no rendering overhead).
# Note: keyboard mode forces ANIMATE = True regardless of this setting,
# because the figure window is required to capture key events.
ANIMATE = False

# How many timesteps to skip between animation redraws.
# 1 = redraw every step (slow), 5 = redraw every 5 steps (smoother).
# In keyboard mode this is ignored — the plot is redrawn after every keypress.
ANIMATION_STEP = 3

# Pause time in seconds between animation frames (passed to plt.pause()).
PAUSE_TIME = 0.05                 # seconds

# Number of standard deviations for covariance ellipses in the plot.
# 1 = ~68% confidence region, 2 = ~95%, 3 = ~99.7%.
ELLIPSE_N_STD = 1

# Colour scheme for the plots.
COLOR_TRUE_PATH       = 'green'
COLOR_EKF_PATH        = 'blue'
COLOR_TRUE_LANDMARK   = 'black'
COLOR_EKF_LANDMARK    = 'red'
COLOR_EKF_ELLIPSE     = 'orange'
COLOR_ROBOT_POSE      = 'blue'
COLOR_ROBOT_ELLIPSE   = 'blue'


# =============================================================================
# SECTION 7 — OUTPUT / LOGGING
# =============================================================================

# Whether to write per-step results to a CSV file.
# The CSV is used for statistical analysis across multiple runs (Week 5).
SAVE_CSV     = True
CSV_FILENAME = 'results/ekf_slam_results.csv'

# Whether to save the final plot as a PNG file.
SAVE_PLOT       = False
PLOT_FILENAME   = 'results/ekf_slam_final_plot.png'
PLOT_DPI        = 150


# =============================================================================
# CONVENIENCE: build noise matrices at import time
# =============================================================================
# These are re-exported so any module can do:
#     from ekf_core import config
#     R = config.R
#     Q = config.Q
# instead of rebuilding the matrices themselves.

def get_R() -> np.ndarray:
    """Returns a fresh copy of the motion noise covariance matrix R (3x3)."""
    return R.copy()

def get_Q() -> np.ndarray:
    """Returns a fresh copy of the measurement noise covariance matrix Q (2x2)."""
    return Q.copy()