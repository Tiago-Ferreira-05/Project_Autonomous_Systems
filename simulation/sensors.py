"""
Sensor Simulation — Noisy Odometry and Range-Bearing Observations.

Reference: Probabilistic Robotics (Thrun, Burgard, Fox)
               Chapter 5  — Odometry motion model (noise model)
               Chapter 6  — Range-bearing sensor model
           micro_simulador_V-2_1.py (friend's reference implementation)

This module replaces the two real-data sources that exist in the ROS 2 node:

    Real robot (Session 4)          →   Simulator (Session 3 / this file)
    ─────────────────────────────────────────────────────────────────────
    /odom  topic callback           →   simulate_odometry()
    /image topic + ArUco detection  →   simulate_observations()

The EKF core (ekf_core/) receives the same data format in both cases:
    u  = (prev_odom, curr_odom)       — odometry control tuple
    z  = [(lm_id, [r, phi]), ...]     — list of range-bearing observations

This clean interface is why the EKF algorithm works identically in simulation
and on real data without any modification.

Noise model:
    Both functions add zero-mean Gaussian noise whose variance is taken from
    the R and Q matrices defined in config.py.  This mimics the imperfect
    wheel encoders (odometry noise) and imprecise ArUco range/bearing
    estimation (measurement noise) of the real TurtleBot3 / Pioneer robot.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal angle utility
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """Wraps angle to [-pi, pi]."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# =============================================================================
# NOISY ODOMETRY SIMULATION
# =============================================================================

def simulate_odometry(
    prev_true_pose : list,
    curr_true_pose : list,
    R              : np.ndarray,
    rng            : np.random.Generator = None,
) -> tuple:
    """
    Simulates noisy wheel encoder readings for one motion step.

    Takes two consecutive true poses (from the predefined path) and adds
    independent Gaussian noise to each displacement component, producing a
    realistic odometry measurement that drifts from the ground truth over time.

    This mimics how a real /odom ROS topic works: the encoder measures how
    far each wheel turned, converts it to a pose delta, and accumulates it —
    but with noise that compounds at every step.

    Args:
        prev_true_pose : [x, y, theta] — true robot pose at time t-1
        curr_true_pose : [x, y, theta] — true robot pose at time t
        R              : np.ndarray shape (3, 3) — motion noise covariance.
                         Diagonal entries are variances for [x, y, theta].
                         Off-diagonal entries are ignored (independence assumed).
        rng            : Optional numpy random Generator for reproducibility.
                         If None, uses the global numpy RNG.
                         Pass np.random.default_rng(seed) for reproducible runs.

    Returns:
        u : tuple (prev_odom, curr_odom)
            prev_odom — the previous odometry pose (passed through unchanged,
                        used as the reference frame for the motion primitives)
            curr_odom — noisy version of curr_true_pose

    Noise model:
        dx_noisy     = dx_true     + N(0, R[0,0])
        dy_noisy     = dy_true     + N(0, R[1,1])
        dtheta_noisy = dtheta_true + N(0, R[2,2])

    Note:
        The noise is added to the displacement (delta), not to the absolute
        pose.  This is the correct model because encoders measure incremental
        wheel rotations, not absolute positions.
    """
    rng = rng or np.random.default_rng()

    # --- True displacement between the two poses ---
    dx     = curr_true_pose[0] - prev_true_pose[0]
    dy     = curr_true_pose[1] - prev_true_pose[1]
    dtheta = _normalize_angle(curr_true_pose[2] - prev_true_pose[2])

    # --- Add independent Gaussian noise to each displacement component ---
    # std dev = sqrt(variance) = sqrt(diagonal of R)
    dx_noisy     = dx     + rng.normal(0.0, np.sqrt(R[0, 0]))
    dy_noisy     = dy     + rng.normal(0.0, np.sqrt(R[1, 1]))
    dtheta_noisy = dtheta + rng.normal(0.0, np.sqrt(R[2, 2]))

    # --- Build the noisy current odometry reading ---
    # prev_odom is the true previous pose (serves as the fixed reference frame)
    # curr_odom is computed by applying the noisy displacement to the previous pose
    curr_odom = [
        prev_true_pose[0] + dx_noisy,
        prev_true_pose[1] + dy_noisy,
        _normalize_angle(prev_true_pose[2] + dtheta_noisy),
    ]

    prev_odom = list(prev_true_pose)   # pass-through, no noise on the reference

    return (prev_odom, curr_odom)


# =============================================================================
# RANGE-BEARING OBSERVATION SIMULATION
# =============================================================================

def simulate_observations(
    true_pose      : list,
    landmarks_true : np.ndarray,
    Q              : np.ndarray,
    max_range      : float,
    max_bearing    : float,
    rng            : np.random.Generator = None,
) -> list:
    """
    Simulates noisy range-bearing observations for all visible landmarks.

    For each true landmark, checks whether it falls within the robot's sensor
    range and field of view.  If visible, computes the true range and bearing
    and adds Gaussian noise — mimicking how a real camera + ArUco detection
    pipeline produces imperfect measurements.

    In the real robot (Session 4):
        - Range   is estimated from the apparent pixel size of the ArUco marker
          (known physical size → distance from camera geometry).
        - Bearing is estimated from the horizontal offset of the marker centre
          in the image relative to the image centre (camera intrinsics).

    Both sources introduce measurement errors that this function models.

    Args:
        true_pose      : [x, y, theta] — current true robot pose
        landmarks_true : np.ndarray shape (N, 2) — true landmark positions
        Q              : np.ndarray shape (2, 2) — measurement noise covariance.
                         Q[0,0] = range variance (metres²)
                         Q[1,1] = bearing variance (radians²)
        max_range      : float — maximum detection range (metres).
                         Landmarks beyond this distance are not visible.
        max_bearing    : float — half field-of-view angle (radians).
                         Only landmarks with |bearing| <= max_bearing are visible.
        rng            : Optional numpy random Generator for reproducibility.

    Returns:
        observations : list of (landmark_id, z)
                       landmark_id — integer index into landmarks_true (0-based)
                       z           — np.ndarray [r_noisy, phi_noisy]

                       Only visible landmarks appear in the list.
                       Order within the list is the order of landmark indices.

    Noise model:
        r_noisy   = r_true   + N(0, Q[0,0])
        phi_noisy = phi_true + N(0, Q[1,1])

    Note on bearing convention:
        phi = atan2(dy, dx) - theta   →   relative to robot heading
        Wrapped to [-pi, pi] before noise is added, and again after,
        to ensure the output is always in the valid range.
    """
    rng          = rng or np.random.default_rng()
    observations = []

    x, y, theta = true_pose[0], true_pose[1], true_pose[2]

    for lm_id, lm in enumerate(landmarks_true):
        lm_x, lm_y = lm[0], lm[1]

        # --- Compute true range and bearing to this landmark ---
        dx      = lm_x - x
        dy      = lm_y - y
        r_true  = np.hypot(dx, dy)
        phi_true = _normalize_angle(np.arctan2(dy, dx) - theta)

        # --- Sensor visibility checks ---
        # 1. Range check: landmark must be within sensor range
        if r_true > max_range:
            continue

        # 2. Field-of-view check: bearing must be within ±max_bearing
        if abs(phi_true) > max_bearing:
            continue

        # 3. Guard against degenerate case (robot exactly on top of landmark)
        if r_true < 1e-6:
            continue

        # --- Add independent Gaussian noise ---
        r_noisy   = r_true   + rng.normal(0.0, np.sqrt(Q[0, 0]))
        phi_noisy = phi_true + rng.normal(0.0, np.sqrt(Q[1, 1]))

        # Ensure range is physically valid (cannot be negative)
        r_noisy   = max(r_noisy, 1e-6)

        # Normalise bearing after noise addition
        phi_noisy = _normalize_angle(phi_noisy)

        observations.append((lm_id, np.array([r_noisy, phi_noisy])))

    return observations