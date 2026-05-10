"""
Range-bearing measurement model for EKF SLAM.

Reference: Probabilistic Robotics (Thrun, Burgard, Fox) — Table 10.2
           Stachniss KF/EKF Lecture (2020), slides pp. 20-21

A landmark at global position (l_x, l_y) is observed from robot pose
(x, y, theta) as a range-bearing pair:

    z = (r, phi)

    r   = sqrt( delta_x^2 + delta_y^2 )          range (metres)
    phi = atan2(delta_y, delta_x) - theta         bearing (radians, [-pi, pi])

where  delta_x = l_x - x,   delta_y = l_y - y.

The Jacobian H (2 x n) linearises h around the current state estimate.
In its low-dimensional (2 x 5) form (robot pose + one landmark):

    low_H = | -delta_x/r   -delta_y/r    0    delta_x/r   delta_y/r |
            |  delta_y/q   -delta_x/q   -1   -delta_y/q   delta_x/q |

where  q = delta_x^2 + delta_y^2,  r = sqrt(q).

low_H is then projected into the full state space via the selection
matrix F_xj (5 x n):

    H = low_H @ F_xj      shape: (2, n)

Sign convention follows the slides (which correct minor sign errors in the
book). The bearing column for the robot-theta entry is -1 (not +1).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Angle utility — exported so ekf_slam.py can import it from here
# ---------------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """
    Wraps angle to the interval [-pi, pi].

    CRITICAL: must be applied to the bearing component of every innovation
    (z - z_hat) before the Kalman update. Failing to do so causes filter
    divergence when a landmark crosses the ±pi bearing boundary.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Measurement model   h(mu, j)
# ---------------------------------------------------------------------------

def measurement_model(mu: np.ndarray, j: int) -> tuple:
    """
    Predicts the range-bearing observation of landmark j from the current
    state estimate.

    Args:
        mu : Full state vector  [x, y, theta, ..., l_jx, l_jy, ...]  shape (n,)
        j  : Start index of landmark j in mu  (i.e. mu[j], mu[j+1] = l_jx, l_jy)

    Returns:
        z_hat : Predicted measurement  [r_hat, phi_hat]  shape (2,)
        q     : Squared range  delta_x^2 + delta_y^2  (scalar, reused in compute_H)

    Notes:
        - phi_hat is normalised to [-pi, pi].
        - q is returned alongside z_hat to avoid recomputing it in compute_H
          when both are called in sequence (they always are in ekf_update).
    """
    # Unpack robot pose
    x, y, theta = mu[0], mu[1], mu[2]

    # Unpack landmark position
    lx, ly = mu[j], mu[j + 1]

    # Displacement from robot to landmark (in global frame)
    dx = lx - x
    dy = ly - y

    # Range and bearing
    q    = dx ** 2 + dy ** 2           # squared range
    r    = np.sqrt(q)                  # range
    phi  = np.arctan2(dy, dx) - theta  # bearing relative to robot heading
    phi  = normalize_angle(phi)

    z_hat = np.array([r, phi])
    return z_hat, q


# ---------------------------------------------------------------------------
# Measurement Jacobian   H(mu, j, n)
# ---------------------------------------------------------------------------

def compute_H(mu: np.ndarray, j: int, n: int) -> np.ndarray:
    """
    Computes the (2 x n) Jacobian of the measurement model with respect to
    the full state vector, evaluated at the current state estimate mu.

    Args:
        mu : Full state vector  shape (n,)
        j  : Start index of landmark j in mu
        n  : Full state dimension  (= len(mu))

    Returns:
        H  : Measurement Jacobian  shape (2, n)

    Implementation:
        Step 1 — compute delta, q from robot pose and landmark j.
        Step 2 — build low_H (2 x 5): the Jacobian w.r.t. the 5-dimensional
                 subvector [x, y, theta, l_jx, l_jy].
        Step 3 — build F_xj (5 x n): a selection matrix that maps the
                 5-dimensional subvector into the full n-dimensional state.
        Step 4 — H = low_H @ F_xj.
    """
    # --- Step 1: geometry ---
    x, y, theta = mu[0], mu[1], mu[2]
    lx, ly      = mu[j], mu[j + 1]

    dx = lx - x
    dy = ly - y
    q  = dx ** 2 + dy ** 2
    r  = np.sqrt(q)

    # Guard against degenerate case (landmark exactly at robot position)
    if q < 1e-9:
        return np.zeros((2, n))

    # --- Step 2: low-dimensional Jacobian (2 x 5) ---
    # Row 0 : partial of range  r   w.r.t. [x, y, theta, l_jx, l_jy]
    # Row 1 : partial of bearing phi w.r.t. [x, y, theta, l_jx, l_jy]
    #
    # Derived by differentiating:
    #   r   = sqrt((l_jx - x)^2 + (l_jy - y)^2)
    #   phi = atan2(l_jy - y, l_jx - x) - theta
    #
    # Sign convention: slides / Stachniss lecture (corrects book typos).
    low_H = np.array([
        [-dx / r,  -dy / r,   0.0,   dx / r,   dy / r ],
        [ dy / q,  -dx / q,  -1.0,  -dy / q,   dx / q ]
    ])  # shape (2, 5)

    # --- Step 3: selection matrix F_xj (5 x n) ---
    # Maps the 5-dim subvector [robot_pose | landmark_j] into the full state.
    # First 3 rows select robot pose (indices 0,1,2).
    # Last  2 rows select landmark j (indices j, j+1).
    F_xj          = np.zeros((5, n))
    F_xj[0:3, 0:3]      = np.eye(3)      # robot pose block
    F_xj[3:5, j: j + 2] = np.eye(2)      # landmark j block

    # --- Step 4: project into full state space ---
    H = low_H @ F_xj      # shape (2, n)

    return H