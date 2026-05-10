"""
Odometry motion model for EKF SLAM.

Reference: Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 5, Table 5.6
           Stachniss KF/EKF Lecture (2020)

The robot state is x = [x, y, theta]^T.

Control input is a pair of consecutive odometry readings:
    u = (prev_odom, curr_odom)   where each is [x_bar, y_bar, theta_bar]

The model decomposes the motion into three primitives:
    delta_rot1  — initial rotation to face the new position
    delta_trans — straight-line translation
    delta_rot2  — final rotation to reach the new heading

Predicted next pose:
    x'     = x     + delta_trans * cos(theta + delta_rot1)
    y'     = y     + delta_trans * sin(theta + delta_rot1)
    theta' = theta + delta_rot1 + delta_rot2

Jacobian G_x (3x3) — partial of motion model w.r.t. robot pose:
    G_x = | 1   0   -delta_trans * sin(theta + delta_rot1) |
          | 0   1    delta_trans * cos(theta + delta_rot1) |
          | 0   0    1                                     |

In EKF SLAM, G_x is embedded in the full-state Jacobian G (n x n):
    G = I_n + F_x^T * (G_x - I_3) * F_x

where F_x = [I_3 | 0_{3 x 2N}] selects the robot-pose block.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Angle utility (used internally and by ekf_slam.py)
# ---------------------------------------------------------------------------

def _normalize_angle(angle: float) -> float:
    """
    Wraps a single angle to the interval [-pi, pi].

    Used internally to keep delta_rot1, delta_rot2, and theta' bounded.
    This prevents numerical issues when angles cross the ±pi boundary.
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Odometry primitive decomposition (shared by both public functions)
# ---------------------------------------------------------------------------

def _odometry_primitives(u: tuple) -> tuple:
    """
    Decomposes a pair of odometry readings into the three motion primitives.

    This computation is shared between motion_model() and compute_G() to
    avoid code duplication and guarantee consistency.

    Args:
        u : Tuple (prev_odom, curr_odom).
            Each is a length-3 array-like [x_bar, y_bar, theta_bar].

    Returns:
        delta_rot1  : Initial rotation to face the new position  (rad)
        delta_trans : Straight-line translation distance         (m)
        delta_rot2  : Final rotation to reach the new heading    (rad)

    Notes:
        - All returned angles are normalised to [-pi, pi].
        - When delta_trans is effectively zero (pure rotation), delta_rot1
          is set to 0 and all rotation is absorbed into delta_rot2 to avoid
          a numerically undefined atan2(0, 0).
    """
    prev_odom, curr_odom = u
    prev_x, prev_y, prev_theta = prev_odom[0], prev_odom[1], prev_odom[2]
    curr_x, curr_y, curr_theta = curr_odom[0], curr_odom[1], curr_odom[2]

    dx = curr_x - prev_x
    dy = curr_y - prev_y

    delta_trans = np.sqrt(dx ** 2 + dy ** 2)

    # When the robot barely translates, atan2 is ill-conditioned.
    # Absorb any rotation into delta_rot2 only.
    if delta_trans < 1e-9:
        delta_rot1 = 0.0
    else:
        delta_rot1 = _normalize_angle(np.arctan2(dy, dx) - prev_theta)

    delta_rot2 = _normalize_angle(curr_theta - prev_theta - delta_rot1)

    return delta_rot1, delta_trans, delta_rot2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def motion_model(mu: np.ndarray, u: tuple) -> np.ndarray:
    """
    Applies the odometry motion model to the full state vector.

    Only the robot-pose block (mu[0:3]) is updated; all landmark entries
    (mu[3:]) are left unchanged because landmarks do not move.

    Args:
        mu : Full state vector  [x, y, theta, l1x, l1y, ...]  shape (n,)
        u  : Tuple (prev_odom, curr_odom), each a length-3 array [x, y, theta]
             as reported by the wheel encoders / odometry topic.

    Returns:
        mu_bar : Updated state vector  shape (n,).
                 mu_bar[0:3] holds the predicted robot pose.
                 mu_bar[3:]  is identical to mu[3:].
    """
    # --- Step 1: decompose odometry into three primitives ---
    delta_rot1, delta_trans, delta_rot2 = _odometry_primitives(u)

    # --- Step 2: extract current robot pose from state vector ---
    x, y, theta = mu[0], mu[1], mu[2]

    # --- Step 3: apply the motion model equations ---
    # These come directly from Probabilistic Robotics Table 5.6.
    x_new     = x     + delta_trans * np.cos(theta + delta_rot1)
    y_new     = y     + delta_trans * np.sin(theta + delta_rot1)
    theta_new = _normalize_angle(theta + delta_rot1 + delta_rot2)

    # --- Step 4: build output — copy mu, update only the robot-pose block ---
    mu_bar        = mu.copy()
    mu_bar[0:3]   = [x_new, y_new, theta_new]

    return mu_bar


def compute_G(mu: np.ndarray, u: tuple) -> np.ndarray:
    """
    Computes the full-state Jacobian G of the motion model  (n x n).

    G linearises the nonlinear motion model g around the current state
    estimate mu. It describes how small perturbations in the previous state
    propagate through the motion.

    The landmark rows/columns of G are identity because landmarks are
    unaffected by the robot's motion.

    Args:
        mu : Full state vector  [x, y, theta, l1x, l1y, ...]  shape (n,)
        u  : Tuple (prev_odom, curr_odom), same format as motion_model().

    Returns:
        G  : Full-state Jacobian  shape (n, n).

    Derivation:
        The 3x3 Jacobian of g w.r.t. the robot pose is:

            G_x = d/d[x,y,θ] [ x + δ_t·cos(θ+δ_r1),
                                y + δ_t·sin(θ+δ_r1),
                                θ + δ_r1 + δ_r2    ]

                = | 1   0   -δ_t · sin(θ + δ_r1) |
                  | 0   1    δ_t · cos(θ + δ_r1) |
                  | 0   0    1                    |

        Only the (0,2) and (1,2) entries are non-trivial (the theta column).

        The full n×n Jacobian is built by embedding G_x via:

            G = I_n  +  F_x^T @ (G_x - I_3) @ F_x

        The (G_x - I_3) trick avoids double-counting the identity that is
        already in I_n. This is equivalent to the formulation in Table 10.2
        of Probabilistic Robotics.
    """
    n = len(mu)

    # --- Step 1: reuse shared odometry primitive decomposition ---
    delta_rot1, delta_trans, _ = _odometry_primitives(u)

    # --- Step 2: current robot heading ---
    theta = mu[2]

    # --- Step 3: build the 3x3 robot-pose Jacobian G_x ---
    # Only the third column (theta partial) has non-identity entries.
    G_x        = np.eye(3)
    G_x[0, 2]  = -delta_trans * np.sin(theta + delta_rot1)
    G_x[1, 2]  =  delta_trans * np.cos(theta + delta_rot1)
    # G_x[2, 2] = 1  (already set by np.eye)

    # --- Step 4: build the selection matrix F_x (3 x n) ---
    # F_x picks out the robot-pose subvector from the full state vector.
    # F_x @ v = v[0:3]   (robot pose block)
    F_x           = np.zeros((3, n))
    F_x[0:3, 0:3] = np.eye(3)

    # --- Step 5: embed G_x into the full-state n x n Jacobian ---
    # G = I_n + F_x^T @ (G_x - I_3) @ F_x
    # (G_x - I_3) isolates the non-trivial part to avoid double-counting I.
    G = np.eye(n) + F_x.T @ (G_x - np.eye(3)) @ F_x

    return G