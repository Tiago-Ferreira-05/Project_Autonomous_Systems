"""
EKF SLAM — Prediction and Correction steps.

Reference: Probabilistic Robotics (Thrun, Burgard, Fox) — Table 10.2
           Stachniss KF/EKF Lecture (2020)

State vector (grows dynamically as new landmarks are seen):
    mu    = [x, y, theta, l1x, l1y, l2x, l2y, ..., lNx, lNy]   shape (n,)

Covariance matrix:
    Sigma = (n x n) symmetric positive semi-definite matrix
            | Sigma_rr   Sigma_rL |
            | Sigma_Lr   Sigma_LL |

landmark_map : dict { landmark_id -> start_index_in_mu }
    Keeps track of which slice of mu belongs to which physical landmark.

Control input:
    u = (prev_odom, curr_odom)   — odometry motion model (see motion_model.py)

Observations:
    List of (landmark_id, z) where z = [r, phi]  (range, bearing)
"""

import numpy as np

from .motion_model import motion_model, compute_G
from .meas_model   import measurement_model, compute_H, normalize_angle


# =============================================================================
# PREDICTION STEP
# =============================================================================

def ekf_predict(
    mu    : np.ndarray,
    Sigma : np.ndarray,
    u     : tuple,
    R     : np.ndarray,
) -> tuple:
    """
    EKF SLAM prediction step — Probabilistic Robotics Table 10.2, lines 2-3.

    Uses the (nonlinear) motion model to advance the state estimate and
    propagates uncertainty through the linearised model Jacobian G.

    Only the robot-pose block of mu and Sigma is updated; landmark entries
    are unaffected by motion (they don't move).

    Args:
        mu    : Full state vector  [x, y, theta, landmarks...]  shape (n,)
        Sigma : State covariance   shape (n, n)
        u     : Odometry control   (prev_odom, curr_odom)
        R     : Motion noise covariance  shape (3, 3)
                (models uncertainty injected by the motion model at each step)

    Returns:
        mu_bar    : Predicted state vector        shape (n,)
        Sigma_bar : Predicted state covariance    shape (n, n)

    Algorithm:
        F_x     = [I_3 | 0_{3 x 2N}]            selection matrix  (3 x n)
        mu_bar  = g(u, mu)                      nonlinear motion model
        G       = I_n + F_x^T G_x F_x           full-state Jacobian  (n x n)
        Sigma_bar = G Sigma G^T + F_x^T R F_x   propagate uncertainty
    """
    n = len(mu)

    # --- Selection matrix: picks the robot-pose block out of the full state ---
    # F_x shape: (3, n).  F_x @ v  extracts [v[0], v[1], v[2]].
    F_x          = np.zeros((3, n))
    F_x[0:3, 0:3] = np.eye(3)

    # --- Apply nonlinear motion model to robot pose only ---
    # motion_model() updates mu[0:3] and leaves landmark entries unchanged.
    mu_bar = motion_model(mu, u)

    # --- Full-state Jacobian G (n x n) ---
    # compute_G() returns the (n x n) matrix embedding the 3x3 G_x Jacobian.
    G = compute_G(mu, u)

    # --- Propagate covariance ---
    # G Sigma G^T : how previous uncertainty is transformed by the motion.
    # F_x^T R F_x : new uncertainty injected by the motion model noise.
    Sigma_bar = G @ Sigma @ G.T + F_x.T @ R @ F_x

    return mu_bar, Sigma_bar


# =============================================================================
# LANDMARK INITIALISATION
# =============================================================================

def initialise_landmark(
    mu          : np.ndarray,
    Sigma       : np.ndarray,
    lm_id       : int,
    z           : np.ndarray,
    Q           : np.ndarray,
    landmark_map: dict,
) -> tuple:
    """
    Inserts a newly observed landmark into the state vector and covariance
    matrix for the first time.

    The landmark's initial position is computed from the current robot pose
    and the first observation (inverse measurement model).  Its initial
    covariance is propagated from both the robot-pose uncertainty and the
    measurement noise via the Jacobians of the inverse measurement function.

    Reference: Probabilistic Robotics Section 10.2 — EKF SLAM with unknown
               correspondences; blueprint Section 1.8.

    Args:
        mu          : Full state vector before insertion  shape (n,)
        Sigma       : State covariance before insertion   shape (n, n)
        lm_id       : Integer ID of the new landmark (e.g. ArUco marker ID)
        z           : First observation  [r, phi]
        Q           : Measurement noise covariance  shape (2, 2)
        landmark_map: Dict {lm_id -> start_index_in_mu}  — modified in place

    Returns:
        mu_new          : Expanded state vector    shape (n+2,)
        Sigma_new       : Expanded covariance      shape (n+2, n+2)
        landmark_map    : Updated dict with entry for lm_id

    Inverse measurement model:
        l_x = x + r * cos(theta + phi)
        l_y = y + r * sin(theta + phi)

    Jacobians of the inverse measurement model:
        G_x (2x3) — partial w.r.t. robot pose [x, y, theta]:
            | 1   0   -r * sin(theta + phi) |
            | 0   1    r * cos(theta + phi) |

        G_z (2x2) — partial w.r.t. measurement [r, phi]:
            | cos(theta + phi)   -r * sin(theta + phi) |
            | sin(theta + phi)    r * cos(theta + phi) |

    Initial landmark covariance:
        Sigma_ll = G_x @ Sigma_rr @ G_x^T + G_z @ Q @ G_z^T

    Cross-covariances (robot ↔ new landmark):
        Sigma_rL_new = Sigma_rr @ G_x^T
        Sigma_Lr_new = G_x @ Sigma_rr
    """
    r, phi        = z[0], z[1]
    x, y, theta   = mu[0], mu[1], mu[2]
    n_old         = len(mu)

    # --- Inverse measurement model: estimate landmark global position ---
    lx = x + r * np.cos(theta + phi)
    ly = y + r * np.sin(theta + phi)

    # --- Expand state vector ---
    mu_new    = np.append(mu, [lx, ly])   # shape (n_old + 2,)
    n_new     = len(mu_new)
    j         = n_old                     # start index of new landmark in mu

    # Record the mapping landmark_id -> index in mu
    landmark_map[lm_id] = j

    # --- Expand covariance matrix ---
    # Copy the old block into the top-left of the new (n+2 x n+2) matrix.
    Sigma_new                   = np.zeros((n_new, n_new))
    Sigma_new[0:n_old, 0:n_old] = Sigma

    # --- Jacobian of inverse measurement w.r.t. robot pose (2 x 3) ---
    G_x = np.array([
        [1.0,  0.0,  -r * np.sin(theta + phi)],
        [0.0,  1.0,   r * np.cos(theta + phi)],
    ])

    # --- Jacobian of inverse measurement w.r.t. measurement [r, phi] (2 x 2) ---
    G_z = np.array([
        [np.cos(theta + phi),  -r * np.sin(theta + phi)],
        [np.sin(theta + phi),   r * np.cos(theta + phi)],
    ])

    # --- Initial landmark self-covariance ---
    # Propagates robot-pose uncertainty and measurement noise into landmark.
    Sigma_rr = Sigma[0:3, 0:3]
    Sigma_ll  = G_x @ Sigma_rr @ G_x.T + G_z @ Q @ G_z.T

    Sigma_new[j: j + 2, j: j + 2] = Sigma_ll

    # --- Cross-covariance: robot pose ↔ new landmark ---
    # Derived from the Jacobian of the inverse measurement model.
    Sigma_rL = Sigma_rr @ G_x.T              # shape (3, 2)
    Sigma_new[0:3,    j: j + 2] = Sigma_rL
    Sigma_new[j: j+2, 0:3     ] = Sigma_rL.T

    # --- Cross-covariance: existing landmarks ↔ new landmark ---
    # Sigma_LL_new = Sigma_Lr_old @ G_x^T  (off-diagonal blocks, row: old lms)
    if n_old > 3:
        Sigma_old_lm_r = Sigma[3:n_old, 0:3]          # old landmarks vs robot
        Sigma_old_lm_new = Sigma_old_lm_r @ G_x.T     # old landmarks vs new lm

        Sigma_new[3:n_old, j: j + 2] = Sigma_old_lm_new
        Sigma_new[j: j+2,  3:n_old ] = Sigma_old_lm_new.T

    return mu_new, Sigma_new, landmark_map


# =============================================================================
# CORRECTION STEP
# =============================================================================

def ekf_update(
    mu_bar      : np.ndarray,
    Sigma_bar   : np.ndarray,
    observations: list,
    landmark_map: dict,
    Q           : np.ndarray,
) -> tuple:
    """
    EKF SLAM correction step — Probabilistic Robotics Table 10.2, lines 4-19.

    For each observation, the filter:
      1. Initialises the landmark if seen for the first time.
      2. Predicts what the sensor should have seen  (h(mu_bar)).
      3. Computes the innovation  (z - h(mu_bar))  with bearing normalisation.
      4. Applies the Kalman update to mu and Sigma.

    All observations are processed sequentially; mu and Sigma are updated
    incrementally after each one (sequential update, equivalent to batch when
    observations are independent given the state).

    Args:
        mu_bar      : Predicted state vector from ekf_predict()   shape (n,)
        Sigma_bar   : Predicted covariance from ekf_predict()     shape (n, n)
        observations: List of (landmark_id, z)  where z = [r, phi]
        landmark_map: Dict {lm_id -> start_index_in_mu}
        Q           : Measurement noise covariance  shape (2, 2)

    Returns:
        mu          : Corrected state vector    shape (n,)   (may be larger
        Sigma       : Corrected covariance      shape (n, n)  if new landmarks
        landmark_map: Updated landmark mapping             were initialised)

    Kalman update equations (per observation):
        z_hat          = h(mu_bar, j)                   predicted observation
        H              = compute_H(mu_bar, j, n)         Jacobian
        S              = H @ Sigma_bar @ H.T + Q         innovation covariance
        K              = Sigma_bar @ H.T @ inv(S)        Kalman gain
        innovation     = z - z_hat                       (bearing normalised!)
        mu_bar         = mu_bar + K @ innovation
        Sigma_bar      = (I - K @ H) @ Sigma_bar
    """
    # Work on mutable copies so we can update in-place across observations
    mu    = mu_bar.copy()
    Sigma = Sigma_bar.copy()

    for (lm_id, z) in observations:

        # -----------------------------------------------------------------
        # 1. Initialise landmark if seen for the first time
        # -----------------------------------------------------------------
        if lm_id not in landmark_map:
            mu, Sigma, landmark_map = initialise_landmark(
                mu, Sigma, lm_id, z, Q, landmark_map
            )
            # After initialisation the landmark is in the state but has high
            # uncertainty — we skip the update for this observation so we
            # don't immediately over-correct with poor geometry.
            continue

        # -----------------------------------------------------------------
        # 2. Retrieve landmark index in mu
        # -----------------------------------------------------------------
        j = landmark_map[lm_id]
        n = len(mu)

        # -----------------------------------------------------------------
        # 3. Predicted observation  z_hat = h(mu, j)
        # -----------------------------------------------------------------
        z_hat, _q = measurement_model(mu, j)

        # -----------------------------------------------------------------
        # 4. Measurement Jacobian H  (2 x n)
        # -----------------------------------------------------------------
        H = compute_H(mu, j, n)

        # -----------------------------------------------------------------
        # 5. Innovation covariance  S = H Sigma H^T + Q
        # -----------------------------------------------------------------
        S = H @ Sigma @ H.T + Q

        # -----------------------------------------------------------------
        # 6. Kalman gain  K = Sigma H^T S^{-1}
        # -----------------------------------------------------------------
        K = Sigma @ H.T @ np.linalg.inv(S)

        # -----------------------------------------------------------------
        # 7. Innovation  (z - z_hat) with bearing normalisation
        #    CRITICAL: the bearing component must be wrapped to [-pi, pi]
        #    to avoid jumps at the ±pi boundary causing filter divergence.
        # -----------------------------------------------------------------
        z      = np.asarray(z, dtype=float)
        innov  = z - z_hat
        innov[1] = normalize_angle(innov[1])

        # -----------------------------------------------------------------
        # 8. State update
        # -----------------------------------------------------------------
        mu = mu + K @ innov

        # Keep robot heading normalised after the update
        mu[2] = normalize_angle(mu[2])

        # -----------------------------------------------------------------
        # 9. Covariance update  Sigma = (I - K H) Sigma
        #    Note: the Joseph form  (I-KH) Sigma (I-KH)^T + K Q K^T
        #    is numerically more stable and guarantees symmetry; consider
        #    switching to it if you observe Sigma becoming non-PSD.
        # -----------------------------------------------------------------
        I_n   = np.eye(n)
        Sigma = (I_n - K @ H) @ Sigma

    return mu, Sigma, landmark_map