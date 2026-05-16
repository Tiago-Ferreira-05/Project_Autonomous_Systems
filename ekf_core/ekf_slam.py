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

Changelog (Week 4 — real data):
    [FIX] ekf_update() now applies a Mahalanobis chi-squared gate before
          every Kalman update.  The threshold is passed in by the caller
          via the 'mahal_threshold' argument (default 9.21, chi2(2, 0.99)).
          Bad ArUco detections that would otherwise corrupt the map are
          rejected before they reach the state update.
          cfg.MAHALANOBIS_THRESHOLD was already defined — it is now used.
"""

import numpy as np

from .motion_model import motion_model, compute_G
from .meas_model   import measurement_model, compute_H, normalize_angle


# =============================================================================
# PREDICTION STEP  — unchanged
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

    Returns:
        mu_bar    : Predicted state vector        shape (n,)
        Sigma_bar : Predicted state covariance    shape (n, n)
    """
    n = len(mu)

    F_x           = np.zeros((3, n))
    F_x[0:3, 0:3] = np.eye(3)

    mu_bar    = motion_model(mu, u)
    G         = compute_G(mu, u)
    Sigma_bar = G @ Sigma @ G.T + F_x.T @ R @ F_x

    return mu_bar, Sigma_bar


# =============================================================================
# LANDMARK INITIALISATION  — unchanged
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

    Args:
        mu          : Full state vector before insertion  shape (n,)
        Sigma       : State covariance before insertion   shape (n, n)
        lm_id       : Integer ID of the new landmark (e.g. ArUco marker ID)
        z           : First observation  [r, phi]
        Q           : Measurement noise covariance  shape (2, 2)
        landmark_map: Dict {lm_id -> start_index_in_mu}  — modified in place

    Returns:
        mu_new       : Expanded state vector    shape (n+2,)
        Sigma_new    : Expanded covariance      shape (n+2, n+2)
        landmark_map : Updated dict with entry for lm_id

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

    Cross-covariances (robot <-> new landmark):
        Sigma_rL = Sigma_rr @ G_x^T
    """
    r, phi      = z[0], z[1]
    x, y, theta = mu[0], mu[1], mu[2]
    n_old       = len(mu)

    # Inverse measurement model: estimate landmark global position
    lx = x + r * np.cos(theta + phi)
    ly = y + r * np.sin(theta + phi)

    # Expand state vector
    mu_new = np.append(mu, [lx, ly])
    n_new  = len(mu_new)
    j      = n_old

    landmark_map[lm_id] = j

    # Expand covariance matrix — copy existing block into top-left
    Sigma_new                   = np.zeros((n_new, n_new))
    Sigma_new[0:n_old, 0:n_old] = Sigma

    # Jacobian of inverse measurement w.r.t. robot pose (2 x 3)
    G_x = np.array([
        [1.0,  0.0,  -r * np.sin(theta + phi)],
        [0.0,  1.0,   r * np.cos(theta + phi)],
    ])

    # Jacobian of inverse measurement w.r.t. measurement [r, phi] (2 x 2)
    G_z = np.array([
        [np.cos(theta + phi),  -r * np.sin(theta + phi)],
        [np.sin(theta + phi),   r * np.cos(theta + phi)],
    ])

    # Initial landmark self-covariance
    Sigma_rr = Sigma[0:3, 0:3]
    Sigma_ll = G_x @ Sigma_rr @ G_x.T + G_z @ Q @ G_z.T
    Sigma_new[j: j + 2, j: j + 2] = Sigma_ll

    # Cross-covariance: robot pose <-> new landmark
    Sigma_rL = Sigma_rr @ G_x.T
    Sigma_new[0:3,   j: j + 2] = Sigma_rL
    Sigma_new[j:j+2, 0:3     ] = Sigma_rL.T

    # Cross-covariance: existing landmarks <-> new landmark
    if n_old > 3:
        Sigma_old_lm_r   = Sigma[3:n_old, 0:3]
        Sigma_old_lm_new = Sigma_old_lm_r @ G_x.T
        Sigma_new[3:n_old, j: j + 2] = Sigma_old_lm_new
        Sigma_new[j: j+2,  3:n_old ] = Sigma_old_lm_new.T

    return mu_new, Sigma_new, landmark_map


# =============================================================================
# CORRECTION STEP
# =============================================================================

def ekf_update(
    mu_bar         : np.ndarray,
    Sigma_bar      : np.ndarray,
    observations   : list,
    landmark_map   : dict,
    Q              : np.ndarray,
    mahal_threshold: float = 9.21,
) -> tuple:
    """
    EKF SLAM correction step — Probabilistic Robotics Table 10.2, lines 4-19.

    For each observation the filter:
      1. Initialises the landmark if seen for the first time (skips update).
      2. Predicts what the sensor should have seen  (h(mu_bar)).
      3. Computes the innovation  (z - h(mu_bar)) with bearing normalisation.
      4. [FIX] Gates the observation with a Mahalanobis chi-squared test.
      5. Applies the Kalman update to mu and Sigma if the gate passes.

    Args:
        mu_bar         : Predicted state vector from ekf_predict()   shape (n,)
        Sigma_bar      : Predicted covariance from ekf_predict()     shape (n, n)
        observations   : List of (landmark_id, z)  where z = [r, phi]
        landmark_map   : Dict {lm_id -> start_index_in_mu}
        Q              : Measurement noise covariance  shape (2, 2)
        mahal_threshold: Chi-squared gate threshold.
                         Default 9.21 = chi2(2, 0.99) — rejects the worst 1%
                         of observations under a Gaussian noise model.
                         Pass cfg.MAHALANOBIS_THRESHOLD from the caller.

    Returns:
        mu          : Corrected state vector    shape (n,)   (may be larger
        Sigma       : Corrected covariance      shape (n, n)  if new landmarks
        landmark_map: Updated landmark mapping              were initialised)

    Kalman update equations (per observation):
        z_hat    = h(mu, j)                      predicted observation
        H        = compute_H(mu, j, n)           Jacobian  (2 x n)
        S        = H Sigma H^T + Q               innovation covariance  (2 x 2)
        innov    = z - z_hat                     innovation (bearing normalised)
        d_mah^2  = innov^T S^{-1} innov          Mahalanobis distance squared
        K        = Sigma H^T S^{-1}              Kalman gain
        mu       = mu + K innov
        Sigma    = (I - K H) Sigma
    """
    mu    = mu_bar.copy()
    Sigma = Sigma_bar.copy()

    for (lm_id, z) in observations:

        # -----------------------------------------------------------------
        # 1. Initialise landmark if seen for the first time.
        #    The landmark position is set directly from this observation
        #    so the innovation would be near-zero — skip the update.
        # -----------------------------------------------------------------
        if lm_id not in landmark_map:
            mu, Sigma, landmark_map = initialise_landmark(
                mu, Sigma, lm_id, z, Q, landmark_map
            )
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
        # 6. Innovation  (z - z_hat) with bearing normalisation
        #    CRITICAL: bearing must be wrapped to [-pi, pi] to prevent
        #    filter divergence at the ±pi boundary.
        # -----------------------------------------------------------------
        z      = np.asarray(z, dtype=float)
        innov  = z - z_hat
        innov[1] = normalize_angle(innov[1])

        # -----------------------------------------------------------------
        # 7. [FIX] Mahalanobis chi-squared gate
        #
        #    Computes the normalised distance between the actual and
        #    predicted observation.  Under correct Gaussian assumptions
        #    this follows a chi-squared distribution with 2 DOF.
        #
        #    Gate: innov^T @ S^{-1} @ innov  >  mahal_threshold
        #
        #    chi2(2, 0.99) = 9.21  →  rejects the worst 1% of observations.
        #    Observations that fail the gate are outliers — bad ArUco
        #    detections, partial occlusions, or data-association errors.
        #    Letting them through would corrupt the map irreversibly.
        # -----------------------------------------------------------------
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # S is singular — degenerate measurement geometry, skip.
            continue

        mahal_sq = float(innov.T @ S_inv @ innov)

        if mahal_sq > mahal_threshold:
            continue   # outlier — discard, do not update state or covariance

        # -----------------------------------------------------------------
        # 8. Kalman gain  K = Sigma H^T S^{-1}
        # -----------------------------------------------------------------
        K = Sigma @ H.T @ S_inv

        # -----------------------------------------------------------------
        # 9. State update
        # -----------------------------------------------------------------
        mu    = mu + K @ innov
        mu[2] = normalize_angle(mu[2])

        # -----------------------------------------------------------------
        # 10. Covariance update  Sigma = (I - K H) Sigma
        # -----------------------------------------------------------------
        I_n   = np.eye(n)
        Sigma = (I_n - K @ H) @ Sigma

    return mu, Sigma, landmark_map
