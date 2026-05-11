'''
EKF SLAM demo
Logic:
    - Prediction update
        - From odometry inputs, how do we change our state estimate?
        - Moving only changes the state estimate of the robot state, NOT landmark location
        - Moving affects uncertainty of the state
    - Observation update
        - From what we observe, how do we change our state estimation?
        - We reconcile prediction uncertainty and observation uncertainty into a single estimate
          that is more certain than before

Key changes vs. vanilla EKF SLAM:
    - Map and landmarks come from get_map_with_obstacles() (as in micro_simulador)
    - prediction_update uses an odometry motion model (rot1/trans/rot2) instead of (v, w)
    - UNKNOWN data association: nearest-neighbour gating in world space
    - 45-degree camera FOV: only +-22.5 deg around robot heading visible
    - State vector and covariance grow dynamically as new landmarks are discovered
    - Live matplotlib visualization + final Procrustes-aligned summary plot
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# <------------------------- MAP SETUP --------------------------------->

def get_map_with_obstacles(option):
    if option == 1:
        map_grid = np.zeros((25, 25), dtype=int)
        map_grid[0, :] = 1;  map_grid[-1, :] = 1
        map_grid[:, 0] = 1;  map_grid[:, -1] = 1
        map_grid[8:20, 12] = 1
        landmarks = np.array([[19,5],[19,15],[12,21],[3,15],[3,5]])
        return map_grid, landmarks
    if option == 2:
        map_grid = np.zeros((25, 25), dtype=int)
        map_grid[0, :] = 1;  map_grid[-1, :] = 1
        map_grid[:, 0] = 1;  map_grid[:, -1] = 1
        map_grid[8:20, 12] = 1
        landmarks = np.array([[19,5],[12,21],[3,5]])
        return map_grid, landmarks
    if option == 3:
        map_grid = np.zeros((25, 25), dtype=int)
        map_grid[0, :] = 1;  map_grid[-1, :] = 1
        map_grid[:, 0] = 1;  map_grid[:, -1] = 1
        map_grid[8:20, 12] = 1
        landmarks = np.array([[12,21]])
        return map_grid, landmarks
    raise ValueError("Opcao invalida.")

# <------------------------- EKF SLAM STUFF --------------------------------->

# ---> Robot / sensor parameters
n_state    = 3                   # Number of robot state variables
robot_fov  = 10.0                # Max detection range (m)
camera_fov = np.deg2rad(45.0)   # Total camera FOV; only +-22.5 deg around heading visible

# ---> Noise parameters
R = np.diag([0.05, 0.05, np.deg2rad(2)]) ** 2  # sigma_x, sigma_y, sigma_theta  (motion)
Q = np.diag([0.05, np.deg2rad(2)])        ** 2  # sigma_r, sigma_phi             (measurement)

# ---> Data-association gate (Mahalanobis^2, chi-squared distributed with 2 DOF)
# 95th percentile of chi2(2) = 5.99 — observations below this threshold match a known landmark.
# New landmarks are only created when ALL known landmarks score above this gate.
ASSOC_GATE = 5.99

# ---> EKF Estimation Variables  (grow dynamically as landmarks are discovered)
mu    = np.zeros(3)      # [x, y, theta,  lm0_x, lm0_y,  lm1_x, lm1_y, ...]
Sigma = np.zeros((3, 3)) # Covariance matrix; expands with every new landmark

# ---> Landmark bookkeeping
landmark_mapping = {}  # { internal_id : index in mu where (lm_x, lm_y) starts }
next_landmark_id = 0

# ---> Helper
def normalize_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

# ---> Measurement function  (unknown correspondence, limited FOV)
def sim_measurement(x, landmarks):
    '''
    Simulate camera measurements. Returns (dist, phi) pairs with NO landmark index —
    correspondence is unknown, mimicking an ArUco camera with a 45-degree FOV.

    Inputs:
     - x         : true robot pose [rx, ry, rtheta]
     - landmarks : Nx2 array of ground-truth landmark positions
    Outputs:
     - zs : list of (dist_noisy, phi_noisy) tuples
    '''
    rx, ry, rtheta = x[0], x[1], x[2]
    zs = []
    for lm in landmarks:
        lx, ly = lm[0], lm[1]
        dist = np.hypot(lx - rx, ly - ry)
        phi  = normalize_angle(np.arctan2(ly - ry, lx - rx) - rtheta)
        if dist < robot_fov and abs(phi) < camera_fov / 2.0:
            dist_noisy = dist + np.random.normal(0, np.sqrt(Q[0, 0]))
            phi_noisy  = normalize_angle(phi + np.random.normal(0, np.sqrt(Q[1, 1])))
            zs.append((dist_noisy, phi_noisy))
    return zs

# ---> EKF SLAM steps
def prediction_update(mu, sigma, u, dt=None):
    '''
    Odometry-based prediction step of the EKF. Replaces the (v, w) velocity model
    with an odometry model decomposed into rot1 / trans / rot2, which is standard
    for wheeled robots with encoder odometry.

    u = (odom_prev, odom_curr) where each is [x, y, theta].
    dt is accepted for interface compatibility but unused.

    Inputs:
     - mu    : state estimate (1-D numpy array, length 3 + 2*N_lm)
     - sigma : state covariance
     - u     : (odom_prev, odom_curr) each a length-3 array/list [x, y, theta]
    Outputs:
     - mu: updated state estimate
     - sigma: updated state uncertainty
    '''
    rx, ry, theta = mu[0], mu[1], mu[2]

    odom_prev, odom_curr = u
    bar_x,  bar_y,  bar_theta  = odom_prev[0], odom_prev[1], odom_prev[2]
    bar_xp, bar_yp, bar_thetap = odom_curr[0], odom_curr[1], odom_curr[2]

    delta_rot1  = normalize_angle(math.atan2(bar_yp - bar_y, bar_xp - bar_x) - bar_theta)
    delta_trans = math.hypot(bar_xp - bar_x, bar_yp - bar_y)
    delta_rot2  = normalize_angle(bar_thetap - bar_theta - delta_rot1)

    # Update state estimate mu with odometry model
    Fx = np.block([np.eye(n_state), np.zeros((n_state, len(mu) - n_state))])
    state_model_mat = np.zeros((n_state, 1))
    state_model_mat[0] = delta_trans * math.cos(theta + delta_rot1)  # delta x
    state_model_mat[1] = delta_trans * math.sin(theta + delta_rot1)  # delta y
    state_model_mat[2] = delta_rot1 + delta_rot2                     # delta theta
    mu = mu + np.matmul(np.transpose(Fx), state_model_mat).flatten()
    mu[2] = normalize_angle(mu[2])

    # Update state uncertainty sigma
    state_jacobian = np.zeros((3, 3))
    state_jacobian[0, 2] = -delta_trans * math.sin(theta + delta_rot1)
    state_jacobian[1, 2] =  delta_trans * math.cos(theta + delta_rot1)
    G = np.eye(sigma.shape[0]) + np.transpose(Fx).dot(state_jacobian).dot(Fx)
    sigma = G.dot(sigma).dot(np.transpose(G)) + np.transpose(Fx).dot(R).dot(Fx)

    return mu, sigma

def measurement_update(mu, sigma, zs):
    '''
    EKF measurement update with unknown data association.

    For each raw (dist, phi) measurement:
      1. Project into world space and find the closest known landmark (nearest-neighbour gate).
      2. If distance > ASSOC_GATE: new landmark — expand mu and sigma dynamically.
      3. Apply standard EKF correction using range-bearing Jacobian.

    Inputs:
     - mu    : state estimate (1-D numpy array)
     - sigma : state covariance
     - zs    : list of (dist, phi) from sim_measurement  (no landmark index)
    Outputs:
     - mu, sigma : updated after all measurements
    '''
    global landmark_mapping, next_landmark_id

    rx, ry, theta = mu[0], mu[1], mu[2]

    for z in zs:
        dist, phi = z

        # --- Data association: nearest neighbour in Mahalanobis distance (measurement space) ---
        # Mahalanobis is computed in (r, phi) space using the innovation covariance S = H*Sigma*H' + Q.
        # This accounts for how uncertain each landmark estimate currently is, so a poorly
        # converged landmark has a large S and is therefore easier to match (wider gate),
        # while a well-converged landmark has a tight S and is harder to match spuriously.
        z_arr = np.array([dist, phi])

        best_id   = None
        best_maha = np.inf
        for lm_id, lm_idx in landmark_mapping.items():
            dx_lm = mu[lm_idx]     - mu[0]
            dy_lm = mu[lm_idx + 1] - mu[1]
            q_lm  = dx_lm**2 + dy_lm**2
            sq_lm = math.sqrt(q_lm)

            # Predicted measurement for this candidate landmark
            dist_est_lm = sq_lm
            phi_est_lm  = normalize_angle(math.atan2(dy_lm, dx_lm) - mu[2])
            z_hat_lm    = np.array([dist_est_lm, phi_est_lm])

            # Jacobian H for this candidate (same structure as correction step)
            n_lm = len(mu)
            H_lm = np.zeros((2, n_lm))
            H_lm[0, 0] = -dx_lm / sq_lm;  H_lm[0, 1] = -dy_lm / sq_lm;  H_lm[0, 2] = 0
            H_lm[1, 0] =  dy_lm / q_lm;   H_lm[1, 1] = -dx_lm / q_lm;   H_lm[1, 2] = -1
            H_lm[0, lm_idx]     =  dx_lm / sq_lm;  H_lm[0, lm_idx + 1] =  dy_lm / sq_lm
            H_lm[1, lm_idx]     = -dy_lm / q_lm;   H_lm[1, lm_idx + 1] =  dx_lm / q_lm

            S_lm  = H_lm @ sigma @ H_lm.T + Q          # innovation covariance (2x2)
            innov_lm    = z_arr - z_hat_lm
            innov_lm[1] = normalize_angle(innov_lm[1])
            maha  = innov_lm @ np.linalg.inv(S_lm) @ innov_lm  # scalar Mahalanobis^2

            if maha < best_maha:
                best_maha = maha
                best_id   = lm_id

        if best_maha > ASSOC_GATE:
            # New landmark — initialise position from current measurement and expand state
            obs_x  = rx + dist * math.cos(theta + phi)
            obs_y  = ry + dist * math.sin(theta + phi)
            lm_idx = len(mu)
            mu     = np.concatenate([mu, [obs_x, obs_y]])
            landmark_mapping[next_landmark_id] = lm_idx
            best_id           = next_landmark_id
            next_landmark_id += 1

            n_old                         = sigma.shape[0]
            sigma_new                     = np.zeros((n_old + 2, n_old + 2))
            sigma_new[:n_old, :n_old]     = sigma
            sigma_new[n_old:, n_old:]     = np.eye(2) * 1e6  # high initial uncertainty
            sigma = sigma_new

        # --- Standard EKF correction ---
        lm_idx = landmark_mapping[best_id]
        n      = len(mu)

        dx = mu[lm_idx]     - mu[0]
        dy = mu[lm_idx + 1] - mu[1]
        q  = dx**2 + dy**2
        sq = math.sqrt(q)

        dist_est = sq
        phi_est  = normalize_angle(math.atan2(dy, dx) - mu[2])
        innov    = np.array([dist - dist_est, normalize_angle(phi - phi_est)])

        H = np.zeros((2, n))
        H[0, 0] = -dx / sq;      H[0, 1] = -dy / sq;  H[0, 2] = 0
        H[1, 0] =  dy / q;       H[1, 1] = -dx / q;   H[1, 2] = -1
        H[0, lm_idx]     =  dx / sq;  H[0, lm_idx + 1] =  dy / sq
        H[1, lm_idx]     = -dy / q;   H[1, lm_idx + 1] =  dx / q

        S = H @ sigma @ H.T + Q
        K = sigma @ H.T @ np.linalg.inv(S)

        mu    = mu + K @ innov
        mu[2] = normalize_angle(mu[2])
        sigma = (np.eye(n) - K @ H) @ sigma

        # Keep local pose in sync for remaining measurements in this step
        rx, ry, theta = mu[0], mu[1], mu[2]

    return mu, sigma
