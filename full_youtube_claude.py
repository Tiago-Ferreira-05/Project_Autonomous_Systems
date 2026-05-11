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

# <------------------------- PLOTTING STUFF --------------------------------->

def plot_covariance_ellipse(ax, mean, cov, n_std=1.0, edgecolor='blue'):
    '''
    Draw a 2-D covariance ellipse on ax.
    '''
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order     = eigenvals.argsort()[::-1]
    eigenvals = np.maximum(eigenvals[order], 0)
    eigenvecs = eigenvecs[:, order]
    angle     = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(eigenvals)
    ellipse = patches.Ellipse(xy=mean, width=width, height=height,
                              angle=angle, facecolor='none',
                              edgecolor=edgecolor, linestyle='--', linewidth=1)
    ax.add_patch(ellipse)

def show_robot_estimate(mu, sigma, ax):
    '''
    Visualize estimated robot position and uncertainty ellipse.
    '''
    ax.plot(mu[0], mu[1], 'bs', markersize=8)
    plot_covariance_ellipse(ax, mu[0:2], sigma[0:2, 0:2], edgecolor='blue')

def show_landmark_estimate(mu, sigma, ax):
    '''
    Visualize estimated landmark positions (red stars) and uncertainty ellipses (orange).
    '''
    for lm_id, lm_idx in landmark_mapping.items():
        lx     = mu[lm_idx]
        ly     = mu[lm_idx + 1]
        lsigma = sigma[lm_idx:lm_idx+2, lm_idx:lm_idx+2]
        ax.plot(lx, ly, 'r*', markersize=10)
        plot_covariance_ellipse(ax, [lx, ly], lsigma, edgecolor='orange')

def show_landmark_location(landmarks, ax):
    '''
    Visualize ground-truth landmark positions (black stars).
    '''
    ax.plot(landmarks[:, 0], landmarks[:, 1], 'k*', markersize=14, label='True Landmarks')

def show_measurements(x, zs, ax):
    '''
    Draw lines from robot to each raw measurement endpoint.
    '''
    rx, ry, rtheta = x[0], x[1], x[2]
    for z in zs:
        dist, phi = z
        lx = rx + dist * math.cos(phi + rtheta)
        ly = ry + dist * math.sin(phi + rtheta)
        ax.plot([rx, lx], [ry, ly], color='gray', linewidth=0.7, alpha=0.5)

def draw_map(map_grid, ax):
    '''
    Draw occupied grid cells as black squares.
    '''
    rows, cols = map_grid.shape
    for i in range(rows):
        for j in range(cols):
            if map_grid[i, j] == 1:
                world_y = rows - 1 - i  # row 0 = top of world
                ax.add_patch(patches.Rectangle((j, world_y), 1, 1, color='black'))

def draw_robot_arrow(x, ax):
    '''
    Draw robot position and heading arrow.
    '''
    rx, ry, rtheta = x[0], x[1], x[2]
    ax.plot(rx, ry, 'bo', markersize=8)
    ax.arrow(rx, ry, 0.6*math.cos(rtheta), 0.6*math.sin(rtheta),
             head_width=0.3, head_length=0.2, fc='blue', ec='blue')

def draw_camera_fov(x, ax):
    '''
    Draw the 45-degree camera FOV wedge.
    '''
    rx, ry, rtheta = x[0], x[1], x[2]
    half   = camera_fov / 2.0
    angles = np.linspace(rtheta - half, rtheta + half, 20)
    for a in angles:
        ax.plot([rx, rx + robot_fov*math.cos(a)],
                [ry, ry + robot_fov*math.sin(a)],
                color='lime', linewidth=0.4, alpha=0.4)

# <------------------------- PLOTTING STUFF --------------------------------->


if __name__ == '__main__':

    # ---> Choose map
    print("Escolhe o mapa:")
    print("1 - c/ 5 landmarks")
    print("2 - c/ 3 landmarks")
    print("3 - c/ 1 landmark")
    map_choice = int(input("Mapa: "))
    map_grid, landmarks_true = get_map_with_obstacles(map_choice)

    # ---> Predefined path (two laps, identical to micro_simulador)
    predefined_path = []
    for x in range(2, 17):       predefined_path.append([x,    2.0,   0.0])
    for y in range(3, 21):       predefined_path.append([16.0, y,     np.pi/2])
    for x in range(15, 4, -1):   predefined_path.append([x,    20.0,  np.pi])
    for y in range(19, 2, -1):   predefined_path.append([5.0,  y,    -np.pi/2])
    for x in range(5, 17):       predefined_path.append([x,    2.0,   0.0])
    for y in range(3, 21):       predefined_path.append([16.0, y,     np.pi/2])
    for x in range(15, 4, -1):   predefined_path.append([x,    20.0,  np.pi])
    for y in range(19, 2, -1):   predefined_path.append([5.0,  y,    -np.pi/2])

    # ---> Initialise EKF state from first waypoint
    mu[0:3]    = predefined_path[0]
    Sigma[:,:] = 0.0  # known starting pose

    # ---> Path history
    true_path  = []
    robot_path = []

    # ---> Live figure
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    # ──────────────────────────── MAIN LOOP ──────────────────────────────────
    for t in range(len(predefined_path) - 1):

        true_pose = predefined_path[t + 1]
        true_path.append(true_pose)
        odom_prev = predefined_path[t]

        # Simulate noisy odometry
        dx     = true_pose[0] - odom_prev[0]
        dy     = true_pose[1] - odom_prev[1]
        dtheta = normalize_angle(true_pose[2] - odom_prev[2])
        odom_curr = [
            odom_prev[0] + dx     + np.random.normal(0, math.sqrt(R[0, 0])),
            odom_prev[1] + dy     + np.random.normal(0, math.sqrt(R[1, 1])),
            normalize_angle(odom_prev[2] + dtheta + np.random.normal(0, math.sqrt(R[2, 2]))),
        ]

        u = (odom_prev, odom_curr)

        # Measurements (unknown correspondence, 45-deg FOV)
        zs = sim_measurement(true_pose, landmarks_true)

        # EKF SLAM
        mu, Sigma = prediction_update(mu, Sigma, u)
        mu, Sigma = measurement_update(mu, Sigma, zs)

        robot_path.append(mu[0:3].copy())

        # Live plot every 5 steps
        if t % 5 == 0:
            ax.clear()
            ax.set_title(f'EKF SLAM  —  step {t}')
            ax.set_xlim(0, 25);  ax.set_ylim(0, 25)
            ax.set_aspect('equal');  ax.grid(True)

            draw_map(map_grid, ax)
            draw_camera_fov(true_pose, ax)
            show_measurements(true_pose, zs, ax)

            tp = np.array(true_path)
            rp = np.array(robot_path)
            ax.plot(tp[:, 0], tp[:, 1], 'g--', linewidth=1,   label='True path')
            ax.plot(rp[:, 0], rp[:, 1], 'b-',  linewidth=1,   label='Estimated path')

            draw_robot_arrow(true_pose, ax)
            show_landmark_location(landmarks_true, ax)   # black stars
            show_landmark_estimate(mu, Sigma, ax)         # red stars + orange ellipses
            show_robot_estimate(mu, Sigma, ax)            # blue square + ellipse

            ax.legend(loc='upper right', fontsize=8)
            plt.pause(0.01)

    plt.ioff()

    # ──────────────────────── PROCRUSTES ALIGNMENT ───────────────────────────
    # Collect estimated positions for as many landmarks as we have seen,
    # matched in order of first observation vs. ground-truth order.
    lm_ids_seen = sorted(landmark_mapping.keys())
    n_align     = min(len(lm_ids_seen), len(landmarks_true))

    if n_align < 2:
        print("Not enough landmarks observed for Procrustes alignment.")
        plt.show()
    else:
        landmarks_estimated = np.array([
            mu[landmark_mapping[lm_id]:landmark_mapping[lm_id]+2]
            for lm_id in lm_ids_seen[:n_align]
        ])
        landmarks_ref = landmarks_true[:n_align].astype(float)

        # Procrustes: centre, SVD, recover scale + rotation + translation
        mu_est = landmarks_estimated.mean(axis=0)
        mu_ref = landmarks_ref.mean(axis=0)
        A      = landmarks_estimated - mu_est
        B      = landmarks_ref       - mu_ref
        M      = B.T @ A
        U, s, Vt = np.linalg.svd(M)
        d          = np.linalg.det(U @ Vt)
        rotation   = U @ np.diag([1, d]) @ Vt       # 2x2 rotation
        scale      = s.sum() / (A ** 2).sum()
        translation = mu_ref - scale * (rotation @ mu_est)

        aligned_landmarks = scale * (rotation @ landmarks_estimated.T).T + translation
        rp_np             = np.array(robot_path)[:, :2]
        aligned_path      = scale * (rotation @ rp_np.T).T + translation

        covariances_aligned = []
        for lm_id in lm_ids_seen[:n_align]:
            lm_idx = landmark_mapping[lm_id]
            cov    = Sigma[lm_idx:lm_idx+2, lm_idx:lm_idx+2]
            covariances_aligned.append(scale**2 * rotation @ cov @ rotation.T)

        # Final plot
        fig2, ax2 = plt.subplots(figsize=(9, 9))
        ax2.set_title('EKF SLAM — Procrustes-Aligned Result')
        ax2.set_aspect('equal');  ax2.grid(True)

        draw_map(map_grid, ax2)

        ax2.plot(landmarks_ref[:, 0], landmarks_ref[:, 1],
                 'k*', markersize=14, label='True Landmarks')
        ax2.plot(aligned_landmarks[:, 0], aligned_landmarks[:, 1],
                 'r*', markersize=10,  label='Estimated Landmarks (aligned)')
        for mean, cov in zip(aligned_landmarks, covariances_aligned):
            plot_covariance_ellipse(ax2, mean, cov, edgecolor='orange')

        true_path_np = np.array(true_path)[:, :2]
        ax2.plot(true_path_np[:, 0],  true_path_np[:, 1],
                 'g--', linewidth=1.5, label='True Path')
        ax2.plot(aligned_path[:, 0],  aligned_path[:, 1],
                 'r-',  linewidth=1.5, label='Estimated Path (aligned)')

        ax2.legend(loc='upper right')
        plt.show()