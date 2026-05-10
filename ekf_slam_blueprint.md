# EKF SLAM — Full Project Blueprint
### Autonomous Systems | IST Lisboa | 2025/26
### Reference: Probabilistic Robotics (Thrun, Burgard, Fox) + Stachniss KF/EKF Lecture

---

## PART 1 — THEORY

---

### 1.1 The State Estimation Problem

A mobile robot navigating an unknown environment must answer two questions simultaneously:
- **Where am I?** (Localization)
- **What does the world look like?** (Mapping)

This is the **SLAM problem**. It is hard because both answers depend on each other:
- To build a good map you need to know where you are
- To know where you are you need a good map

The key insight is to treat both the robot pose and the landmark positions as **random variables** with uncertainty, and **maintain a joint probability distribution** over all of them simultaneously.

---

### 1.2 Probability Foundations

The robot's belief at time t is a probability distribution:

```
bel(x_t) = p(x_t | z_1:t, u_1:t)
```

Where:
- `x_t` = robot state at time t
- `z_1:t` = all observations up to time t
- `u_1:t` = all control inputs up to time t

This is maintained recursively using two steps every timestep:

**Prediction (motion update):**
```
bel_bar(x_t) = ∫ p(x_t | u_t, x_{t-1}) · bel(x_{t-1}) dx_{t-1}
```
"Where might I be now, given where I was and what I did?"

**Correction (measurement update):**
```
bel(x_t) = η · p(z_t | x_t) · bel_bar(x_t)
```
"Given what I observed, how should I update my belief?"

This is the **Bayes Filter** — the foundation of all probabilistic robot state estimation.

---

### 1.3 The Kalman Filter (KF)

The Kalman Filter is a Bayes Filter with one critical assumption:

> **Everything is Gaussian and linear.**

This means:
- The belief is represented as a **Gaussian**: `bel(x) = N(μ, Σ)`
- The motion model is **linear**: `x_t = A·x_{t-1} + B·u_t + noise`
- The measurement model is **linear**: `z_t = C·x_t + noise`
- All noise is **zero-mean Gaussian**

Under these assumptions, the Bayes Filter has a **closed-form analytical solution**.

**State representation:**
```
μ  = mean vector (best estimate)
Σ  = covariance matrix (uncertainty)
```

**KF Algorithm (Probabilistic Robotics Table 3.1):**

```
Prediction step:
  μ_bar = A·μ + B·u
  Σ_bar = A·Σ·A^T + R          (R = motion noise covariance)

Correction step:
  K     = Σ_bar·C^T · (C·Σ_bar·C^T + Q)^{-1}   (Kalman Gain)
  μ     = μ_bar + K·(z - C·μ_bar)               (state update)
  Σ     = (I - K·C)·Σ_bar                        (covariance update)
```

**Intuition for the Kalman Gain K:**
- If sensor noise Q is small → K is large → trust the measurement more
- If prediction uncertainty Σ_bar is small → K is small → trust the model more
- K always trades off model confidence vs sensor confidence optimally

---

### 1.4 The Extended Kalman Filter (EKF)

Real robots are **nonlinear**. The motion model and measurement model both involve trigonometric functions (sin, cos, atan2). The standard KF cannot handle this.

The EKF approximates nonlinear functions by **linearising them around the current estimate** using a first-order Taylor expansion (Jacobian).

**For the motion model g(u, x):**
```
x_t ≈ g(u_t, μ_{t-1}) + G_t · (x_{t-1} - μ_{t-1})
```
Where `G_t = dg/dx |_{μ_{t-1}}` is the **Jacobian of g** w.r.t. the state.

**For the measurement model h(x):**
```
z_t ≈ h(μ_bar_t) + H_t · (x_t - μ_bar_t)
```
Where `H_t = dh/dx |_{μ_bar_t}` is the **Jacobian of h** w.r.t. the state.

**EKF Algorithm (Probabilistic Robotics Table 3.3):**

```
Prediction step:
  μ_bar = g(u_t, μ_{t-1})
  Σ_bar = G_t · Σ · G_t^T + R

Correction step:
  K     = Σ_bar · H_t^T · (H_t · Σ_bar · H_t^T + Q)^{-1}
  μ     = μ_bar + K · (z - h(μ_bar))
  Σ     = (I - K · H_t) · Σ_bar
```

**Key difference from KF:**
- `g` and `h` replace the linear `A·x` and `C·x`
- `G` and `H` (Jacobians) replace the linear matrices `A` and `C`
- Everything else is structurally identical

**Important limitation:** Linearisation introduces errors when the true function is highly nonlinear or when uncertainty is large. The EKF is an approximation — it is not optimal for nonlinear systems.

---

### 1.5 The Odometry Motion Model

The robot state is `x = (x, y, θ)`. The odometry motion model uses **wheel encoder readings** as the control input.

**Input:** `u = (prev_odom, curr_odom)` where each is `(x̄, ȳ, θ̄)` from encoders.

**Decomposition into 3 primitives** (Probabilistic Robotics Ch. 5):
```
δ_rot1  = atan2(ȳ' - ȳ, x̄' - x̄) - θ̄       (initial rotation)
δ_trans = sqrt((x̄'-x̄)² + (ȳ'-ȳ)²)           (translation)
δ_rot2  = θ̄' - θ̄ - δ_rot1                    (final rotation)
```

**Predicted next pose:**
```
x'  = x + δ_trans · cos(θ + δ_rot1)
y'  = y + δ_trans · sin(θ + δ_rot1)
θ'  = θ + δ_rot1 + δ_rot2
```

**Jacobian G_x** (3×3, partial of motion model w.r.t. robot pose):
```
G_x = | 1  0  -δ_trans·sin(θ + δ_rot1) |
      | 0  1   δ_trans·cos(θ + δ_rot1) |
      | 0  0   1                        |
```

In EKF SLAM, G is embedded in the full state Jacobian, affecting only the robot pose subblock — landmarks are not directly affected by motion (they don't move).

---

### 1.6 The Range-Bearing Measurement Model

Landmarks are observed as `z = (r, φ)` where:
- `r` = range (distance to landmark)
- `φ` = bearing (angle to landmark relative to robot heading)

**Predicted measurement** for landmark at `(l_x, l_y)` and robot at `(x, y, θ)`:
```
δx  = l_x - x
δy  = l_y - y
r̂   = sqrt(δx² + δy²)
φ̂   = atan2(δy, δx) - θ        (normalised to [-π, π])
```

**Jacobian H** (2×n, partial of measurement model w.r.t. full state):

For a landmark at index `j` in the state vector:
```
         robot pose cols       ...   landmark cols
H = | -δx/r   -δy/r    0  ...  δx/r    δy/r   |
    |  δy/q   -δx/q   -1  ... -δy/q    δx/q   |

where q = δx² + δy², r = sqrt(q)
```

All other columns are zero. H has shape `(2 × n)` where `n` is the full state size.

---

### 1.7 EKF SLAM — The Full Algorithm

**State vector** (grows dynamically as new landmarks are discovered):
```
μ = [x, y, θ, l_1x, l_1y, l_2x, l_2y, ..., l_Nx, l_Ny]^T
```

**Covariance matrix** Σ is `(3+2N) × (3+2N)` and has block structure:
```
Σ = | Σ_rr   Σ_rL |
    | Σ_Lr   Σ_LL |
```
Where:
- `Σ_rr` = robot-robot covariance (3×3)
- `Σ_rL` = robot-landmark cross-covariance (3×2N)
- `Σ_LL` = landmark-landmark covariance (2N×2N)

**The cross-covariances are critical** — they encode the fact that robot pose uncertainty and landmark uncertainty are correlated. Ignoring them would cause the filter to be inconsistent.

**Full EKF SLAM Algorithm (Probabilistic Robotics Table 10.2):**

```
Algorithm EKF_SLAM(μ_{t-1}, Σ_{t-1}, u_t, z_t):

--- PREDICTION STEP ---
F_x = [I_3 | 0_{3×2N}]           (selection matrix, picks robot pose block)

μ_bar_t = g(u_t, μ_{t-1})         (apply motion model to robot pose only)

G_t = I + F_x^T · G_x · F_x      (embed 3×3 Jacobian into full n×n matrix)

Σ_bar_t = G_t · Σ_{t-1} · G_t^T + F_x^T · R · F_x

--- CORRECTION STEP (for each observation z_t^i = (r^i, φ^i)) ---
j = correspondence(z_t^i)         (which landmark does this observation match?)

if landmark j never seen before:
    initialise landmark position in μ_bar_t
    expand Σ_bar_t with high uncertainty block

z_hat^i = h(μ_bar_t, j)           (predicted measurement)
H^i     = compute_H(μ_bar_t, j)   (2×n Jacobian)

S^i     = H^i · Σ_bar · H^i^T + Q
K^i     = Σ_bar · H^i^T · (S^i)^{-1}

innovation = z^i - z_hat^i
innovation[1] = normalise_angle(innovation[1])   (CRITICAL: wrap bearing)

μ_bar   = μ_bar + K^i · innovation
Σ_bar   = (I - K^i · H^i) · Σ_bar

return μ_t = μ_bar, Σ_t = Σ_bar
```

**Key implementation notes:**
- Angle normalisation `[-π, π]` must be applied to the bearing innovation, otherwise the filter diverges when crossing the ±π boundary
- When initialising a new landmark, use high initial covariance (e.g. `1e6 * I`) — do NOT use zero
- The cross-covariance blocks must be updated in Σ — do not just expand with zeros
- The `(I - K·H)` update can become numerically unstable; the **Joseph form** `(I-KH)Σ(I-KH)^T + KQK^T` is more stable but more expensive

---

### 1.8 New Landmark Initialisation

When a landmark is observed for the first time, its position is estimated from the current robot pose and the observation:
```
l_x = x + r·cos(θ + φ)
l_y = y + r·sin(θ + φ)
```

The covariance of the new landmark is propagated from both the robot pose uncertainty and the measurement noise using the inverse measurement Jacobians:

```
G_x = d(g_inv)/d(robot_pose)   (2×3)
G_z = d(g_inv)/d(measurement)  (2×2)

Σ_ll = G_x · Σ_rr · G_x^T + G_z · Q · G_z^T
```

This is the **correct** initialisation. Using `1e6 · I` is a simpler approximation.

---

### 1.9 Data Association

The algorithm above assumes we know which observation corresponds to which landmark. In practice this is the **data association problem** — one of the hardest parts of SLAM.

**Approaches (from simple to robust):**

1. **Known IDs** — ArUco markers have unique IDs, so association is trivial (used in your project)
2. **Nearest neighbour** — associate each observation to the closest landmark in state space
3. **Mahalanobis distance gate** — only accept associations within a statistical distance threshold (used in `realife.py`):
   ```
   d²_M = ν^T · S^{-1} · ν
   ```
   Where `ν = z - z_hat` is the innovation and `S = H·Σ·H^T + Q` is the innovation covariance. Reject if `d²_M > χ²_threshold` (e.g. 9.21 for 99% confidence, 2 DOF).

---

### 1.10 Performance Evaluation — Procrustes Alignment

EKF SLAM builds a map in the robot's own reference frame, which may differ from the global frame by an unknown rotation, translation, and scale. To compare estimated landmarks with ground truth:

1. **Procrustes analysis** finds the optimal rotation R, scale s, and translation t to align the estimated map to the true map
2. **Apply the same transform** to the entire robot trajectory
3. **Compute RMSE** between aligned estimates and ground truth

```
RMSE = sqrt( (1/N) · Σ ||p_i_aligned - p_i_true||² )
```

---

## PART 2 — CODE SKELETON

---

```python
"""
EKF SLAM — Main Simulator Skeleton
===================================
Based on: Probabilistic Robotics (Thrun, Burgard, Fox) — Table 10.2
          Stachniss KF/EKF Lecture (2020)

Project: SKT/SKP — EKF SLAM with ArUco landmarks
Course:  Autonomous Systems, IST Lisboa 2025/26

State vector:  mu  = [x, y, θ, l1x, l1y, l2x, l2y, ..., lNx, lNy]
Covariance:    Sigma = (3+2N) × (3+2N)
Control input: u = (prev_odom, curr_odom) — odometry motion model
Observations:  z = [(id, [r, φ]), ...] — range-bearing per landmark
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import time
import csv


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

class Config:
    """All tunable parameters in one place."""

    # --- Motion noise (R matrix) ---
    # Diagonal: variance in x, y, theta
    # Increase to trust odometry less
    MOTION_NOISE_X     = 0.05        # metres std dev
    MOTION_NOISE_Y     = 0.05        # metres std dev
    MOTION_NOISE_THETA = np.deg2rad(2.0)  # radians std dev

    # --- Measurement noise (Q matrix) ---
    # Diagonal: variance in range and bearing
    MEAS_NOISE_RANGE   = 0.05        # metres std dev
    MEAS_NOISE_BEARING = np.deg2rad(2.0)  # radians std dev

    # --- Sensor parameters ---
    MAX_RANGE   = 10.0               # max landmark detection range (m)
    MAX_BEARING = np.pi / 2          # half field of view (rad) — ±90°

    # --- Initial state ---
    INIT_ROBOT_POSE = np.array([2.0, 2.0, 0.0])  # [x, y, theta]

    # --- Landmark initialisation covariance ---
    INIT_LM_UNCERTAINTY = 1e6        # large initial uncertainty for new landmarks

    # --- Mahalanobis outlier rejection threshold ---
    # chi2 with 2 DOF: 95% = 5.99, 99% = 9.21
    MAHALANOBIS_THRESHOLD = 9.21

    # --- Visualisation ---
    ANIMATE      = True              # set False to skip animation, show only final plot
    PAUSE_TIME   = 0.05              # seconds between animation frames

    # --- Output ---
    SAVE_CSV     = True
    CSV_FILENAME = 'ekf_slam_results.csv'


# =============================================================================
# SECTION 2 — MAP AND ENVIRONMENT
# =============================================================================

def get_map(option: int):
    """
    Returns (map_grid, landmarks_true).

    map_grid:        25×25 numpy array, 1=obstacle, 0=free
    landmarks_true:  (N, 2) array of true landmark positions [x, y]

    Options:
        1  — 5 landmarks (full loop experiment)
        2  — 3 landmarks
        3  — 1 landmark  (degenerate case test)
        50 — 200 random landmarks (stress test)
    """
    map_grid = np.zeros((25, 25), dtype=int)
    # Outer walls
    map_grid[0, :]  = 1
    map_grid[-1, :] = 1
    map_grid[:, 0]  = 1
    map_grid[:, -1] = 1
    # Internal obstacle
    map_grid[8:20, 12] = 1

    if option == 1:
        landmarks = np.array([
            [19,  5],   # right wall
            [19, 15],   # right wall
            [12, 21],   # top
            [ 3, 15],   # left wall
            [ 3,  5],   # left wall
        ], dtype=float)

    elif option == 2:
        landmarks = np.array([
            [19,  5],
            [12, 21],
            [ 3,  5],
        ], dtype=float)

    elif option == 3:
        landmarks = np.array([[12, 21]], dtype=float)

    elif option == 50:
        # TODO: implement random landmark generation if needed
        raise NotImplementedError("Option 50 not implemented in skeleton")

    else:
        raise ValueError(f"Unknown map option: {option}")

    return map_grid, landmarks


def build_predefined_path() -> list:
    """
    Builds a hardcoded 2-lap rectangular trajectory.
    Returns list of [x, y, theta] waypoints.
    """
    path = []

    # --- LAP 1 ---
    for x in range(2, 17):        path.append([float(x),  2.0,       0.0])
    for y in range(3, 21):        path.append([16.0,       float(y),  np.pi/2])
    for x in range(15, 4, -1):    path.append([float(x),  20.0,      np.pi])
    for y in range(19, 2, -1):    path.append([5.0,        float(y), -np.pi/2])

    # --- LAP 2 ---
    for x in range(5, 17):        path.append([float(x),  2.0,       np.pi/2])
    for y in range(3, 21):        path.append([16.0,       float(y),  np.pi/2])
    for x in range(15, 4, -1):    path.append([float(x),  20.0,      np.pi])
    for y in range(19, 2, -1):    path.append([5.0,        float(y), -np.pi/2])

    return path


# =============================================================================
# SECTION 3 — MATH UTILITIES
# =============================================================================

def normalize_angle(angle: float) -> float:
    """Wraps angle to [-pi, pi]. CRITICAL: must be applied to bearing innovations."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def build_R_matrix() -> np.ndarray:
    """Motion noise covariance matrix (3×3)."""
    return np.diag([
        Config.MOTION_NOISE_X     ** 2,
        Config.MOTION_NOISE_Y     ** 2,
        Config.MOTION_NOISE_THETA ** 2,
    ])


def build_Q_matrix() -> np.ndarray:
    """Measurement noise covariance matrix (2×2)."""
    return np.diag([
        Config.MEAS_NOISE_RANGE   ** 2,
        Config.MEAS_NOISE_BEARING ** 2,
    ])


# =============================================================================
# SECTION 4 — MOTION MODEL (Odometry)
# =============================================================================

def motion_model(mu: np.ndarray, u: tuple) -> np.ndarray:
    """
    Odometry motion model — Probabilistic Robotics Ch. 5.

    Decomposes the motion between two odometry readings into:
        delta_rot1  — initial rotation to face the new position
        delta_trans — straight-line translation
        delta_rot2  — final rotation to reach the new heading

    Args:
        mu: current state vector (full: robot + landmarks)
        u:  tuple (prev_odom, curr_odom), each is [x, y, theta]

    Returns:
        mu_pred: predicted state vector (only robot pose changes)
    """
    x, y, theta = mu[0:3]
    x_bar,  y_bar,  theta_bar  = u[0]   # previous odometry reading
    x_bar_, y_bar_, theta_bar_ = u[1]   # current  odometry reading

    # TODO: compute delta_rot1, delta_trans, delta_rot2
    delta_rot1  = normalize_angle(
        math.atan2(y_bar_ - y_bar, x_bar_ - x_bar) - theta_bar
    )
    delta_trans = math.sqrt((x_bar_ - x_bar)**2 + (y_bar_ - y_bar)**2)
    delta_rot2  = normalize_angle(theta_bar_ - theta_bar - delta_rot1)

    # TODO: apply motion to robot pose
    x_pred     = x + delta_trans * math.cos(theta + delta_rot1)
    y_pred     = y + delta_trans * math.sin(theta + delta_rot1)
    theta_pred = normalize_angle(theta + delta_rot1 + delta_rot2)

    mu_pred = mu.copy()
    mu_pred[0:3] = [x_pred, y_pred, theta_pred]
    return mu_pred


def compute_G(mu: np.ndarray, u: tuple) -> np.ndarray:
    """
    Jacobian of the motion model w.r.t. the full state vector.

    The 3×3 block G_x affects only the robot pose.
    It is embedded into the full n×n identity matrix via F_x.

    Args:
        mu: current state vector
        u:  odometry control input tuple

    Returns:
        G: (n×n) Jacobian matrix
    """
    theta = mu[2]
    x_bar,  y_bar,  theta_bar = u[0]
    x_bar_, y_bar_, _         = u[1]

    delta_rot1  = normalize_angle(
        np.arctan2(y_bar_ - y_bar, x_bar_ - x_bar) - theta_bar
    )
    delta_trans = np.sqrt((x_bar_ - x_bar)**2 + (y_bar_ - y_bar)**2)

    # TODO: fill in G_x (3×3 Jacobian of motion w.r.t. robot pose)
    G_x = np.eye(3)
    G_x[0, 2] = -delta_trans * np.sin(theta + delta_rot1)
    G_x[1, 2] =  delta_trans * np.cos(theta + delta_rot1)

    # TODO: embed G_x into full state Jacobian using F_x selection matrix
    n = len(mu)
    G = np.eye(n)
    G[0:3, 0:3] = G_x
    return G


# =============================================================================
# SECTION 5 — MEASUREMENT MODEL (Range-Bearing)
# =============================================================================

def measurement_model(mu: np.ndarray, lm_index: int) -> np.ndarray:
    """
    Predicts the expected observation of a landmark given the current state.

    Args:
        mu:       current state vector
        lm_index: index in mu where the landmark's x coordinate lives

    Returns:
        z_hat: np.array([r, phi]) — predicted range and bearing
    """
    x, y, theta = mu[0:3]
    l_x = mu[lm_index]
    l_y = mu[lm_index + 1]

    # TODO: compute range and bearing
    dx  = l_x - x
    dy  = l_y - y
    r   = np.sqrt(dx**2 + dy**2)
    phi = normalize_angle(np.arctan2(dy, dx) - theta)

    return np.array([r, phi])


def compute_H(mu: np.ndarray, lm_index: int) -> np.ndarray:
    """
    Jacobian of the measurement model w.r.t. the full state vector.

    Non-zero columns only for: robot pose (0,1,2) and landmark (lm_index, lm_index+1).

    Args:
        mu:       current state vector
        lm_index: index in mu where the landmark's x coordinate lives

    Returns:
        H: (2×n) Jacobian matrix
    """
    x, y, theta = mu[0:3]
    l_x = mu[lm_index]
    l_y = mu[lm_index + 1]

    dx = l_x - x
    dy = l_y - y
    q  = dx**2 + dy**2
    r  = np.sqrt(q)

    n = len(mu)
    H = np.zeros((2, n))

    # TODO: derivatives w.r.t. robot pose (x, y, theta)
    H[0, 0] = -dx / r
    H[0, 1] = -dy / r
    H[0, 2] =  0.0
    H[1, 0] =  dy / q
    H[1, 1] = -dx / q
    H[1, 2] = -1.0

    # TODO: derivatives w.r.t. landmark (l_x, l_y)
    H[0, lm_index]     =  dx / r
    H[0, lm_index + 1] =  dy / r
    H[1, lm_index]     = -dy / q
    H[1, lm_index + 1] =  dx / q

    return H


# =============================================================================
# SECTION 6 — EKF SLAM CORE
# =============================================================================

def ekf_predict(mu: np.ndarray, Sigma: np.ndarray,
                u: tuple, R: np.ndarray) -> tuple:
    """
    EKF SLAM prediction step.

    Propagates robot pose forward using odometry.
    Landmark estimates are unchanged.
    Covariance grows due to motion noise.

    Args:
        mu:    current state mean
        Sigma: current state covariance
        u:     odometry control (prev_odom, curr_odom)
        R:     3×3 motion noise covariance

    Returns:
        (mu_bar, Sigma_bar)
    """
    # TODO: predict mean
    mu_bar = motion_model(mu, u)

    # TODO: compute full Jacobian G
    G = compute_G(mu, u)

    # TODO: F_x selection matrix — maps 3×3 R into full state space
    n  = len(mu)
    Fx = np.hstack([np.eye(3), np.zeros((3, n - 3))])

    # TODO: propagate covariance
    Sigma_bar = G @ Sigma @ G.T + Fx.T @ R @ Fx

    return mu_bar, Sigma_bar


def initialise_landmark(mu: np.ndarray, Sigma: np.ndarray,
                         z: np.ndarray, Q: np.ndarray) -> tuple:
    """
    Adds a new landmark to the state vector and covariance matrix.

    Called the first time a landmark is observed.

    Args:
        mu:    current state mean (before expansion)
        Sigma: current covariance (before expansion)
        z:     observation [r, bearing] for this new landmark
        Q:     2×2 measurement noise covariance

    Returns:
        (mu_expanded, Sigma_expanded, lm_index)
        lm_index: the index in the new mu where this landmark's x lives
    """
    r, bearing = z
    x, y, theta = mu[0:3]

    # TODO: compute initial landmark position from current pose + observation
    lm_x = x + r * np.cos(theta + bearing)
    lm_y = y + r * np.sin(theta + bearing)

    # TODO: expand mu
    lm_index = len(mu)
    mu_new   = np.concatenate([mu, [lm_x, lm_y]])

    # TODO: expand Sigma
    # Simple approach: high uncertainty block on diagonal
    old_size    = Sigma.shape[0]
    new_size    = old_size + 2
    Sigma_new   = np.zeros((new_size, new_size))
    Sigma_new[:old_size, :old_size] = Sigma
    Sigma_new[old_size:, old_size:] = np.eye(2) * Config.INIT_LM_UNCERTAINTY

    # TODO (optional, more correct): propagate uncertainty from robot pose
    # G_x = d(inverse_meas)/d(robot_pose)  shape (2×3)
    # G_z = d(inverse_meas)/d(measurement) shape (2×2)
    # Sigma_ll = G_x @ Sigma[0:3,0:3] @ G_x.T + G_z @ Q @ G_z.T
    # ... and fill cross-covariance blocks too

    return mu_new, Sigma_new, lm_index


def ekf_update(mu: np.ndarray, Sigma: np.ndarray,
               observations: list, landmark_map: dict,
               Q: np.ndarray) -> tuple:
    """
    EKF SLAM correction step.

    Processes each observation, initialises new landmarks,
    and updates the state estimate with Kalman correction.

    Args:
        mu:           predicted state mean
        Sigma:        predicted covariance
        observations: list of (landmark_id, np.array([r, bearing]))
        landmark_map: dict mapping landmark_id → index in mu
        Q:            2×2 measurement noise covariance

    Returns:
        (mu_updated, Sigma_updated, landmark_map)
    """
    for landmark_id, z in observations:

        # --- Initialise new landmark ---
        if landmark_id not in landmark_map:
            mu, Sigma, lm_index = initialise_landmark(mu, Sigma, z, Q)
            landmark_map[landmark_id] = lm_index

        lm_index = landmark_map[landmark_id]

        # --- Predicted measurement ---
        z_hat = measurement_model(mu, lm_index)

        # --- Jacobian ---
        H = compute_H(mu, lm_index)

        # --- Innovation covariance ---
        S = H @ Sigma @ H.T + Q

        # --- Optional: Mahalanobis outlier rejection ---
        nu    = z - z_hat
        nu[1] = normalize_angle(nu[1])    # CRITICAL: wrap bearing innovation

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        mahal_sq = nu.T @ S_inv @ nu
        if mahal_sq > Config.MAHALANOBIS_THRESHOLD:
            # Observation is an outlier — skip this update
            continue

        # --- Kalman gain ---
        K = Sigma @ H.T @ S_inv

        # --- State update ---
        mu    = mu + K @ nu
        mu[2] = normalize_angle(mu[2])    # normalise robot heading

        # --- Covariance update ---
        n     = len(mu)
        Sigma = (np.eye(n) - K @ H) @ Sigma

    return mu, Sigma, landmark_map


# =============================================================================
# SECTION 7 — SENSOR SIMULATION
# =============================================================================

def simulate_odometry(prev_pose: list, curr_pose: list,
                       R: np.ndarray) -> tuple:
    """
    Simulates noisy wheel encoder readings.

    Takes the true poses and adds Gaussian noise to the deltas,
    producing a realistic odometry measurement.

    Args:
        prev_pose: true pose at t-1 [x, y, theta]
        curr_pose: true pose at t   [x, y, theta]
        R:         3×3 motion noise covariance

    Returns:
        u = (prev_odom, curr_odom) — noisy odometry control input
    """
    dx     = curr_pose[0] - prev_pose[0]
    dy     = curr_pose[1] - prev_pose[1]
    dtheta = normalize_angle(curr_pose[2] - prev_pose[2])

    # TODO: add Gaussian noise to each component
    dx_noisy     = dx     + np.random.normal(0, np.sqrt(R[0, 0]))
    dy_noisy     = dy     + np.random.normal(0, np.sqrt(R[1, 1]))
    dtheta_noisy = dtheta + np.random.normal(0, np.sqrt(R[2, 2]))

    odom_curr = [
        prev_pose[0] + dx_noisy,
        prev_pose[1] + dy_noisy,
        normalize_angle(prev_pose[2] + dtheta_noisy),
    ]

    return (prev_pose, odom_curr)


def simulate_observations(true_pose: list, landmarks_true: np.ndarray,
                           Q: np.ndarray) -> list:
    """
    Simulates noisy range-bearing observations.

    For each landmark within range and field of view, computes
    the true range and bearing and adds Gaussian noise.

    Args:
        true_pose:      current true robot pose [x, y, theta]
        landmarks_true: (N, 2) array of true landmark positions
        Q:              2×2 measurement noise covariance

    Returns:
        observations: list of (landmark_id, np.array([r_noisy, bearing_noisy]))
    """
    observations = []

    for i, lm in enumerate(landmarks_true):
        dx      = lm[0] - true_pose[0]
        dy      = lm[1] - true_pose[1]
        r       = np.hypot(dx, dy)
        bearing = normalize_angle(np.arctan2(dy, dx) - true_pose[2])

        # Check sensor constraints
        if r > Config.MAX_RANGE:
            continue
        if abs(bearing) > Config.MAX_BEARING:
            continue

        # TODO: add noise to measurement
        r_noisy       = r       + np.random.normal(0, np.sqrt(Q[0, 0]))
        bearing_noisy = bearing + np.random.normal(0, np.sqrt(Q[1, 1]))

        observations.append((i, np.array([r_noisy, bearing_noisy])))

    return observations


# =============================================================================
# SECTION 8 — VISUALISATION
# =============================================================================

def plot_covariance_ellipse(ax, mean: np.ndarray, cov: np.ndarray,
                             n_std: float = 1.0,
                             edgecolor: str = 'blue') -> None:
    """Draws a 2D covariance ellipse on ax."""
    if cov.shape != (2, 2):
        return

    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order     = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    # Clamp negative eigenvalues (numerical noise)
    eigenvals = np.maximum(eigenvals, 0)

    angle  = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width  = 2 * n_std * np.sqrt(eigenvals[0])
    height = 2 * n_std * np.sqrt(eigenvals[1])

    ellipse = patches.Ellipse(
        xy=mean, width=width, height=height,
        angle=angle, facecolor='none',
        edgecolor=edgecolor, linestyle='--', linewidth=1.2
    )
    ax.add_patch(ellipse)


def setup_figure():
    """Initialises the matplotlib figure with two subplots."""
    fig, (ax_map, ax_info) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('EKF SLAM Simulator', fontsize=14)

    ax_map.set_title('Map')
    ax_map.set_xlim(0, 25)
    ax_map.set_ylim(0, 25)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)
    ax_map.set_xlabel('x (m)')
    ax_map.set_ylabel('y (m)')

    ax_info.set_title('State & Covariance')
    ax_info.axis('off')

    plt.ion()
    return fig, ax_map, ax_info


def update_plot(ax_map, ax_info, mu, Sigma, landmark_map,
                landmarks_true, true_path, robot_path,
                step: int) -> None:
    """
    Redraws the animation frame.

    Shows:
    - True landmark positions (black stars)
    - Estimated landmark positions + uncertainty ellipses (red/orange)
    - True robot path (green dashed)
    - Estimated robot path (blue)
    - Current robot pose + heading arrow
    - Robot pose uncertainty ellipse
    """
    ax_map.clear()
    ax_map.set_title(f'EKF SLAM — Step {step}')
    ax_map.set_xlim(0, 25)
    ax_map.set_ylim(0, 25)
    ax_map.set_aspect('equal')
    ax_map.grid(True, alpha=0.3)

    # True landmarks
    ax_map.plot(landmarks_true[:, 0], landmarks_true[:, 1],
                'k*', markersize=12, label='True Landmarks', zorder=5)

    # True path
    if len(true_path) > 1:
        tp = np.array(true_path)
        ax_map.plot(tp[:, 0], tp[:, 1], 'g--', linewidth=1, label='True Path')

    # Estimated path
    if len(robot_path) > 1:
        rp = np.array(robot_path)
        ax_map.plot(rp[:, 0], rp[:, 1], 'b-', linewidth=1, label='EKF Path')

    # Estimated landmarks + ellipses
    for lm_id, lm_idx in landmark_map.items():
        if lm_idx + 1 >= len(mu):
            continue
        mean = mu[lm_idx:lm_idx + 2]
        cov  = Sigma[lm_idx:lm_idx + 2, lm_idx:lm_idx + 2]
        ax_map.plot(mean[0], mean[1], 'rx', markersize=8, markeredgewidth=2)
        ax_map.text(mean[0] + 0.3, mean[1] + 0.3, f'{lm_id}',
                    fontsize=7, color='red')
        plot_covariance_ellipse(ax_map, mean, cov, n_std=1, edgecolor='orange')

    # Robot pose
    x, y, theta = mu[0:3]
    ax_map.plot(x, y, 'bo', markersize=9, zorder=6, label='EKF Robot')
    ax_map.arrow(x, y, 0.5 * np.cos(theta), 0.5 * np.sin(theta),
                 head_width=0.3, head_length=0.2, fc='b', ec='b')
    robot_cov = Sigma[0:2, 0:2]
    plot_covariance_ellipse(ax_map, np.array([x, y]), robot_cov,
                             n_std=1, edgecolor='blue')

    ax_map.legend(loc='upper right', fontsize=7)

    # Info panel — state vector summary
    ax_info.clear()
    ax_info.axis('off')
    lines = [f'Step: {step}',
             f'Robot:  x={x:.2f}  y={y:.2f}  θ={np.degrees(theta):.1f}°',
             f'Landmarks in map: {len(landmark_map)}',
             '',
             'Estimated landmarks:']
    for lm_id, lm_idx in landmark_map.items():
        if lm_idx + 1 < len(mu):
            lines.append(f'  LM {lm_id}: ({mu[lm_idx]:.2f}, {mu[lm_idx+1]:.2f})')
    ax_info.text(0.05, 0.95, '\n'.join(lines), fontsize=9,
                 family='monospace', va='top', transform=ax_info.transAxes,
                 bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.pause(Config.PAUSE_TIME)


# =============================================================================
# SECTION 9 — EVALUATION
# =============================================================================

def procrustes_align(X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Aligns estimated landmark set Y to true landmark set X.

    Finds optimal rotation R, scale s, translation t such that:
        Y_aligned ≈ s * Y @ R^T + t  ≈  X

    Returns:
        Y_aligned, scale, R, t
    """
    from procrustes import procrustes_custom
    return procrustes_custom(X, Y)


def compute_rmse(true_pts: np.ndarray, estimated_pts: np.ndarray) -> float:
    """Computes RMSE between two (N, 2) point sets."""
    errors = np.linalg.norm(true_pts - estimated_pts, axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def evaluate(mu: np.ndarray, landmark_map: dict,
             landmarks_true: np.ndarray,
             robot_path: list, true_path: list) -> dict:
    """
    Runs Procrustes alignment and computes all error metrics.

    Returns dict with:
        rmse_landmarks, mean_error, scale, individual_errors,
        aligned_landmarks, aligned_path
    """
    results = {}

    # Extract estimated landmark positions in order of true landmark IDs
    estimated_pts = []
    true_pts      = []

    for lm_id in range(len(landmarks_true)):
        if lm_id in landmark_map:
            lm_idx = landmark_map[lm_id]
            if lm_idx + 1 < len(mu):
                estimated_pts.append(mu[lm_idx:lm_idx + 2])
                true_pts.append(landmarks_true[lm_id])

    if len(estimated_pts) < 2:
        print("WARNING: fewer than 2 landmarks seen — cannot align.")
        results['rmse_landmarks'] = float('inf')
        return results

    estimated_pts = np.array(estimated_pts)
    true_pts      = np.array(true_pts)

    # Align estimated to true
    aligned, scale, R, t = procrustes_align(true_pts, estimated_pts)

    # Individual errors
    individual_errors = np.linalg.norm(aligned - true_pts, axis=1)
    rmse = compute_rmse(true_pts, aligned)

    results['rmse_landmarks']    = rmse
    results['mean_error']        = float(np.mean(individual_errors))
    results['individual_errors'] = individual_errors
    results['scale']             = scale
    results['R']                 = R
    results['t']                 = t
    results['aligned_landmarks'] = aligned
    results['true_pts']          = true_pts

    # Align robot trajectory too
    if len(robot_path) > 0:
        rp = np.array(robot_path)[:, :2]
        results['aligned_path'] = scale * rp @ R + t
        tp = np.array(true_path)[:, :2]
        results['true_path_np'] = tp

    return results


def plot_final(results: dict, landmarks_true: np.ndarray) -> None:
    """Plots the final aligned map with true vs estimated comparison."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title('Final Map — EKF Estimates Aligned to Ground Truth', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # True landmarks
    ax.plot(landmarks_true[:, 0], landmarks_true[:, 1],
            'k*', markersize=14, label='True Landmarks', zorder=5)

    # Aligned estimated landmarks
    if 'aligned_landmarks' in results:
        ax.scatter(results['aligned_landmarks'][:, 0],
                   results['aligned_landmarks'][:, 1],
                   c='orange', s=80, zorder=4, label='EKF Landmarks (aligned)')

    # Paths
    if 'true_path_np' in results:
        tp = results['true_path_np']
        ax.plot(tp[:, 0], tp[:, 1], 'g--', linewidth=1.5, label='True Path')

    if 'aligned_path' in results:
        ap = results['aligned_path']
        ax.plot(ap[:, 0], ap[:, 1], 'b-', linewidth=1.5, label='EKF Path (aligned)')

    # Print errors
    if 'rmse_landmarks' in results:
        info = (f"RMSE: {results['rmse_landmarks']:.4f} m\n"
                f"Mean error: {results['mean_error']:.4f} m\n"
                f"Scale: {results['scale']:.4f}")
        ax.text(0.02, 0.98, info, transform=ax.transAxes,
                fontsize=10, va='top', family='monospace',
                bbox=dict(facecolor='white', alpha=0.8))

    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


# =============================================================================
# SECTION 10 — CSV LOGGING
# =============================================================================

def init_csv(filename: str):
    """Opens CSV file and writes header. Returns (file, writer)."""
    try:
        f = open(filename, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['step', 'timestamp_s', 'num_landmarks',
                         'ekf_duration_ms', 'robot_x', 'robot_y', 'robot_theta'])
        return f, writer
    except IOError as e:
        print(f"WARNING: could not open CSV: {e}")
        return None, None


def log_step(writer, step: int, mu: np.ndarray,
             num_landmarks: int, duration_ms: float) -> None:
    """Logs one timestep to CSV."""
    if writer is None:
        return
    writer.writerow([step, time.time(), num_landmarks,
                     duration_ms, mu[0], mu[1], np.degrees(mu[2])])


# =============================================================================
# SECTION 11 — MAIN LOOP
# =============================================================================

def main():
    print("EKF SLAM Simulator")
    print("==================")
    print("Map options:  1 (5 landmarks)  2 (3 landmarks)  3 (1 landmark)")
    map_choice = int(input("Choose map: "))

    # --- Setup ---
    map_grid, landmarks_true = get_map(map_choice)
    predefined_path = build_predefined_path()

    R = build_R_matrix()
    Q = build_Q_matrix()

    # --- Initialise EKF state ---
    mu    = np.zeros(3)
    mu[:] = predefined_path[0]
    Sigma = np.zeros((3, 3))          # start with zero uncertainty (known start)
    landmark_map = {}                  # landmark_id → index in mu

    # --- Logging ---
    true_path  = []
    robot_path = []

    csv_file, csv_writer = (init_csv(Config.CSV_FILENAME)
                            if Config.SAVE_CSV else (None, None))

    # --- Visualisation ---
    if Config.ANIMATE:
        fig, ax_map, ax_info = setup_figure()

    print(f"\nRunning {len(predefined_path)-1} steps...")

    # --- Main simulation loop ---
    for t in range(len(predefined_path) - 1):

        true_pose = predefined_path[t + 1]
        true_path.append(true_pose)

        # 1. Simulate noisy odometry
        u = simulate_odometry(predefined_path[t], true_pose, R)

        # 2. Simulate noisy observations
        observations = simulate_observations(true_pose, landmarks_true, Q)

        # 3. EKF SLAM
        step_start = time.time()

        mu, Sigma = ekf_predict(mu, Sigma, u, R)
        mu, Sigma, landmark_map = ekf_update(mu, Sigma, observations,
                                              landmark_map, Q)

        duration_ms = (time.time() - step_start) * 1000

        robot_path.append(mu[0:3].copy())

        # 4. Log
        log_step(csv_writer, t, mu, len(landmark_map), duration_ms)

        # 5. Animate
        if Config.ANIMATE and t % 3 == 0:   # update every 3 steps for speed
            update_plot(ax_map, ax_info, mu, Sigma, landmark_map,
                        landmarks_true, true_path, robot_path, t)

    print("\nSimulation complete.")
    print(f"Landmarks discovered: {len(landmark_map)} / {len(landmarks_true)}")

    # --- Close CSV ---
    if csv_file:
        csv_file.close()
        print(f"Results saved to {Config.CSV_FILENAME}")

    # --- Evaluate and plot final result ---
    results = evaluate(mu, landmark_map, landmarks_true, robot_path, true_path)

    if 'rmse_landmarks' in results:
        print(f"\nRMSE (landmarks): {results['rmse_landmarks']:.4f} m")
        print(f"Mean error:        {results['mean_error']:.4f} m")
        print(f"Scale factor:      {results['scale']:.4f}")
        print("\nIndividual landmark errors:")
        for i, err in enumerate(results['individual_errors']):
            print(f"  LM {i}: {err:.4f} m")

    plot_final(results, landmarks_true)


if __name__ == '__main__':
    main()
```
