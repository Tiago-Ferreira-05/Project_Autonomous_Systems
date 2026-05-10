"""
EKF SLAM Simulation Runner.
============================

Entry point for the Week 3 simulation.  Ties together all four packages:

    simulation/   — noisy odometry and range-bearing observations
    ekf_core/     — EKF predict / update / landmark initialisation
    visualisation/— live animation and final plot
    evaluation/   — Procrustes alignment and RMSE metrics

Usage
-----
From the my_ekf_slam/ directory:

    python run_simulation.py

The user is prompted to choose a map option (1 / 2 / 3) at startup.
All other parameters (noise, FOV, animation, output paths) are read
from config.py — no magic numbers in this file.

Simulation loop (one iteration = one waypoint step)
----------------------------------------------------
    1. Simulate noisy odometry  u = (prev_odom, curr_odom)
    2. EKF prediction step      mu_bar, Sigma_bar = ekf_predict(...)
    3. Simulate observations    z_all = simulate_observations(...)
    4. EKF correction step      mu, Sigma = ekf_update(...)
    5. Record true and estimated paths
    6. Animate every ANIMATION_STEP steps  (if ANIMATE = True)

After the loop
--------------
    7. Evaluate  — Procrustes alignment + RMSE report + optional CSV
    8. Final plot — static summary figure (with alignment subplot)

ROS 2 integration (Week 4+)
---------------------------
The ekf_core/ package is intentionally decoupled from the simulation
layer.  When switching to real robot data, replace steps 1 and 3 with
ROS 2 topic callbacks that produce the same data formats:

    u      : tuple (prev_odom, curr_odom)  each a length-3 array [x,y,theta]
    z_all  : list of (landmark_id, np.array([r, phi]))

No changes to ekf_core/, evaluation/, or visualisation/ are required.
"""

import sys
import numpy as np

# ---------------------------------------------------------------------------
# Package imports — all relative to my_ekf_slam/
# ---------------------------------------------------------------------------
import config as cfg

from simulation   import get_map, build_predefined_path
from simulation   import simulate_odometry, simulate_observations

from ekf_core     import ekf_predict, ekf_update

from visualization import update_plot, plot_final
from evaluation    import evaluate


# ===========================================================================
# HELPERS
# ===========================================================================

def _prompt_map_choice() -> int:
    """
    Interactively asks the user to select a map configuration.

    Returns:
        option : int in {1, 2, 3}
    """
    print("\n" + "=" * 50)
    print("  EKF SLAM Simulator — Map Selection")
    print("=" * 50)
    print("  1 — 5 landmarks  (standard experiment)")
    print("  2 — 3 landmarks  (reduced experiment)")
    print("  3 — 1 landmark   (sanity check)")
    print("=" * 50)

    while True:
        try:
            choice = int(input("  Select map [1 / 2 / 3]: ").strip())
            if choice in {1, 2, 3}:
                return choice
            print("  Please enter 1, 2, or 3.")
        except ValueError:
            print("  Invalid input — please enter an integer.")


def _initialise_ekf(path: list) -> tuple:
    """
    Initialises the EKF state vector and covariance matrix.

    The robot pose is seeded from the first waypoint of the predefined
    path.  Initial robot-pose uncertainty is set to near-zero (we assume
    we know where the robot starts).  The landmark portion of the state
    grows dynamically as new landmarks are observed.

    Args:
        path : List of [x, y, theta] waypoints from build_predefined_path().

    Returns:
        mu           : Initial state vector  [x, y, theta]  shape (3,)
        Sigma        : Initial covariance    shape (3, 3)
        landmark_map : Empty dict — grows during the simulation loop.
    """
    x0, y0, theta0 = path[0]

    mu    = np.array([x0, y0, theta0], dtype=float)

    # Small but non-zero initial robot uncertainty.
    # Heading uncertainty is set to zero: we trust our starting orientation.
    Sigma = np.diag([0.01, 0.01, 0.0])

    landmark_map = {}   # {landmark_id : start_index_in_mu}

    return mu, Sigma, landmark_map


# ===========================================================================
# MAIN SIMULATION LOOP
# ===========================================================================

def run() -> None:
    """
    Runs the full EKF SLAM simulation and evaluation pipeline.

    Steps:
        1. User selects a map.
        2. Environment (grid + landmarks) and path are loaded.
        3. EKF state is initialised.
        4. Main loop: predict → observe → update → (animate).
        5. Post-loop: evaluate and plot final results.
    """

    # ------------------------------------------------------------------
    # 1. Map selection
    # ------------------------------------------------------------------
    map_option                  = _prompt_map_choice()
    map_grid, landmarks_true    = get_map(map_option)
    path                        = build_predefined_path(num_laps=cfg.NUM_LAPS)

    n_steps = len(path) - 1   # number of motion steps

    print(f"\n  Map option   : {map_option}  ({len(landmarks_true)} landmarks)")
    print(f"  Path length  : {n_steps} steps  ({cfg.NUM_LAPS} lap(s))")
    print(f"  Animate      : {cfg.ANIMATE}  (every {cfg.ANIMATION_STEP} step(s))")
    print()

    # ------------------------------------------------------------------
    # 2. EKF initialisation
    # ------------------------------------------------------------------
    mu, Sigma, landmark_map = _initialise_ekf(path)

    # Random number generator — seeded for reproducibility.
    # Change the seed or pass None for a different run each time.
    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------
    # 3. Path recording buffers
    # ------------------------------------------------------------------
    true_path  = []   # list of [x, y] — ground-truth robot positions
    robot_path = []   # list of [x, y] — EKF-estimated robot positions

    # ------------------------------------------------------------------
    # 4. Main simulation loop
    # ------------------------------------------------------------------
    print("  Running simulation …\n")

    for t in range(n_steps):

        prev_pose = path[t]       # true pose at time t
        curr_pose = path[t + 1]   # true pose at time t+1

        # --------------------------------------------------------------
        # 4a. Simulate noisy odometry
        #     u = (prev_odom, curr_odom) — same format as the ROS /odom
        #     callback will produce in Week 4.
        # --------------------------------------------------------------
        u = simulate_odometry(
            prev_true_pose = prev_pose,
            curr_true_pose = curr_pose,
            R              = cfg.R,
            rng            = rng,
        )

        # --------------------------------------------------------------
        # 4b. EKF prediction step
        #     Advances the state estimate using the motion model.
        #     Only the robot-pose block of mu and Sigma is affected.
        # --------------------------------------------------------------
        mu, Sigma = ekf_predict(
            mu    = mu,
            Sigma = Sigma,
            u     = u,
            R     = cfg.R,
        )

        # --------------------------------------------------------------
        # 4c. Simulate range-bearing observations
        #     Uses the *true* robot pose (curr_pose) so that sensor noise
        #     is correctly centred on the ground truth, not the EKF estimate.
        #     In Week 4 these come from the ArUco detection pipeline.
        # --------------------------------------------------------------
        observations = simulate_observations(
            true_pose      = curr_pose,
            landmarks_true = landmarks_true,
            Q              = cfg.Q,
            max_range      = cfg.MAX_RANGE,
            max_bearing    = cfg.MAX_BEARING,
            rng            = rng,
        )

        # --------------------------------------------------------------
        # 4d. EKF correction step
        #     Updates mu and Sigma for every observation.
        #     New landmarks are initialised automatically inside ekf_update.
        # --------------------------------------------------------------
        mu, Sigma, landmark_map = ekf_update(
            mu_bar       = mu,
            Sigma_bar    = Sigma,
            observations = observations,
            landmark_map = landmark_map,
            Q            = cfg.Q,
        )

        # --------------------------------------------------------------
        # 4e. Record paths
        # --------------------------------------------------------------
        true_path.append(curr_pose[:2])     # [x, y] ground truth
        robot_path.append(mu[:2].copy())    # [x, y] EKF estimate

        # --------------------------------------------------------------
        # 4f. Live animation (optional, every ANIMATION_STEP steps)
        # --------------------------------------------------------------
        if cfg.ANIMATE and (t % cfg.ANIMATION_STEP == 0):
            update_plot(
                true_path      = true_path,
                robot_path     = robot_path,
                landmarks_true = landmarks_true,
                mu             = mu,
                Sigma          = Sigma,
                landmark_map   = landmark_map,
                cfg            = cfg,
                step           = t,
            )

        # --------------------------------------------------------------
        # 4g. Progress logging (every 50 steps to keep stdout clean)
        # --------------------------------------------------------------
        if t % 50 == 0 or t == n_steps - 1:
            n_lm = len(landmark_map)
            print(
                f"  step {t:>4d}/{n_steps}  |  "
                f"robot ({mu[0]:6.2f}, {mu[1]:6.2f})  |  "
                f"landmarks in map: {n_lm}"
            )

    # ------------------------------------------------------------------
    # 5. Post-loop evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("  Simulation complete — running evaluation …")
    print("=" * 50)

    eval_result = evaluate(
        mu             = mu,
        landmark_map   = landmark_map,
        landmarks_true = landmarks_true,
        cfg            = cfg,
    )

    # ------------------------------------------------------------------
    # 6. Final plot
    #    Pass aligned_landmarks from the evaluation result so plot_final()
    #    can render the Procrustes alignment subplot automatically.
    # ------------------------------------------------------------------
    aligned_landmarks = (
        eval_result['aligned_landmarks'] if eval_result is not None else None
    )

    plot_final(
        true_path          = true_path,
        robot_path         = robot_path,
        landmarks_true     = landmarks_true,
        mu                 = mu,
        Sigma              = Sigma,
        landmark_map       = landmark_map,
        cfg                = cfg,
        aligned_landmarks  = aligned_landmarks,
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Simulation interrupted by user.")
        sys.exit(0)