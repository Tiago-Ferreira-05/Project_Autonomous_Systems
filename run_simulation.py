"""
EKF SLAM Simulation Runner.
============================

Entry point for the EKF SLAM project.  Ties together all four packages:

    simulation/    — noisy odometry and range-bearing observations
    ekf_core/      — EKF predict / update / landmark initialisation
    visualization/ — live animation and final plot
    evaluation/    — Procrustes alignment and RMSE metrics

Usage
-----
From the project root directory:

    python run_simulation.py

The user is prompted to select:
    1. A map configuration (number and layout of landmarks).
    2. A control mode — automated path or interactive keyboard.

Control modes
-------------
    path     — Robot follows a hardcoded two-lap rectangular trajectory.
               Fully automated and reproducible (seeded RNG).
               Best for benchmarking and noise-parameter tuning.

    keyboard — Robot is driven interactively via the matplotlib figure window.
               W / S  : translate forward / backward  (KEYBOARD_STEP_M metres)
               A / D  : rotate left / right            (KEYBOARD_ROT_DEG °)
               Q      : finish session and run evaluation

               The robot's true pose is tracked explicitly; noisy odometry
               and sensor observations are synthesised from it at each step,
               so the EKF experiences the same uncertainty model as in path
               mode.

Simulation loop (one step = one motion command)
-----------------------------------------------
    1. Compute noisy odometry u = (prev_odom, curr_odom)
    2. EKF prediction step     mu_bar, Sigma_bar = ekf_predict(…)
    3. Synthesise observations z_all = simulate_observations(…)
    4. EKF correction step     mu, Sigma = ekf_update(…)
    5. Record paths and animate

Post-loop
---------
    6. Procrustes alignment + RMSE evaluation report
    7. Static final plot (with optional alignment subplot)

ROS 2 integration (Week 4+)
---------------------------
The ekf_core/ package is intentionally decoupled from the simulation layer.
When switching to real robot data, replace steps 1 and 3 with ROS 2 topic
callbacks that produce the same data formats:

    u     : tuple (prev_odom, curr_odom)  — each a length-3 array [x, y, θ]
    z_all : list of (landmark_id, np.array([r, φ]))

No changes to ekf_core/, evaluation/, or visualization/ are required.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Package imports — all relative to the project root
# ---------------------------------------------------------------------------
import config as cfg

from simulation    import get_map, build_predefined_path
from simulation    import simulate_odometry, simulate_observations

from ekf_core      import ekf_predict, ekf_update

from visualization import update_plot, plot_final
from evaluation    import evaluate


# ===========================================================================
# ANGLE UTILITY
# ===========================================================================

def _normalise_angle(angle: float) -> float:
    """
    Wraps *angle* to the interval [-π, π].

    Defined locally so this module has no hidden import dependency on any
    ekf_core submodule.  Consistent with the convention used throughout the
    codebase (meas_model.normalize_angle, motion_model._normalize_angle).
    """
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


# ===========================================================================
# STARTUP PROMPTS
# ===========================================================================

def _prompt_map_choice() -> int:
    """
    Interactively asks the user to select a map configuration.

    Presents a numbered menu and loops until a valid integer is entered.

    Returns:
        option : int — one of {1, 2, 3, 4}
    """
    print("\n" + "=" * 52)
    print("  EKF SLAM Simulator — Map Selection")
    print("=" * 52)
    print("  1 — 5 landmarks  (standard experiment)")
    print("  2 — 3 landmarks  (reduced experiment)")
    print("  3 — 1 landmark   (sanity check / degenerate case)")
    print("  4 — N landmarks  (random positions, custom count)")
    print("=" * 52)

    while True:
        try:
            choice = int(input("  Select map [1 / 2 / 3 / 4]: ").strip())
            if choice in {1, 2, 3, 4}:
                return choice
            print("  Please enter 1, 2, 3, or 4.")
        except ValueError:
            print("  Invalid input — please enter an integer.")


def _prompt_control_mode() -> str:
    """
    Interactively asks the user to choose a control mode.

    Returns:
        mode : str — either ``'path'`` or ``'keyboard'``

    Modes:
        path     — Automated rectangular trajectory (reproducible, no input
                   required during the simulation).
        keyboard — Interactive W/A/S/D control via the matplotlib window.
    """
    print("\n" + "=" * 52)
    print("  EKF SLAM Simulator — Control Mode")
    print("=" * 52)
    print("  1 — path      Automated rectangular trajectory")
    print("  2 — keyboard  Interactive W / A / S / D control")
    print("=" * 52)

    mode_map = {'1': 'path', '2': 'keyboard', 'path': 'path', 'keyboard': 'keyboard'}

    while True:
        raw = input("  Select mode [1 / 2]: ").strip().lower()
        if raw in mode_map:
            return mode_map[raw]
        print("  Please enter 1 (path) or 2 (keyboard).")


# ===========================================================================
# EKF INITIALISATION
# ===========================================================================

def _initialise_ekf(init_pose: np.ndarray) -> tuple:
    """
    Initialises the EKF state vector and covariance matrix from a given pose.

    Robot-pose uncertainty is seeded to near-zero: we assume we know where
    the robot starts.  The landmark portion of the state grows dynamically
    as new landmarks are observed during the simulation loop.

    Args:
        init_pose : np.ndarray shape (3,) — [x, y, θ] starting pose.

    Returns:
        mu           : Initial state vector  [x, y, θ]          shape (3,)
        Sigma        : Initial covariance    diag(0.01, 0.01, 0) shape (3, 3)
        landmark_map : Empty dict — grows during the simulation loop.
                       Format: {landmark_id : start_index_in_mu}
    """
    mu    = np.array(init_pose, dtype=float)

    # Small but non-zero position uncertainty; zero heading uncertainty
    # (we trust our initial orientation from the odometry source).
    Sigma = np.diag([0.01, 0.01, 0.0])

    landmark_map: dict = {}   # {lm_id : start_index_in_mu}

    return mu, Sigma, landmark_map


# ===========================================================================
# PATH MODE — AUTOMATED TRAJECTORY
# ===========================================================================

def _run_path_mode(
    mu:             np.ndarray,
    Sigma:          np.ndarray,
    landmark_map:   dict,
    landmarks_true: np.ndarray,
    rng:            np.random.Generator,
) -> tuple:
    """
    Runs the automated path-following simulation loop.

    The robot steps through every waypoint of the hardcoded rectangular
    trajectory (``build_predefined_path``).  Noisy odometry and range-bearing
    observations are synthesised at each step; the EKF is updated accordingly.

    Animation is shown every ``cfg.ANIMATION_STEP`` steps when
    ``cfg.ANIMATE`` is True.  Progress is logged to stdout every 50 steps.

    Args:
        mu             : Initial EKF state vector         shape (3,)
        Sigma          : Initial EKF covariance            shape (3, 3)
        landmark_map   : Empty landmark map dict.
        landmarks_true : (N, 2) ground-truth landmark positions.
        rng            : Seeded numpy random Generator for reproducibility.

    Returns:
        mu           : Final EKF state vector    shape (3 + 2K,)
        Sigma        : Final EKF covariance      shape (n, n)
        landmark_map : Fully populated landmark map.
        true_path    : List of [x, y] ground-truth positions.
        robot_path   : List of [x, y] EKF-estimated positions.
    """
    path    = build_predefined_path(num_laps=cfg.NUM_LAPS)
    n_steps = len(path) - 1

    true_path:  list = []
    robot_path: list = []

    print(f"  Path length  : {n_steps} steps  ({cfg.NUM_LAPS} lap(s))")
    print(f"  Animate      : {cfg.ANIMATE}  (every {cfg.ANIMATION_STEP} step(s))")
    print("\n  Running path simulation …\n")

    for t in range(n_steps):

        prev_pose = path[t]       # true pose at time t
        curr_pose = path[t + 1]   # true pose at time t+1

        # -------------------------------------------------------------- #
        # 1. Simulate noisy odometry                                       #
        #    Matches the format produced by a real /odom ROS topic.        #
        # -------------------------------------------------------------- #
        u = simulate_odometry(
            prev_true_pose = prev_pose,
            curr_true_pose = curr_pose,
            R              = cfg.R,
            rng            = rng,
        )

        # -------------------------------------------------------------- #
        # 2. EKF prediction — advances robot-pose block of mu and Sigma   #
        # -------------------------------------------------------------- #
        mu, Sigma = ekf_predict(
            mu    = mu,
            Sigma = Sigma,
            u     = u,
            R     = cfg.R,
        )

        # -------------------------------------------------------------- #
        # 3. Synthesise observations from the *true* pose                 #
        #    (noise is centred on ground truth, not the EKF estimate)     #
        # -------------------------------------------------------------- #
        observations = simulate_observations(
            true_pose      = curr_pose,
            landmarks_true = landmarks_true,
            Q              = cfg.Q,
            max_range      = cfg.MAX_RANGE,
            max_bearing    = cfg.MAX_BEARING,
            rng            = rng,
        )

        # -------------------------------------------------------------- #
        # 4. EKF correction — sequential update for each observation      #
        #    New landmarks are initialised automatically inside ekf_update #
        # -------------------------------------------------------------- #
        mu, Sigma, landmark_map = ekf_update(
            mu_bar       = mu,
            Sigma_bar    = Sigma,
            observations = observations,
            landmark_map = landmark_map,
            Q            = cfg.Q,
        )

        # -------------------------------------------------------------- #
        # 5. Record paths                                                  #
        # -------------------------------------------------------------- #
        true_path.append(list(curr_pose[:2]))   # [x, y] ground truth
        robot_path.append(mu[:2].tolist())      # [x, y] EKF estimate

        # -------------------------------------------------------------- #
        # 6. Live animation (optional)                                     #
        # -------------------------------------------------------------- #
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

        # -------------------------------------------------------------- #
        # 7. Progress log — every 50 steps                                #
        # -------------------------------------------------------------- #
        if t % 50 == 0 or t == n_steps - 1:
            print(
                f"  step {t:>4d}/{n_steps}  |  "
                f"robot ({mu[0]:6.2f}, {mu[1]:6.2f})  |  "
                f"landmarks in map: {len(landmark_map)}"
            )

    return mu, Sigma, landmark_map, true_path, robot_path


# ===========================================================================
# KEYBOARD MODE — INTERACTIVE CONTROL
# ===========================================================================

def _run_keyboard_mode(
    mu:             np.ndarray,
    Sigma:          np.ndarray,
    landmark_map:   dict,
    landmarks_true: np.ndarray,
    rng:            np.random.Generator,
) -> tuple:
    """
    Runs the interactive keyboard-driven simulation loop.

    The robot is controlled in real-time via key presses captured from the
    matplotlib figure window.  At each key press the simulation executes one
    full EKF predict–observe–update cycle and redraws the figure.

    Key bindings:
        W   — move forward  (cfg.KEYBOARD_STEP_M metres along current heading)
        S   — move backward (cfg.KEYBOARD_STEP_M metres against heading)
        A   — rotate left   (cfg.KEYBOARD_ROT_DEG degrees counter-clockwise)
        D   — rotate right  (cfg.KEYBOARD_ROT_DEG degrees clockwise)
        Q   — finish session and proceed to evaluation

    The robot's true pose is the single source of ground truth.  Noisy
    odometry and observations are synthesised from it at every step, so the
    EKF experiences the same noise model as in path mode.

    Boundary clamping ensures the robot cannot leave the map grid.

    Args:
        mu             : Initial EKF state vector         shape (3,)
        Sigma          : Initial EKF covariance            shape (3, 3)
        landmark_map   : Empty landmark map dict.
        landmarks_true : (N, 2) ground-truth landmark positions.
        rng            : numpy random Generator (may be seeded or free).

    Returns:
        mu           : Final EKF state vector    shape (3 + 2K,)
        Sigma        : Final EKF covariance      shape (n, n)
        landmark_map : Fully populated landmark map.
        true_path    : List of [x, y] ground-truth positions.
        robot_path   : List of [x, y] EKF-estimated positions.
    """

    # ------------------------------------------------------------------ #
    # Mutable session state held in a dict so the nested closure can      #
    # write to it without the 'nonlocal' keyword (Python 2 compat style,  #
    # but also cleaner than many nonlocal declarations).                  #
    # ------------------------------------------------------------------ #
    state: dict = {
        'true_pose':    np.array(cfg.INIT_ROBOT_POSE, dtype=float),
        'mu':           mu,
        'Sigma':        Sigma,
        'landmark_map': landmark_map,
        'true_path':    [],          # list of [x, y]
        'robot_path':   [],          # list of [x, y]
        'step':         0,
        'done':         False,
    }

    # Pre-compute step sizes from config (converted once for efficiency)
    _step_m   = float(cfg.KEYBOARD_STEP_M)
    _rot_rad  = float(np.deg2rad(cfg.KEYBOARD_ROT_DEG))

    # ------------------------------------------------------------------ #
    # Inner helpers                                                        #
    # ------------------------------------------------------------------ #

    def _clamp_pose(pose: np.ndarray) -> np.ndarray:
        """
        Clamps x and y to the interior of the map grid (1 cell margin).

        The 1-cell margin keeps the robot clear of the outer walls defined
        in build_map_grid(), where cell 0 and cell 24 (for a 25-cell map)
        are obstacles.  Heading θ is left unchanged.

        Args:
            pose : np.ndarray shape (3,) — [x, y, θ].

        Returns:
            clamped : np.ndarray shape (3,) — pose with x, y bounded.
        """
        clamped    = pose.copy()
        clamped[0] = np.clip(pose[0], 1.0, float(cfg.MAP_WIDTH  - 1))
        clamped[1] = np.clip(pose[1], 1.0, float(cfg.MAP_HEIGHT - 1))
        return clamped

    def _ekf_step(prev_pose: np.ndarray, curr_pose: np.ndarray) -> None:
        """
        Executes one full EKF predict → observe → update cycle.

        Writes results back into ``state`` in-place.

        Args:
            prev_pose : True robot pose before the motion  shape (3,).
            curr_pose : True robot pose after  the motion  shape (3,).
        """
        # 1. Noisy odometry from the two true poses
        u = simulate_odometry(
            prev_true_pose = prev_pose.tolist(),
            curr_true_pose = curr_pose.tolist(),
            R              = cfg.R,
            rng            = rng,
        )

        # 2. EKF prediction
        state['mu'], state['Sigma'] = ekf_predict(
            mu    = state['mu'],
            Sigma = state['Sigma'],
            u     = u,
            R     = cfg.R,
        )

        # 3. Synthesise observations from the *true* current pose
        observations = simulate_observations(
            true_pose      = curr_pose.tolist(),
            landmarks_true = landmarks_true,
            Q              = cfg.Q,
            max_range      = cfg.MAX_RANGE,
            max_bearing    = cfg.MAX_BEARING,
            rng            = rng,
        )

        # 4. EKF correction
        state['mu'], state['Sigma'], state['landmark_map'] = ekf_update(
            mu_bar       = state['mu'],
            Sigma_bar    = state['Sigma'],
            observations = observations,
            landmark_map = state['landmark_map'],
            Q            = cfg.Q,
        )

        # 5. Record paths
        state['true_path'].append(curr_pose[:2].tolist())
        state['robot_path'].append(state['mu'][:2].tolist())

    def _redraw(title_suffix: str = "") -> None:
        """
        Refreshes the live matplotlib figure after a key press.

        Calls ``update_plot`` (which reuses the existing figure window via
        the module-level cache in plotting.py) and appends a status line to
        the axes title showing the current step count and key bindings.

        Args:
            title_suffix : Optional extra text appended to the step counter.
        """
        update_plot(
            true_path      = state['true_path'],
            robot_path     = state['robot_path'],
            landmarks_true = landmarks_true,
            mu             = state['mu'],
            Sigma          = state['Sigma'],
            landmark_map   = state['landmark_map'],
            cfg            = cfg,
            step           = state['step'],
        )

        # Overwrite the axes title with key-binding reminder
        ax = plt.gca()
        ax.set_title(
            f"Step {state['step']}{title_suffix}  |  "
            f"W/S: fwd/back  A/D: turn  Q: finish",
            fontsize=9,
        )
        plt.draw()

    def _on_key(event) -> None:
        """
        Matplotlib key-press event handler.

        Translates a key press into a robot motion, runs one EKF step, and
        redraws the figure.  Invalid or unrecognised keys are silently ignored.

        Bound keys:
            W  — forward translation
            S  — backward translation
            A  — left rotation
            D  — right rotation
            Q  — quit (sets state['done'] = True, closes the figure)

        Args:
            event : matplotlib KeyEvent — carries the ``event.key`` string.
        """
        if state['done'] or event.key is None:
            return

        key       = event.key.lower()
        prev_pose = state['true_pose'].copy()
        new_pose  = prev_pose.copy()

        # ---------------------------------------------------------- #
        # Motion commands — mutate new_pose according to the key      #
        # ---------------------------------------------------------- #
        if key == 'w':
            # Translate forward along current heading
            new_pose[0] += _step_m * np.cos(prev_pose[2])
            new_pose[1] += _step_m * np.sin(prev_pose[2])

        elif key == 's':
            # Translate backward against current heading
            new_pose[0] -= _step_m * np.cos(prev_pose[2])
            new_pose[1] -= _step_m * np.sin(prev_pose[2])

        elif key == 'a':
            # Rotate counter-clockwise (left)
            new_pose[2] = _normalise_angle(prev_pose[2] + _rot_rad)

        elif key == 'd':
            # Rotate clockwise (right)
            new_pose[2] = _normalise_angle(prev_pose[2] - _rot_rad)

        elif key == 'q':
            # Finish session — set flag and close the interactive window
            print("\n  Q pressed — ending keyboard session …")
            state['done'] = True
            plt.close('all')
            return

        else:
            # Unrecognised key — ignore silently
            return

        # ---------------------------------------------------------- #
        # Boundary clamping — keep robot inside map walls             #
        # ---------------------------------------------------------- #
        new_pose = _clamp_pose(new_pose)

        # ---------------------------------------------------------- #
        # EKF step and state update                                   #
        # ---------------------------------------------------------- #
        state['true_pose'] = new_pose
        state['step']     += 1

        _ekf_step(prev_pose, new_pose)

        # ---------------------------------------------------------- #
        # Console log (same cadence as path mode: every 10 steps)    #
        # ---------------------------------------------------------- #
        if state['step'] % 10 == 0:
            print(
                f"  step {state['step']:>4d}  |  "
                f"true ({new_pose[0]:5.2f}, {new_pose[1]:5.2f})  |  "
                f"EKF  ({state['mu'][0]:5.2f}, {state['mu'][1]:5.2f})  |  "
                f"landmarks: {len(state['landmark_map'])}"
            )

        # ---------------------------------------------------------- #
        # Redraw figure                                               #
        # ---------------------------------------------------------- #
        _redraw()

    # ------------------------------------------------------------------ #
    # Figure setup                                                         #
    # ------------------------------------------------------------------ #
    print(f"  Starting pose : ({cfg.INIT_ROBOT_POSE[0]:.1f}, "
          f"{cfg.INIT_ROBOT_POSE[1]:.1f},  "
          f"θ={np.degrees(cfg.INIT_ROBOT_POSE[2]):.0f}°)")
    print(f"  Step size     : {_step_m} m  |  "
          f"Rotation step : {cfg.KEYBOARD_ROT_DEG}°\n")
    print("  ┌─────────────────────────────────────┐")
    print("  │  KEY BINDINGS                       │")
    print("  │  W / S  — forward / backward        │")
    print("  │  A / D  — turn left / right         │")
    print("  │  Q      — finish and evaluate       │")
    print("  └─────────────────────────────────────┘")
    print("\n  Focus the matplotlib window and use the keys above.\n")

    # Perform initial draw (creates the figure via update_plot's cache)
    _redraw(title_suffix="  [waiting for input]")

    # Connect key-press handler to whichever figure update_plot created
    fig = plt.gcf()
    fig.canvas.mpl_connect('key_press_event', _on_key)

    # ------------------------------------------------------------------ #
    # Event loop — plt.pause() yields control to the GUI event loop       #
    # so key-press callbacks are fired while this thread sleeps.          #
    # The loop exits when the user presses Q (state['done'] = True).      #
    # ------------------------------------------------------------------ #
    while not state['done']:
        try:
            plt.pause(0.05)   # 50 ms sleep — keeps GUI responsive
        except Exception:
            # Window was closed externally (e.g. clicking the X button)
            state['done'] = True
            break

    return (
        state['mu'],
        state['Sigma'],
        state['landmark_map'],
        state['true_path'],
        state['robot_path'],
    )


# ===========================================================================
# POST-LOOP: EVALUATION AND FINAL PLOT
# ===========================================================================

def _evaluate_and_plot(
    mu:             np.ndarray,
    Sigma:          np.ndarray,
    landmark_map:   dict,
    landmarks_true: np.ndarray,
    true_path:      list,
    robot_path:     list,
) -> None:
    """
    Runs the post-simulation evaluation pipeline and shows the final plot.

    This function is shared by both path mode and keyboard mode so neither
    loop has to duplicate evaluation or plotting logic.

    Steps:
        1. Procrustes alignment + RMSE computation (via ``evaluate``).
        2. Formatted report printed to stdout.
        3. Optional CSV output (governed by cfg.SAVE_CSV).
        4. Final static plot — with Procrustes alignment subplot when at
           least two landmarks were observed.

    Args:
        mu             : Final EKF state vector.
        Sigma          : Final EKF covariance.
        landmark_map   : Final landmark map dict.
        landmarks_true : (N, 2) ground-truth landmark positions.
        true_path      : List of [x, y] ground-truth robot positions.
        robot_path     : List of [x, y] EKF-estimated robot positions.
    """
    if len(true_path) == 0:
        print("\n  [evaluate] No steps recorded — skipping evaluation.")
        return

    print("\n" + "=" * 52)
    print("  Simulation complete — running evaluation …")
    print("=" * 52)

    eval_result = evaluate(
        mu             = mu,
        landmark_map   = landmark_map,
        landmarks_true = landmarks_true,
        cfg            = cfg,
    )

    aligned_landmarks = (
        eval_result['aligned_landmarks']
        if eval_result is not None else None
    )

    plot_final(
        true_path         = true_path,
        robot_path        = robot_path,
        landmarks_true    = landmarks_true,
        mu                = mu,
        Sigma             = Sigma,
        landmark_map      = landmark_map,
        cfg               = cfg,
        aligned_landmarks = aligned_landmarks,
    )


# ===========================================================================
# TOP-LEVEL RUNNER
# ===========================================================================

def run() -> None:
    """
    Orchestrates the full EKF SLAM simulation: prompts → loop → evaluation.

    Prompt sequence:
        1. Map configuration (1–4 landmark layouts)
        2. Control mode      (path or keyboard)

    Then delegates to _run_path_mode() or _run_keyboard_mode() and finally
    calls _evaluate_and_plot() which is shared by both modes.
    """

    # ------------------------------------------------------------------ #
    # 1. Map selection                                                     #
    # ------------------------------------------------------------------ #
    map_option               = _prompt_map_choice()
    map_grid, landmarks_true = get_map(map_option)

    print(f"\n  Map option   : {map_option}  ({len(landmarks_true)} landmarks)")

    # ------------------------------------------------------------------ #
    # 2. Control-mode selection                                            #
    # ------------------------------------------------------------------ #
    control_mode = _prompt_control_mode()

    print(f"  Control mode : {control_mode}\n")

    # ------------------------------------------------------------------ #
    # 3. Determine starting pose and initialise EKF                       #
    # ------------------------------------------------------------------ #
    if control_mode == 'path':
        # Seed pose from the first waypoint of the predefined trajectory
        path       = build_predefined_path(num_laps=cfg.NUM_LAPS)
        init_pose  = np.array(path[0], dtype=float)
    else:
        # Keyboard mode: start from the user-defined pose in config.py
        init_pose  = np.array(cfg.INIT_ROBOT_POSE, dtype=float)

    mu, Sigma, landmark_map = _initialise_ekf(init_pose)

    # Seeded RNG for reproducibility in path mode.
    # Keyboard mode uses the same RNG; the seed has less effect because
    # timing and key sequence introduce non-determinism anyway.
    rng = np.random.default_rng(seed=42)

    # ------------------------------------------------------------------ #
    # 4. Run the selected control mode                                     #
    # ------------------------------------------------------------------ #
    if control_mode == 'path':
        mu, Sigma, landmark_map, true_path, robot_path = _run_path_mode(
            mu             = mu,
            Sigma          = Sigma,
            landmark_map   = landmark_map,
            landmarks_true = landmarks_true,
            rng            = rng,
        )
    elif control_mode == 'keyboard':
        mu, Sigma, landmark_map, true_path, robot_path = _run_keyboard_mode(
            mu             = mu,
            Sigma          = Sigma,
            landmark_map   = landmark_map,
            landmarks_true = landmarks_true,
            rng            = rng,
        )

    # ------------------------------------------------------------------ #
    # 5. Shared evaluation and final plot                                  #
    # ------------------------------------------------------------------ #
    _evaluate_and_plot(
        mu             = mu,
        Sigma          = Sigma,
        landmark_map   = landmark_map,
        landmarks_true = landmarks_true,
        true_path      = true_path,
        robot_path     = robot_path,
    )


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("\n\n  Simulation interrupted by user (Ctrl-C).")
        sys.exit(0)