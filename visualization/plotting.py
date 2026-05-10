"""
Visualisation — EKF SLAM Live Animation and Final Plot.
========================================================

Provides two public functions consumed by the main simulation runner:

    update_plot(state, config)
        Called every ANIMATION_STEP timesteps during the simulation loop.
        Redraws the live matplotlib figure in-place without opening a new
        window.  Uses plt.pause() for a non-blocking update.

    plot_final(state, config)
        Called once after the simulation ends.  Produces a clean, static
        summary figure that can be saved to disk and/or displayed
        interactively.

Internal helpers (prefixed with _) handle:
    - Covariance ellipse computation and drawing
    - Robot heading arrow
    - Consistent colour/style lookup from config

Design notes
------------
- All figure and axis objects are kept in a module-level cache (_fig, _axes)
  so update_plot() reuses the same window across calls rather than spawning
  a new one every step — important for smooth animation.
- plot_final() always creates a fresh figure so it is independent of whether
  update_plot() was ever called (ANIMATE = False path).
- No hardcoded colours: every colour is read from the config module so the
  user can restyle the plots entirely from config.py.

Reference
---------
Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 10 figures.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level figure cache (used by update_plot for live animation)
# ---------------------------------------------------------------------------
_fig: Optional[plt.Figure] = None
_ax:  Optional[plt.Axes]   = None


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _covariance_ellipse(
    ax:     plt.Axes,
    mean:   np.ndarray,
    cov:    np.ndarray,
    n_std:  float,
    colour: str,
    lw:     float = 1.2,
    alpha:  float = 0.7,
) -> None:
    """
    Draws a 2-D covariance ellipse on *ax*.

    The ellipse represents the n_std-sigma confidence region of a 2-D
    Gaussian with the given mean and covariance.

    Args:
        ax     : Target matplotlib Axes.
        mean   : Centre of the ellipse  shape (2,).
        cov    : 2×2 covariance matrix.
        n_std  : Number of standard deviations (1 → ~68 %, 2 → ~95 %).
        colour : Edge colour string (e.g. 'orange').
        lw     : Line width.
        alpha  : Line opacity.
    """
    # Guard: skip degenerate or non-PSD covariances
    if cov.shape != (2, 2):
        return
    if not np.all(np.isfinite(cov)):
        return

    # Eigendecomposition → semi-axes and rotation angle
    eigenvals, eigenvecs = np.linalg.eigh(cov)

    # Clamp small negative eigenvalues caused by floating-point drift
    eigenvals = np.maximum(eigenvals, 0.0)

    # Sort descending so width >= height
    order     = eigenvals.argsort()[::-1]
    eigenvals = eigenvals[order]
    eigenvecs = eigenvecs[:, order]

    angle  = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    width  = 2.0 * n_std * np.sqrt(eigenvals[0])
    height = 2.0 * n_std * np.sqrt(eigenvals[1])

    ellipse = mpatches.Ellipse(
        xy          = mean,
        width       = width,
        height      = height,
        angle       = angle,
        facecolor   = 'none',
        edgecolor   = colour,
        linewidth   = lw,
        alpha       = alpha,
        linestyle   = '--',
    )
    ax.add_patch(ellipse)


def _robot_arrow(
    ax:     plt.Axes,
    x:      float,
    y:      float,
    theta:  float,
    colour: str,
    length: float = 0.6,
) -> None:
    """
    Draws a heading arrow for the robot pose on *ax*.

    Args:
        ax     : Target matplotlib Axes.
        x, y   : Robot position (metres).
        theta  : Robot heading (radians).
        colour : Arrow colour.
        length : Arrow shaft length in metres.
    """
    dx = length * np.cos(theta)
    dy = length * np.sin(theta)
    ax.annotate(
        "",
        xy      = (x + dx, y + dy),
        xytext  = (x, y),
        arrowprops=dict(
            arrowstyle  = "-|>",
            color       = colour,
            lw          = 1.5,
        ),
    )


def _draw_scene(
    ax:              plt.Axes,
    true_path:       np.ndarray,
    robot_path:      np.ndarray,
    landmarks_true:  np.ndarray,
    mu:              np.ndarray,
    Sigma:           np.ndarray,
    landmark_map:    dict,
    cfg,
) -> None:
    """
    Core drawing routine shared by update_plot() and plot_final().

    Clears *ax* and redraws:
        1. True robot path (ground truth)
        2. EKF-estimated robot path
        3. Current robot pose with heading arrow and covariance ellipse
        4. True landmark positions
        5. EKF-estimated landmark positions with covariance ellipses

    Args:
        ax             : Axes to draw on.
        true_path      : (T, 2) array of true [x, y] positions.
        robot_path     : (T, 2) array of EKF-estimated [x, y] positions.
        landmarks_true : (N, 2) array of ground-truth landmark positions.
        mu             : Full EKF state vector  shape (3 + 2N,).
        Sigma          : Full EKF covariance     shape (n, n).
        landmark_map   : Dict {lm_id -> start_index_in_mu}.
        cfg            : The config module (imported by the caller).
    """
    ax.cla()  # Clear axes but keep the figure window

    # ------------------------------------------------------------------ #
    # 1. Ground-truth robot path
    # ------------------------------------------------------------------ #
    if len(true_path) > 1:
        tp = np.asarray(true_path)
        ax.plot(
            tp[:, 0], tp[:, 1],
            color     = cfg.COLOR_TRUE_PATH,
            linewidth = 1.2,
            linestyle = '--',
            label     = 'True path',
            zorder    = 2,
        )

    # ------------------------------------------------------------------ #
    # 2. EKF-estimated robot path
    # ------------------------------------------------------------------ #
    if len(robot_path) > 1:
        rp = np.asarray(robot_path)
        ax.plot(
            rp[:, 0], rp[:, 1],
            color     = cfg.COLOR_EKF_PATH,
            linewidth = 1.4,
            label     = 'EKF path',
            zorder    = 3,
        )

    # ------------------------------------------------------------------ #
    # 3. Current robot pose
    # ------------------------------------------------------------------ #
    x, y, theta = mu[0], mu[1], mu[2]

    ax.plot(
        x, y,
        marker     = 'o',
        color      = cfg.COLOR_ROBOT_POSE,
        markersize = 7,
        zorder     = 5,
        label      = 'Robot (EKF)',
    )
    _robot_arrow(ax, x, y, theta, colour=cfg.COLOR_ROBOT_POSE)

    robot_cov = Sigma[0:2, 0:2]
    _covariance_ellipse(
        ax,
        mean   = mu[0:2],
        cov    = robot_cov,
        n_std  = cfg.ELLIPSE_N_STD,
        colour = cfg.COLOR_ROBOT_ELLIPSE,
    )

    # ------------------------------------------------------------------ #
    # 4. True landmark positions
    # ------------------------------------------------------------------ #
    ax.scatter(
        landmarks_true[:, 0],
        landmarks_true[:, 1],
        marker  = '*',
        s       = 120,
        color   = cfg.COLOR_TRUE_LANDMARK,
        zorder  = 4,
        label   = 'True landmarks',
    )

    # ------------------------------------------------------------------ #
    # 5. EKF-estimated landmark positions + covariance ellipses
    # ------------------------------------------------------------------ #
    ekf_lm_plotted = False
    for lm_id, j in landmark_map.items():
        if j + 1 >= len(mu):
            continue  # Safety: index out of range

        lm_x, lm_y = mu[j], mu[j + 1]
        lm_cov     = Sigma[j:j + 2, j:j + 2]

        ax.scatter(
            lm_x, lm_y,
            marker = 'x',
            s      = 60,
            color  = cfg.COLOR_EKF_LANDMARK,
            zorder = 4,
            label  = 'EKF landmarks' if not ekf_lm_plotted else '_nolegend_',
        )
        ekf_lm_plotted = True

        # Landmark ID label (small, offset so it doesn't overlap the marker)
        ax.text(
            lm_x + 0.3, lm_y + 0.3,
            str(lm_id),
            fontsize = 7,
            color    = cfg.COLOR_EKF_LANDMARK,
            zorder   = 6,
        )

        _covariance_ellipse(
            ax,
            mean   = np.array([lm_x, lm_y]),
            cov    = lm_cov,
            n_std  = cfg.ELLIPSE_N_STD,
            colour = cfg.COLOR_EKF_ELLIPSE,
        )

    # ------------------------------------------------------------------ #
    # Axes cosmetics
    # ------------------------------------------------------------------ #
    ax.set_xlim(0, cfg.MAP_WIDTH)
    ax.set_ylim(0, cfg.MAP_HEIGHT)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_xlabel('x  (m)', fontsize=9)
    ax.set_ylabel('y  (m)', fontsize=9)

    # Deduplicated legend (avoids repeated entries when called in a loop)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize  = 8,
        loc       = 'upper right',
        framealpha= 0.8,
    )


# ===========================================================================
# PUBLIC API
# ===========================================================================

def update_plot(
    true_path:      list,
    robot_path:     list,
    landmarks_true: np.ndarray,
    mu:             np.ndarray,
    Sigma:          np.ndarray,
    landmark_map:   dict,
    cfg,
    step:           int,
) -> None:
    """
    Updates the live animation figure for the current simulation step.

    Should be called every ANIMATION_STEP timesteps from the simulation
    loop.  The function reuses the same figure window across calls, so
    only one window is ever open.

    Args:
        true_path      : List of [x, y] true robot positions so far.
        robot_path     : List of [x, y] EKF-estimated positions so far.
        landmarks_true : (N, 2) ground-truth landmark positions.
        mu             : Current EKF state vector.
        Sigma          : Current EKF covariance matrix.
        landmark_map   : Dict mapping landmark IDs to state-vector indices.
        cfg            : The config module (imported by the caller).
        step           : Current simulation timestep (used in the title).
    """
    global _fig, _ax

    # Create figure once; reuse on subsequent calls
    if _fig is None or not plt.fignum_exists(_fig.number):
        _fig, _ax = plt.subplots(figsize=(8, 8))
        _fig.suptitle('EKF SLAM — Live Simulation', fontsize=12, fontweight='bold')
        plt.ion()   # Non-blocking interactive mode

    _ax.set_title(f'Step {step}', fontsize=10)

    _draw_scene(
        ax             = _ax,
        true_path      = true_path,
        robot_path     = robot_path,
        landmarks_true = landmarks_true,
        mu             = mu,
        Sigma          = Sigma,
        landmark_map   = landmark_map,
        cfg            = cfg,
    )

    _fig.tight_layout()
    plt.pause(cfg.PAUSE_TIME)   # Non-blocking redraw; duration from config


def plot_final(
    true_path:      list,
    robot_path:     list,
    landmarks_true: np.ndarray,
    mu:             np.ndarray,
    Sigma:          np.ndarray,
    landmark_map:   dict,
    cfg,
    aligned_landmarks: Optional[np.ndarray] = None,
) -> None:
    """
    Produces the final summary plot after the simulation ends.

    If *aligned_landmarks* is provided (i.e. Procrustes alignment was
    performed), a second subplot shows the estimated landmarks after
    alignment alongside the ground truth, making the residual error
    visually obvious.

    Args:
        true_path          : List of [x, y] true robot positions.
        robot_path         : List of [x, y] EKF-estimated positions.
        landmarks_true     : (N, 2) ground-truth landmark positions.
        mu                 : Final EKF state vector.
        Sigma              : Final EKF covariance matrix.
        landmark_map       : Dict mapping landmark IDs to state indices.
        cfg                : The config module.
        aligned_landmarks  : (N, 2) Procrustes-aligned EKF landmark
                             estimates.  Pass None to skip the second subplot.
    """
    # ------------------------------------------------------------------ #
    # Decide layout: one or two subplots
    # ------------------------------------------------------------------ #
    has_alignment = (aligned_landmarks is not None)
    ncols = 2 if has_alignment else 1
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
    fig.suptitle('EKF SLAM — Final Results', fontsize=13, fontweight='bold')

    ax_main = axes[0] if has_alignment else axes

    # ------------------------------------------------------------------ #
    # Left / only subplot: raw EKF estimate
    # ------------------------------------------------------------------ #
    ax_main.set_title('EKF SLAM Estimate', fontsize=11)
    _draw_scene(
        ax             = ax_main,
        true_path      = true_path,
        robot_path     = robot_path,
        landmarks_true = landmarks_true,
        mu             = mu,
        Sigma          = Sigma,
        landmark_map   = landmark_map,
        cfg            = cfg,
    )

    # ------------------------------------------------------------------ #
    # Right subplot: Procrustes-aligned landmark comparison
    # ------------------------------------------------------------------ #
    if has_alignment:
        ax_align = axes[1]
        ax_align.set_title('Landmark Alignment (Procrustes)', fontsize=11)
        ax_align.set_aspect('equal')
        ax_align.grid(True, linewidth=0.4, alpha=0.5)
        ax_align.set_xlabel('x  (m)', fontsize=9)
        ax_align.set_ylabel('y  (m)', fontsize=9)

        # True landmarks
        ax_align.scatter(
            landmarks_true[:, 0],
            landmarks_true[:, 1],
            marker = '*',
            s      = 140,
            color  = cfg.COLOR_TRUE_LANDMARK,
            zorder = 4,
            label  = 'True landmarks',
        )

        # Aligned estimated landmarks
        al = np.asarray(aligned_landmarks)
        ax_align.scatter(
            al[:, 0], al[:, 1],
            marker = 'x',
            s      = 80,
            color  = cfg.COLOR_EKF_LANDMARK,
            zorder = 4,
            label  = 'Aligned EKF landmarks',
        )

        # Auto-zoom: fit axes tightly around the data with padding
        n_lm  = min(len(landmarks_true), len(al))
        all_x = np.concatenate([landmarks_true[:n_lm, 0], al[:, 0]])
        all_y = np.concatenate([landmarks_true[:n_lm, 1], al[:, 1]])
        pad   = max((all_x.max() - all_x.min()), (all_y.max() - all_y.min())) * 0.15
        pad   = max(pad, 1.0)   # minimum 1 m padding
        ax_align.set_xlim(all_x.min() - pad, all_x.max() + pad)
        ax_align.set_ylim(all_y.min() - pad, all_y.max() + pad)

        # Error lines + per-landmark ID labels
        for i in range(n_lm):
            # Dotted line connecting true → aligned estimate
            ax_align.plot(
                [landmarks_true[i, 0], al[i, 0]],
                [landmarks_true[i, 1], al[i, 1]],
                color     = 'grey',
                linewidth = 0.9,
                linestyle = ':',
                zorder    = 3,
            )
            # ID label next to the true position
            ax_align.text(
                landmarks_true[i, 0] + pad * 0.12,
                landmarks_true[i, 1] + pad * 0.12,
                f'ID {i}',
                fontsize = 8,
                color    = cfg.COLOR_TRUE_LANDMARK,
                zorder   = 6,
            )

        ax_align.legend(fontsize=8, loc='best', framealpha=0.8)

    fig.tight_layout()

    # ------------------------------------------------------------------ #
    # Optional: save to disk
    # ------------------------------------------------------------------ #
    if cfg.SAVE_PLOT:
        import os
        os.makedirs(os.path.dirname(cfg.PLOT_FILENAME), exist_ok=True)
        fig.savefig(cfg.PLOT_FILENAME, dpi=cfg.PLOT_DPI, bbox_inches='tight')
        print(f"[plot_final] Figure saved → {cfg.PLOT_FILENAME}")

    # ------------------------------------------------------------------ #
    # Display — blocking so the user can inspect the final plot.
    # The program exits cleanly once the window is closed.
    # ------------------------------------------------------------------ #
    plt.ioff()   # Switch back to blocking mode for the final show
    print("\n  [plot_final] Close the figure window to exit.")
    plt.show()   # Blocks until the user closes the window — this is intentional