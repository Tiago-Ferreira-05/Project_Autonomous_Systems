"""
Simulation Environment — Map, Landmarks, and Predefined Path.

Reference: Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 10
           micro_simulador_V-2_1.py (friend's reference implementation)

This module defines the simulated world in which the EKF SLAM algorithm
is validated before being deployed on real robot data (rosbag, Session 4).

The environment is a 25x25 grid (each cell = 1 metre) with:
    - Outer walls on all four borders
    - One internal obstacle column to create a non-trivial map structure
    - A set of true landmark positions (ArUco marker proxies)

The robot follows a hardcoded two-lap rectangular trajectory designed to:
    1. Visit all landmarks at least once (loop closure opportunity)
    2. Re-observe landmarks on the second lap (reduces covariance ellipses)
    3. Be fully reproducible across runs for fair comparison

Map options (matching the friend's simulator):
    1  — 5 landmarks  (standard experiment)
    2  — 3 landmarks  (reduced experiment)
    3  — 1 landmark   (degenerate/sanity check)
"""

import numpy as np


# =============================================================================
# MAP GRID
# =============================================================================

def build_map_grid() -> np.ndarray:
    """
    Builds the 25x25 occupancy grid.

    Cell convention:
        0 = free space
        1 = obstacle / wall

    Returns:
        grid : np.ndarray  shape (25, 25)  dtype int
               Row index = y coordinate, column index = x coordinate.
    """
    grid = np.zeros((25, 25), dtype=int)

    # --- Outer walls (all four borders) ---
    grid[0,  :]  = 1   # bottom wall   (y = 0)
    grid[-1, :]  = 1   # top wall      (y = 24)
    grid[:,  0]  = 1   # left wall     (x = 0)
    grid[:, -1]  = 1   # right wall    (x = 24)

    # --- Internal obstacle column ---
    # Creates a barrier that forces the robot to navigate around it,
    # producing a non-trivial path and partial observability of landmarks.
    grid[8:20, 12] = 1

    return grid


# =============================================================================
# LANDMARK CONFIGURATIONS
# =============================================================================

# True landmark positions for each map option.
# Coordinates are in metres, matching the grid (1 cell = 1 m).
# These are the ground-truth values used for Procrustes evaluation at the end.

_LANDMARKS_OPT1 = np.array([
    [19.0,  5.0],   # landmark 0 — bottom-right area
    [19.0, 15.0],   # landmark 1 — right wall, mid-height
    [12.0, 21.0],   # landmark 2 — top centre
    [ 3.0, 15.0],   # landmark 3 — left wall, mid-height
    [ 3.0,  5.0],   # landmark 4 — bottom-left area
], dtype=float)

_LANDMARKS_OPT2 = np.array([
    [19.0,  5.0],   # landmark 0
    [12.0, 21.0],   # landmark 1
    [ 3.0,  5.0],   # landmark 2
], dtype=float)

_LANDMARKS_OPT3 = np.array([
    [12.0, 21.0],   # landmark 0 — single landmark (degenerate case)
], dtype=float)

_LANDMARK_OPTIONS = {
    1: _LANDMARKS_OPT1,
    2: _LANDMARKS_OPT2,
    3: _LANDMARKS_OPT3,
}


# =============================================================================
# PUBLIC ENVIRONMENT API
# =============================================================================

def get_map(option: int) -> tuple:
    """
    Returns the map grid and true landmark positions for a given option.

    Args:
        option : int — map configuration to load.
                 1  →  5 landmarks  (standard, recommended for development)
                 2  →  3 landmarks  (faster experiment)
                 3  →  1 landmark   (sanity check / degenerate case)

    Returns:
        map_grid       : np.ndarray  shape (25, 25)  — occupancy grid
        landmarks_true : np.ndarray  shape (N, 2)    — true landmark positions

    Raises:
        ValueError if option is not in {1, 2, 3}.

    Example:
        map_grid, landmarks_true = get_map(1)
    """
    if option not in _LANDMARK_OPTIONS:
        raise ValueError(
            f"Unknown map option '{option}'. "
            f"Valid options are: {sorted(_LANDMARK_OPTIONS.keys())}"
        )

    map_grid       = build_map_grid()
    landmarks_true = _LANDMARK_OPTIONS[option].copy()

    return map_grid, landmarks_true


# =============================================================================
# PREDEFINED PATH
# =============================================================================

def build_predefined_path(num_laps: int = 2) -> list:
    """
    Builds a hardcoded rectangular trajectory for the robot.

    The path makes num_laps complete loops around the inside of the map,
    keeping a safe distance from walls and the internal obstacle.

    Each waypoint is [x, y, theta] where theta is the robot's heading in
    radians.  Consecutive waypoints are 1 metre apart (one grid cell).

    Args:
        num_laps : int — number of complete loops to perform (default 2).
                   More laps → more landmark re-observations → tighter Sigma.

    Returns:
        path : list of [x, y, theta]  — ordered waypoints.
               The robot starts at path[0] and ends at path[-1].

    Path shape (per lap):
        Segment 1 — right along the bottom  (y = 2,  x: 2 → 16,  theta = 0)
        Segment 2 — up the right side       (x = 16, y: 3 → 20,  theta = π/2)
        Segment 3 — left along the top      (y = 20, x: 15 → 5,  theta = π)
        Segment 4 — down the left side      (x = 5,  y: 19 → 2,  theta = -π/2)

    Note:
        The heading theta is set to the direction of travel along each segment,
        which ensures the simulated camera FOV faces the landmarks along the
        route.  In a real deployment the heading comes from odometry, but here
        it must be specified to give the sensor simulation a reference direction.
    """
    path = []

    for _lap in range(num_laps):
        # Segment 1: move right along the bottom  (theta = 0 rad = East)
        for x in range(2, 17):
            path.append([float(x), 2.0, 0.0])

        # Segment 2: move up the right side  (theta = π/2 = North)
        for y in range(3, 21):
            path.append([16.0, float(y), np.pi / 2])

        # Segment 3: move left along the top  (theta = π = West)
        for x in range(15, 4, -1):
            path.append([float(x), 20.0, np.pi])

        # Segment 4: move down the left side  (theta = -π/2 = South)
        for y in range(19, 1, -1):
            path.append([5.0, float(y), -np.pi / 2])

    return path


# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_initial_pose(path: list) -> np.ndarray:
    """
    Extracts the initial robot pose from the predefined path.

    Args:
        path : list of [x, y, theta] waypoints (from build_predefined_path)

    Returns:
        pose : np.ndarray  shape (3,)  — [x, y, theta]
    """
    return np.array(path[0], dtype=float)


def path_length(path: list) -> float:
    """
    Computes the total Euclidean length of the predefined path in metres.

    Useful for sanity-checking that the path covers the expected distance.

    Args:
        path : list of [x, y, theta] waypoints

    Returns:
        length : float — total path length in metres
    """
    pts    = np.array(path)[:, :2]          # extract x, y only
    diffs  = np.diff(pts, axis=0)           # step vectors
    length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    return length