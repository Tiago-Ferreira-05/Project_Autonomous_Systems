"""
Public API for the simulation package.

This package provides everything needed to run the EKF SLAM micro-simulator:
    - environment.py : map grid, true landmark positions, predefined robot path
    - sensors.py     : noisy odometry and range-bearing observation simulation

The EKF core (prediction, update, motion/measurement models) lives in the
sibling package ekf_core/ and is intentionally kept separate so the same
algorithm can be reused with real ROS 2 data in Session 4 without any changes.

Typical usage in main_sim.py:
    from simulation import get_map, build_predefined_path
    from simulation import simulate_odometry, simulate_observations
"""

from .enviroment import get_map, build_predefined_path
from .sensors      import simulate_odometry, simulate_observations

__all__ = [
    "get_map",
    "build_predefined_path",
    "simulate_odometry",
    "simulate_observations",
]