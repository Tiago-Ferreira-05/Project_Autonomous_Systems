"""
Public API for the EKF SLAM core module.

Exposes only the three functions that the main simulation loop calls:
    - ekf_predict        : prediction step (motion model)
    - ekf_update         : correction step (measurement model + Kalman update)
    - initialise_landmark: first-time landmark insertion into state

All mathematical details are contained in the submodules:
    - motion_model.py  : odometry motion model  g(u, x)  and its Jacobian G
    - meas_model.py    : range-bearing model     h(x)     and its Jacobian H
    - ekf_slam.py      : EKF predict / update orchestration
"""

from .ekf_slam import ekf_predict, ekf_update, initialise_landmark

__all__ = ["ekf_predict", "ekf_update", "initialise_landmark"]