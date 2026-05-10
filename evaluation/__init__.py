"""
Public API for the evaluation package.

Exposes the three functions used by the main simulation runner:

    procrustes_align : aligns EKF landmark estimates to ground truth via
                       Procrustes analysis (removes the arbitrary offset
                       introduced by the robot's starting reference frame).

    compute_rmse     : computes per-landmark and overall RMSE after alignment.

    evaluate         : convenience wrapper that runs both steps and prints
                       a formatted summary to stdout.
"""

from .procrustes import procrustes_align
from .metrics    import compute_rmse, evaluate

__all__ = ["procrustes_align", "compute_rmse", "evaluate"]