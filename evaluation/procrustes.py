"""
Procrustes Alignment for EKF SLAM Evaluation.
==============================================

Why Procrustes?
---------------
EKF SLAM builds its map in the robot's own reference frame, which is
anchored to wherever the robot happened to start.  The ground-truth
landmark positions live in a separate, absolute coordinate frame (e.g.
the lab floor grid).  Before we can compute any meaningful position
error we must find the rigid-body transformation (rotation + translation
+ uniform scale) that best maps the estimated map onto the ground-truth
map.  That optimal alignment is the solution to the *orthogonal Procrustes
problem*.

The function procrustes_align() solves:

    min_{s, R, t}  ||X - (s * Y @ R^T + t)||_F^2

where
    X  — ground-truth landmark positions     (N × 2)
    Y  — EKF-estimated landmark positions    (N × 2)
    s  — uniform scale factor                (scalar)
    R  — rotation matrix                     (2 × 2)
    t  — translation vector                  (1 × 2)

The closed-form solution uses singular value decomposition (SVD) of the
cross-covariance matrix H = X0^T @ Y0, where X0 and Y0 are the
mean-centred versions of X and Y.

The reflection fix (det(R) < 0 branch) ensures R is a proper rotation
and not a reflection, which would produce an unphysical mirrored map.

Reference
---------
Gower & Dijksterhuis, "Procrustes Problems", Oxford University Press, 2004.
Umeyama, S. (1991). "Least-squares estimation of transformation parameters
between two point patterns." IEEE TPAMI 13(4).
"""

import numpy as np
from typing import Tuple


def procrustes_align(
    X: np.ndarray,
    Y: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Aligns estimated landmark positions Y to ground-truth positions X
    using generalised Procrustes analysis (with uniform scale).

    The function finds the scalar s, rotation R, and translation t that
    minimise the Frobenius norm of the residual  X - (s * Y @ R^T + t).

    Args:
        X : Ground-truth landmark positions,   shape (N, 2).
            Acts as the *reference* — it is never modified.
        Y : EKF-estimated landmark positions,  shape (N, 2).
            This is the set that will be transformed.

    Returns:
        Y_aligned : Transformed version of Y aligned to X,  shape (N, 2).
        scale     : Optimal uniform scale factor  s  (float).
        R         : Optimal 2×2 rotation matrix.
        t         : Optimal translation vector,  shape (2,).

    Raises:
        ValueError : If X and Y do not have the same shape, or if fewer
                     than 2 landmarks are provided (Procrustes is undefined
                     for a single point in 2-D).

    Notes:
        - A reflection guard (det(R) < 0) is applied to ensure R is a
          proper rotation matrix.
        - If either point cloud is degenerate (all points coincide), the
          function returns Y unchanged with s=1, R=I, t=0 and logs a
          warning, rather than raising a division-by-zero error.

    Example:
        >>> X = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        >>> Y = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]])
        >>> Y_al, s, R, t = procrustes_align(X, Y)
        >>> np.allclose(Y_al, X, atol=1e-6)
        True
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if X.shape != Y.shape:
        raise ValueError(
            f"procrustes_align: X and Y must have the same shape, "
            f"got X={X.shape} and Y={Y.shape}."
        )

    n_points, dim = X.shape

    if n_points < 2:
        raise ValueError(
            f"procrustes_align: at least 2 landmarks are required for a "
            f"well-defined 2-D alignment, got {n_points}."
        )

    # ------------------------------------------------------------------
    # Step 1 — Centre both point clouds
    # ------------------------------------------------------------------
    mu_X = X.mean(axis=0)   # centroid of ground-truth   shape (2,)
    mu_Y = Y.mean(axis=0)   # centroid of estimates       shape (2,)

    X0 = X - mu_X           # mean-centred ground-truth   shape (N, 2)
    Y0 = Y - mu_Y           # mean-centred estimates       shape (N, 2)

    # ------------------------------------------------------------------
    # Step 2 — Degeneracy check
    # ------------------------------------------------------------------
    norm_X0 = np.linalg.norm(X0)
    norm_Y0 = np.linalg.norm(Y0)

    if norm_X0 < 1e-9 or norm_Y0 < 1e-9:
        # One or both point clouds are effectively a single point.
        # Alignment is trivial: translate centroids, no rotation or scale.
        print(
            "[procrustes_align] WARNING: degenerate point cloud "
            "(all landmarks coincide). Returning identity alignment."
        )
        identity_R = np.eye(dim)
        identity_t = mu_X - mu_Y
        Y_aligned  = Y + identity_t
        return Y_aligned, 1.0, identity_R, identity_t

    # ------------------------------------------------------------------
    # Step 3 — SVD of the cross-covariance matrix
    # ------------------------------------------------------------------
    # H = X0^T @ Y0   shape (dim, dim)
    # SVD: H = U @ diag(S) @ Vt
    H        = X0.T @ Y0
    U, S, Vt = np.linalg.svd(H)

    # ------------------------------------------------------------------
    # Step 4 — Optimal rotation (with reflection guard)
    # ------------------------------------------------------------------
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        # Flip the last singular vector to turn the reflection into a
        # proper rotation (det = +1).
        Vt[-1, :] *= -1
        R          = Vt.T @ U.T

    # ------------------------------------------------------------------
    # Step 5 — Optimal scale
    # ------------------------------------------------------------------
    # Umeyama (1991) closed-form: s = sum(S) / ||Y0||_F^2
    # This minimises  ||X0 - s * Y0 @ R^T||_F^2  over s > 0.
    # NOTE: divides by norm_Y0^2 (the SOURCE cloud), not norm_X0^2.
    scale = float(np.sum(S) / (norm_Y0 ** 2))

    # ------------------------------------------------------------------
    # Step 6 — Optimal translation
    # ------------------------------------------------------------------
    # Derived from setting d/dt of the cost to zero:
    #   t = mu_X - s * (mu_Y @ R)
    # This maps the centroid of Y to the centroid of X after scaling
    # and rotation.
    t = mu_X - scale * (mu_Y @ R)

    # ------------------------------------------------------------------
    # Step 7 — Apply transformation to Y
    # ------------------------------------------------------------------
    # Full transform:  Y_aligned = s * (Y @ R) + t
    # Expanding t:     = s * Y @ R  +  mu_X  -  s * mu_Y @ R
    #                  = s * (Y - mu_Y) @ R  +  mu_X
    #                  = s * Y0 @ R  +  mu_X
    #
    # Writing it in the CENTRED form (s * Y0 @ R + mu_X) avoids
    # floating-point cancellation that occurs when s, mu_Y, and R
    # are composed separately and then added.
    Y_aligned = scale * (Y0 @ R) + mu_X

    return Y_aligned, scale, R, t