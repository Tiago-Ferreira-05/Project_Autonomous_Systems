"""
Evaluation Metrics for EKF SLAM.
=================================

Provides two public functions:

    compute_rmse(landmarks_true, landmarks_estimated)
        Computes per-landmark Euclidean errors and the overall RMSE after
        Procrustes alignment has been applied.  The input *landmarks_estimated*
        must already be the Procrustes-aligned estimates (i.e. the Y_aligned
        output of procrustes_align()), not the raw EKF output.

    evaluate(mu, landmark_map, landmarks_true, cfg)
        End-to-end convenience wrapper called by the simulation runner.
        Extracts estimated landmark positions from the EKF state vector,
        calls procrustes_align(), calls compute_rmse(), prints a formatted
        report to stdout, and optionally writes a CSV file.

Design notes
------------
- Both functions are intentionally stateless (no class, no globals) so
  they can be unit-tested in isolation.
- CSV writing is guarded by cfg.SAVE_CSV; the CSV schema is compatible
  with the one produced by the reference micro_simulador_V-2_1.py so
  existing analysis scripts can be reused.
- The evaluate() function returns a result dict so the caller can use
  the numbers programmatically (e.g. for multi-run statistical studies).

Reference
---------
Probabilistic Robotics (Thrun, Burgard, Fox) — Chapter 10.
Stachniss, C. — Robot Mapping lecture notes (2013/2020).
"""

import os
import csv
import numpy as np
from typing import Dict, List, Optional, Tuple

from .procrustes import procrustes_align


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _extract_estimated_landmarks(
    mu:           np.ndarray,
    landmark_map: dict,
    n_true:       int,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extracts the estimated landmark positions from the EKF state vector
    for the landmarks that were actually observed during the simulation.

    Only landmarks whose ID falls within [0, n_true) are extracted, which
    guards against spurious IDs being injected.

    Args:
        mu           : Full EKF state vector  shape (3 + 2K,).
        landmark_map : Dict {lm_id -> start_index_in_mu}.
        n_true       : Number of ground-truth landmarks (upper bound on IDs).

    Returns:
        estimated : (M, 2) array of estimated positions for M observed lms.
        valid_ids : List of M landmark IDs in the same row order as estimated.
    """
    estimated  = []
    valid_ids  = []

    # Iterate in sorted ID order so rows of estimated match rows of
    # landmarks_true[valid_ids] — essential for paired RMSE computation.
    for lm_id in sorted(landmark_map.keys()):
        if lm_id < 0 or lm_id >= n_true:
            continue                         # ignore out-of-range IDs

        j = landmark_map[lm_id]
        if j + 1 >= len(mu):
            continue                         # state vector too short (safety)

        estimated.append([mu[j], mu[j + 1]])
        valid_ids.append(lm_id)

    return np.array(estimated, dtype=float), valid_ids


# ===========================================================================
# PUBLIC API
# ===========================================================================

def compute_rmse(
    landmarks_true:      np.ndarray,
    landmarks_estimated: np.ndarray,
) -> Dict:
    """
    Computes per-landmark Euclidean errors and overall RMSE.

    Both arrays must already be in the same coordinate frame — i.e.
    *landmarks_estimated* should be the Procrustes-aligned output, not
    the raw EKF estimates.

    Args:
        landmarks_true      : (N, 2) ground-truth positions.
        landmarks_estimated : (N, 2) aligned estimated positions.
                              Must have the same number of rows as
                              landmarks_true.

    Returns:
        results : dict with keys
            'errors'      — list of N per-landmark Euclidean distances (m)
            'mean_error'  — arithmetic mean of errors (m)
            'rmse'        — root-mean-squared error (m)
            'max_error'   — worst-case landmark error (m)
            'min_error'   — best-case landmark error (m)
            'n_landmarks' — N (number of landmarks evaluated)

    Raises:
        ValueError : If the two arrays do not have the same shape.
    """
    X = np.asarray(landmarks_true,      dtype=float)
    Y = np.asarray(landmarks_estimated, dtype=float)

    if X.shape != Y.shape:
        raise ValueError(
            f"compute_rmse: arrays must have the same shape, "
            f"got true={X.shape}, estimated={Y.shape}."
        )

    # Per-landmark Euclidean distance
    diff   = X - Y                                      # (N, 2)
    errors = np.linalg.norm(diff, axis=1).tolist()      # (N,)  list of floats

    n      = len(errors)
    rmse   = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    mean_e = float(np.mean(errors))
    max_e  = float(np.max(errors))
    min_e  = float(np.min(errors))

    return {
        'errors':       errors,
        'mean_error':   mean_e,
        'rmse':         rmse,
        'max_error':    max_e,
        'min_error':    min_e,
        'n_landmarks':  n,
    }


def evaluate(
    mu:             np.ndarray,
    landmark_map:   dict,
    landmarks_true: np.ndarray,
    cfg,
) -> Optional[Dict]:
    """
    End-to-end evaluation: extract → align → compute RMSE → report.

    This is the single function called by the simulation runner at the
    end of the simulation.  It orchestrates the full evaluation pipeline
    and returns a result dictionary for programmatic use.

    Args:
        mu             : Final EKF state vector  shape (3 + 2K,).
        landmark_map   : Dict {lm_id -> start_index_in_mu}.
        landmarks_true : (N, 2) ground-truth landmark positions.
        cfg            : The config module (provides SAVE_CSV, CSV_FILENAME).

    Returns:
        result : dict with keys
            'aligned_landmarks' — (M, 2) Procrustes-aligned EKF estimates
            'valid_ids'         — list of M landmark IDs that were observed
            'scale'             — Procrustes scale factor
            'R'                 — Procrustes rotation matrix (2, 2)
            't'                 — Procrustes translation vector (2,)
            'rmse'              — overall RMSE in metres
            'mean_error'        — mean per-landmark error in metres
            'max_error'         — worst-case landmark error in metres
            'min_error'         — best-case landmark error in metres
            'errors'            — list of per-landmark errors in metres

        Returns None (with a warning) if fewer than 2 landmarks were
        observed — Procrustes alignment requires at least 2 points.

    Side effects:
        - Prints a formatted evaluation report to stdout.
        - If cfg.SAVE_CSV is True, appends a summary row to cfg.CSV_FILENAME.
    """
    n_true = len(landmarks_true)

    # ------------------------------------------------------------------
    # Step 1 — Extract EKF-estimated positions for observed landmarks
    # ------------------------------------------------------------------
    estimated, valid_ids = _extract_estimated_landmarks(mu, landmark_map, n_true)

    if len(valid_ids) < 2:
        print(
            "\n[evaluate] WARNING: fewer than 2 landmarks observed "
            f"({len(valid_ids)}).  Procrustes alignment requires at least 2. "
            "Skipping evaluation."
        )
        return None

    # Ground-truth positions for the observed subset, in the same row order
    true_subset = landmarks_true[valid_ids]   # (M, 2)

    # ------------------------------------------------------------------
    # Step 2 — Procrustes alignment
    # ------------------------------------------------------------------
    aligned, scale, R, t = procrustes_align(X=true_subset, Y=estimated)

    # ------------------------------------------------------------------
    # Step 3 — RMSE
    # ------------------------------------------------------------------
    metrics = compute_rmse(
        landmarks_true      = true_subset,
        landmarks_estimated = aligned,
    )

    # ------------------------------------------------------------------
    # Step 4 — Console report
    # ------------------------------------------------------------------
    separator = "=" * 52

    print(f"\n{separator}")
    print("  EKF SLAM — Evaluation Report")
    print(separator)
    print(f"  Landmarks observed  : {len(valid_ids)} / {n_true}")
    print(f"  Procrustes scale    : {scale:.4f}")
    print(f"  RMSE                : {metrics['rmse']:.4f}  m")
    print(f"  Mean error          : {metrics['mean_error']:.4f}  m")
    print(f"  Max error           : {metrics['max_error']:.4f}  m")
    print(f"  Min error           : {metrics['min_error']:.4f}  m")
    print(f"\n  Per-landmark breakdown:")
    print(f"  {'ID':>4}   {'True (x,y)':^22}   {'EKF aligned (x,y)':^22}   {'Error (m)':>10}")
    print(f"  {'-'*4}   {'-'*22}   {'-'*22}   {'-'*10}")

    for i, lm_id in enumerate(valid_ids):
        tx, ty  = true_subset[i]
        ex, ey  = aligned[i]
        err     = metrics['errors'][i]
        print(
            f"  {lm_id:>4}   "
            f"({tx:8.3f}, {ty:8.3f})   "
            f"({ex:8.3f}, {ey:8.3f})   "
            f"{err:10.4f}"
        )

    print(separator)

    # ------------------------------------------------------------------
    # Step 5 — Optional CSV output
    # ------------------------------------------------------------------
    if cfg.SAVE_CSV:
        _write_csv(
            filename   = cfg.CSV_FILENAME,
            valid_ids  = valid_ids,
            true_sub   = true_subset,
            aligned    = aligned,
            metrics    = metrics,
            scale      = scale,
        )

    # ------------------------------------------------------------------
    # Step 6 — Return result dict for programmatic use
    # ------------------------------------------------------------------
    return {
        'aligned_landmarks': aligned,
        'valid_ids':         valid_ids,
        'scale':             scale,
        'R':                 R,
        't':                 t,
        'rmse':              metrics['rmse'],
        'mean_error':        metrics['mean_error'],
        'max_error':         metrics['max_error'],
        'min_error':         metrics['min_error'],
        'errors':            metrics['errors'],
    }


# ===========================================================================
# PRIVATE — CSV WRITER
# ===========================================================================

def _write_csv(
    filename:  str,
    valid_ids: List[int],
    true_sub:  np.ndarray,
    aligned:   np.ndarray,
    metrics:   Dict,
    scale:     float,
) -> None:
    """
    Writes the per-landmark evaluation results to a CSV file.

    Creates the output directory automatically if it does not exist.
    Each call *overwrites* the file so that re-running the simulation
    always produces a fresh, clean CSV.

    CSV schema:
        landmark_id, true_x, true_y, ekf_x, ekf_y, error_m,
        rmse, mean_error, max_error, procrustes_scale

    Args:
        filename  : Path to the output CSV file (from cfg.CSV_FILENAME).
        valid_ids : List of observed landmark IDs.
        true_sub  : (M, 2) ground-truth positions for observed landmarks.
        aligned   : (M, 2) Procrustes-aligned EKF estimates.
        metrics   : Output dict from compute_rmse().
        scale     : Procrustes scale factor.
    """
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

    with open(filename, 'w', newline='') as fh:
        writer = csv.writer(fh)

        # Header
        writer.writerow([
            'landmark_id',
            'true_x', 'true_y',
            'ekf_aligned_x', 'ekf_aligned_y',
            'error_m',
            'rmse', 'mean_error', 'max_error',
            'procrustes_scale',
        ])

        # One row per observed landmark
        for i, lm_id in enumerate(valid_ids):
            writer.writerow([
                lm_id,
                f"{true_sub[i, 0]:.6f}",
                f"{true_sub[i, 1]:.6f}",
                f"{aligned[i, 0]:.6f}",
                f"{aligned[i, 1]:.6f}",
                f"{metrics['errors'][i]:.6f}",
                f"{metrics['rmse']:.6f}",
                f"{metrics['mean_error']:.6f}",
                f"{metrics['max_error']:.6f}",
                f"{scale:.6f}",
            ])

    print(f"[evaluate] Results saved → {filename}")