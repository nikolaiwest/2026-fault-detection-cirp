"""Cross-validation utilities for two-stage pipeline."""

import numpy as np
from numpy.typing import NDArray


def prepare_cv_folds(
    x_values: NDArray,
    y_true: NDArray,
    n_splits: int,
    target_ok_ratio: float,
    random_state: int,
) -> list[tuple[NDArray, NDArray]]:
    """
    Prepare cross-validation folds with stratified NOK splitting and OK upsampling.

    Strategy:
    - Split NOK samples into n_splits folds (stratified by fault class)
    - Each fold gets: ALL OK samples + 1/n NOK samples
    - Upsample OK to achieve target_ok_ratio (e.g., 99% OK)

    This reduces SMOTE multiplication from ~40x to ~4-5x, creating more
    realistic synthetic samples.

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels (n_samples,)
        n_splits: Number of CV folds
        target_ok_ratio: Target ratio of OK samples (e.g., 0.99 = 99% OK)
        random_state: Random seed for reproducibility

    Returns:
        list of tuples: [(x_fold_1, y_fold_1), (x_fold_2, y_fold_2), ...]
        Each fold is upsampled to target_ok_ratio
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    # Split OK vs NOK
    ok_mask = y_true == 0
    x_ok, y_ok = x_values[ok_mask], y_true[ok_mask]
    x_nok, y_nok = x_values[~ok_mask], y_true[~ok_mask]

    print(f"\nPreparing {n_splits}-fold cross-validation:")
    print(f"Total: {len(x_ok)} OK, {len(x_nok)} NOK")
    print(f"Each fold: {len(x_ok)} OK + ~{len(x_nok)//n_splits} NOK")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        # Combine ALL OK + current NOK fold
        x_combined = np.vstack([x_ok, x_nok[nok_idx]])
        y_combined = np.hstack([y_ok, y_nok[nok_idx]])

        # Upsample OK to target_ok_ratio
        n_ok = len(x_ok)
        n_nok = len(nok_idx)
        n_ok_target = int((target_ok_ratio * n_nok) / (1 - target_ok_ratio))

        if n_ok_target > n_ok:
            # Need to upsample OK
            smote = SMOTE(sampling_strategy={0: n_ok_target}, random_state=random_state)
            x_upsampled, y_upsampled = smote.fit_resample(x_combined, y_combined)
            print(
                f"Fold {fold_idx+1}: Upsampled {n_ok} → {n_ok_target} OK samples ({n_ok_target/n_ok:.1f}x)"
            )
        else:
            x_upsampled, y_upsampled = x_combined, y_combined
            print(f"Fold {fold_idx+1}: No upsampling needed")

        folds.append((x_upsampled, y_upsampled))

    return folds


def report_cv_results(results: list[dict]):
    """
    Report aggregated cross-validation results.

    TODO: Implement comprehensive metrics:
    - Stage 1: Average precision, recall, F1
    - Stage 2: Fault-pure cluster statistics
    - Two-stage: Final metrics after cluster filtering

    Args:
        results: List of dicts containing per-fold results
            Each dict contains: y_anomalies, anomaly_scores, y_clusters, y_true
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    n_folds = len(results)

    # Stage 1 metrics
    print("\nStage 1 Performance (averaged across folds):")
    # TODO: Calculate precision, recall, F1 per fold and average

    # Stage 2 metrics
    print("\nStage 2 Performance (averaged across folds):")
    # TODO: Calculate fault-pure cluster stats, filtering effectiveness

    print(f"\nCompleted {n_folds}-fold cross-validation")
