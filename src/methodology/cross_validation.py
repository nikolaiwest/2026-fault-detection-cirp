"""Cross-validation utilities for two-stage pipeline."""

import numpy as np
from numpy.typing import NDArray


def prepare_cv_folds(
    x_values: NDArray,
    y_true: NDArray,
    n_splits: int,
    target_ok_per_fold: int | float | None,
    target_nok_per_fold: int | None,
    random_state: int,
) -> list[tuple[NDArray, NDArray]]:
    """
    Prepare cross-validation folds with stratified NOK splitting and OK upsampling.

    Strategy:
    - Split NOK samples into n_splits folds (stratified by fault class)
    - Each fold gets: ALL OK samples + 1/n NOK samples
    - Optionally downsample NOK to target_nok_per_fold for equal fold sizes
    - Optionally upsample OK based on target_ok_per_fold

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels (n_samples,)
        n_splits: Number of CV folds
        target_ok_per_fold: OK upsampling strategy:
            - float (0-1): ratio (e.g., 0.99 = 99% OK samples)
            - int: exact count (e.g., 5000 = exactly 5000 OK samples)
            - None: no upsampling (use original OK samples)
        target_nok_per_fold: If int, downsample each fold to this many NOK samples
                            (None = use all)
        random_state: Random seed for reproducibility

    Returns:
        list of tuples: [(x_fold_1, y_fold_1), (x_fold_2, y_fold_2), ...]
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    # Split OK vs NOK
    ok_mask = y_true == 0
    x_ok, y_ok = x_values[ok_mask], y_true[ok_mask]
    x_nok, y_nok = x_values[~ok_mask], y_true[~ok_mask]

    print(f"\nPreparing {n_splits}-fold cross-validation:")
    print(f"Total: {len(x_ok)} OK, {len(x_nok)} NOK")

    if target_nok_per_fold:
        print(f"Target: {target_nok_per_fold} NOK per fold (equal fold sizes)")
    else:
        print(f"Each fold: {len(x_ok)} OK + ~{len(x_nok)//n_splits} NOK")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    rng = np.random.RandomState(random_state)

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        # Get NOK samples for this fold
        x_nok_fold = x_nok[nok_idx]
        y_nok_fold = y_nok[nok_idx]

        # Optionally downsample NOK to target size
        if target_nok_per_fold and len(nok_idx) > target_nok_per_fold:
            sample_idx = rng.choice(
                len(nok_idx), size=target_nok_per_fold, replace=False
            )
            x_nok_fold = x_nok_fold[sample_idx]
            y_nok_fold = y_nok_fold[sample_idx]
            n_dropped = len(nok_idx) - target_nok_per_fold
            print(
                f"Fold {fold_idx+1}: Downsampled {len(nok_idx)} → {target_nok_per_fold} NOK (dropped {n_dropped})"
            )
        elif target_nok_per_fold and len(nok_idx) < target_nok_per_fold:
            print(
                f"Warning: Fold {fold_idx+1} has only {len(nok_idx)} NOK samples (target: {target_nok_per_fold})"
            )

        # Combine ALL OK + NOK fold
        x_combined = np.vstack([x_ok, x_nok_fold])
        y_combined = np.hstack([y_ok, y_nok_fold])

        # Determine target OK count based on parameter type
        n_ok = len(x_ok)
        n_nok = len(y_nok_fold)

        if target_ok_per_fold is None:
            # No upsampling
            n_ok_target = n_ok
            print(
                f"Fold {fold_idx+1}: No OK upsampling (using {n_ok} original samples)"
            )
        elif isinstance(target_ok_per_fold, float):
            # Ratio-based upsampling
            n_ok_target = int((target_ok_per_fold * n_nok) / (1 - target_ok_per_fold))
        elif isinstance(target_ok_per_fold, int):
            # Exact count upsampling
            n_ok_target = target_ok_per_fold
        else:
            raise ValueError(
                f"target_ok_per_fold must be int, float, or None, got {type(target_ok_per_fold)}"
            )

        # Apply upsampling if needed
        if n_ok_target > n_ok:
            smote = SMOTE(sampling_strategy={0: n_ok_target}, random_state=random_state)
            x_upsampled, y_upsampled = smote.fit_resample(x_combined, y_combined)

            if isinstance(target_ok_per_fold, float):
                print(
                    f"Fold {fold_idx+1}: Upsampled {n_ok} → {n_ok_target} OK (ratio: {target_ok_per_fold:.2%}, {n_ok_target/n_ok:.1f}x)"
                )
            else:
                print(
                    f"Fold {fold_idx+1}: Upsampled {n_ok} → {n_ok_target} OK (exact count, {n_ok_target/n_ok:.1f}x)"
                )
        elif n_ok_target < n_ok:
            # Downsampling OK (if exact count is less than available)
            sample_idx = rng.choice(n_ok, size=n_ok_target, replace=False)
            x_ok_sampled = x_combined[sample_idx]
            y_ok_sampled = y_combined[sample_idx]
            x_upsampled = np.vstack([x_ok_sampled, x_nok_fold])
            y_upsampled = np.hstack([y_ok_sampled, y_nok_fold])
            print(
                f"Fold {fold_idx+1}: Downsampled {n_ok} → {n_ok_target} OK (exact count)"
            )
        else:
            # No change needed
            x_upsampled, y_upsampled = x_combined, y_combined

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
