"""
Cross-validation utilities for two-stage pipeline.

Provides stratified k-fold cross-validation with OK upsampling and NOK
stratification to ensure balanced, representative folds for evaluation.
"""

import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_cv_folds(
    x_values: NDArray,
    y_values: NDArray,
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
        y_values: Ground truth labels (n_samples,)
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

    Raises:
        ValueError: If target_ok_per_fold has invalid type
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import StratifiedKFold

    logger.info(f"Preparing {n_splits}-fold cross-validation")
    logger.debug(f"Random state: {random_state}")

    # Split OK vs NOK
    ok_mask = y_values == 0
    x_ok, y_ok = x_values[ok_mask], y_values[ok_mask]
    x_nok, y_nok = x_values[~ok_mask], y_values[~ok_mask]

    logger.info(f"Dataset split: {len(x_ok)} OK samples, {len(x_nok)} NOK samples")
    logger.debug(f"OK ratio: {len(x_ok)/len(x_values):.1%}")

    if target_nok_per_fold:
        logger.info(f"Target NOK per fold: {target_nok_per_fold} (equal fold sizes)")
    else:
        estimated_nok = len(x_nok) // n_splits
        logger.info(f"Estimated NOK per fold: ~{estimated_nok} (stratified split)")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    rng = np.random.RandomState(random_state)

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        fold_num = fold_idx + 1
        logger.debug(f"Processing fold {fold_num}/{n_splits}")

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
            logger.info(
                f"Fold {fold_num}: Downsampled NOK {len(nok_idx)} → {target_nok_per_fold} (dropped {n_dropped})"
            )
        elif target_nok_per_fold and len(nok_idx) < target_nok_per_fold:
            logger.warning(
                f"Fold {fold_num}: Only {len(nok_idx)} NOK samples available (target: {target_nok_per_fold})"
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
            logger.info(f"Fold {fold_num}: No OK upsampling ({n_ok} original samples)")
        elif isinstance(target_ok_per_fold, float):
            # Ratio-based upsampling
            n_ok_target = int((target_ok_per_fold * n_nok) / (1 - target_ok_per_fold))
            logger.debug(
                f"Fold {fold_num}: Calculated OK target from ratio {target_ok_per_fold:.1%}: {n_ok_target}"
            )
        elif isinstance(target_ok_per_fold, int):
            # Exact count upsampling
            n_ok_target = target_ok_per_fold
            logger.debug(f"Fold {fold_num}: Using exact OK count: {n_ok_target}")
        else:
            logger.error(f"Invalid target_ok_per_fold type: {type(target_ok_per_fold)}")
            raise ValueError(
                f"target_ok_per_fold must be int, float, or None, got {type(target_ok_per_fold)}"
            )

        # Apply upsampling if needed
        if n_ok_target > n_ok:
            logger.debug(f"Fold {fold_num}: Applying SMOTE upsampling")
            smote = SMOTE(sampling_strategy={0: n_ok_target}, random_state=random_state)
            x_upsampled, y_upsampled = smote.fit_resample(x_combined, y_combined)

            if isinstance(target_ok_per_fold, float):
                logger.info(
                    f"Fold {fold_num}: Upsampled OK {n_ok} → {n_ok_target} (ratio {target_ok_per_fold:.1%}, {n_ok_target/n_ok:.1f}x)"
                )
            else:
                logger.info(
                    f"Fold {fold_num}: Upsampled OK {n_ok} → {n_ok_target} (exact count, {n_ok_target/n_ok:.1f}x)"
                )

        elif n_ok_target < n_ok:
            # Downsampling OK (if exact count is less than available)
            logger.debug(f"Fold {fold_num}: Downsampling OK samples")
            sample_idx = rng.choice(n_ok, size=n_ok_target, replace=False)
            x_ok_sampled = x_combined[sample_idx]
            y_ok_sampled = y_combined[sample_idx]
            x_upsampled = np.vstack([x_ok_sampled, x_nok_fold])
            y_upsampled = np.hstack([y_ok_sampled, y_nok_fold])
            logger.info(
                f"Fold {fold_num}: Downsampled OK {n_ok} → {n_ok_target} (exact count)"
            )
        else:
            # No change needed
            x_upsampled, y_upsampled = x_combined, y_combined
            logger.debug(f"Fold {fold_num}: No OK sampling needed (target == current)")

        logger.debug(
            f"Fold {fold_num}: Final shape X={x_upsampled.shape}, y={y_upsampled.shape}"
        )
        folds.append((x_upsampled, y_upsampled))

    logger.info(f"Created {len(folds)} cross-validation folds")
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
            Each dict contains: y_anomalies, anomaly_scores, y_clusters, y_values
    """
    logger.subsection("Cross-Validation Summary")

    n_folds = len(results)
    logger.info(f"Aggregating results from {n_folds} folds")

    # Stage 1 metrics
    logger.info("Stage 1 Performance (averaged across folds):")
    # TODO: Calculate precision, recall, F1 per fold and average
    logger.warning("Stage 1 aggregated metrics not yet implemented")

    # Stage 2 metrics
    logger.info("Stage 2 Performance (averaged across folds):")
    # TODO: Calculate fault-pure cluster stats, filtering effectiveness
    logger.warning("Stage 2 aggregated metrics not yet implemented")

    logger.info(f"Completed {n_folds}-fold cross-validation")
