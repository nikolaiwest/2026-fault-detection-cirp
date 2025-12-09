"""
Stage 2: Cluster-based false positive filtering.

Clusters anomalies detected in Stage 1 together with OK reference samples
to identify fault-pure clusters and filter false positives.
"""

from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.stage_2 import STAGE2_MODELS, Stage2Model
from src.utils import get_logger

logger = get_logger(__name__)


def run_stage2(
    x_values: NDArray,
    y_anomalies: NDArray,
    y_true: NDArray,
    label_mapping: dict,
    model_name: str,
    target_ok_to_sample: Union[int, float],
    metric: str,
    n_clusters: int,
    random_state: int,
    ok_reference_threshold: int,
) -> tuple[NDArray, dict]:
    """
    Stage 2: Cluster-based false positive filtering.

    Clusters flagged anomalies together with OK reference samples to:
    1. Identify "Fault-pure" clusters (high confidence real faults)
    2. Identify "OK-dominated" clusters (false positives that cluster with OK refs)
    3. Identify "FP-dominated" clusters (false positives forming distinct groups)

    Args:
        x_values: Feature matrix
        y_anomalies: Stage 1 predictions (0=OK, 1=NOK)
        y_true: Ground truth for analysis (not used for clustering)
        label_mapping: Fault class names (for future use)
        model_name: Name of clustering model to use (from STAGE2_MODELS)
        target_ok_to_sample: OK samples to include as reference
            - If float (0-1): Use this fraction of available OK samples (e.g., 0.3 = 30%)
            - If int (≥1): Use exactly this many OK samples
        metric: Distance metric ('euclidean', 'dtw', 'msm', 'erp', etc.)
        n_clusters: Number of clusters to find
        random_state: Random seed for reproducibility
        ok_reference_threshold: Maximum OK references allowed per cluster for filtering
            - 0: Strict filtering (no OK allowed, default)
            - >0: Threshold filtering (allow up to N OK refs)
            - -1: No filtering (keep all clusters)

    Returns:
        tuple: (y_clusters_full, filter_stats)
            - y_clusters_full: Cluster assignments for full dataset (-1 for non-clustered)
            - filter_stats: Dictionary with filtering statistics

    Raises:
        ValueError: If model_name not found in STAGE2_MODELS registry
        ImportError: If required library (sktime/sklearn) not installed
    """

    # Step 1: Log stage configuration
    logger.subsection("Stage 2: Fault Clustering")
    logger.info(f"Model: {model_name}")
    logger.info(f"Distance metric: {metric}")
    logger.info(f"Number of clusters: {n_clusters}")
    logger.debug(f"Random state: {random_state}")

    # Step 2: Instantiate the "stage 2" model from the registry using model_name
    if model_name not in STAGE2_MODELS:
        available = list(STAGE2_MODELS.keys())
        logger.error(f"Unknown model '{model_name}'. Available models: {available}")
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    ModelClass = STAGE2_MODELS[model_name]
    model: Stage2Model = ModelClass(
        n_clusters=n_clusters, random_state=random_state, metric=metric
    )
    # Note: All remaining parameter are set via src/models/stage_2/hyperparameters.yml
    logger.debug(f"Instantiatied {ModelClass.__name__}")

    # Step 3: Sample OK references and prepare clustering input
    ok_indices = np.where(y_anomalies == 0)[0]  # Stage 1 predicted as OK
    nok_indices = np.where(y_anomalies == 1)[0]  # Stage 1 predicted as NOK
    logger.debug(f"Available samples: {len(nok_indices)} NOK, {len(ok_indices)} OK")

    # Determine how many OK samples to use as reference for clustering
    n_ok_total = len(ok_indices)
    n_ok_to_sample = _determine_ok_sample_count(target_ok_to_sample, n_ok_total, logger)

    # Sample OK references and combine with ALL NOK samples
    np.random.seed(random_state)
    ok_sample_indices = np.random.choice(ok_indices, size=n_ok_to_sample, replace=False)
    clustering_indices = np.concatenate([nok_indices, ok_sample_indices])
    x_cluster = x_values[clustering_indices]
    y_cluster_true = y_true[clustering_indices]  # Ground truth for later evaluation

    # Track which samples are OK references (used for filtering in Step 6)
    ok_reference_mask = np.zeros(len(clustering_indices), dtype=bool)
    ok_reference_mask[len(nok_indices) :] = True

    logger.info(f"Clustering input: {len(nok_indices)} NOK + {n_ok_to_sample} OK ref")
    logger.debug(f"OK reference: {n_ok_to_sample / len(clustering_indices):.1%}")

    # Step 4: Run clustering
    logger.info("Fitting clustering model (unsupervised)")
    y_clusters = model.fit_predict(x_cluster)
    logger.debug("Clustering complete")

    # Validate cluster count
    n_found = len(np.unique(y_clusters))
    logger.info(f"Formed {n_found} clusters")
    if n_found < n_clusters:
        logger.warning(f"Fewer clusters ({n_found}) than requested ({n_clusters})")
    elif n_found > n_clusters:
        logger.warning(f"More clusters ({n_found}) than requested ({n_clusters})")

    # Analyze cluster composition (debug only)
    logger.debug("Analyzing cluster composition")
    _analyze_clusters(y_clusters, ok_reference_mask, y_cluster_true)

    # Step 5: Generate predictions using rule-based filtering
    stage2_predictions = _apply_cluster_filtering_rules(
        y_clusters=y_clusters,
        ok_reference_mask=ok_reference_mask,
        ok_reference_threshold=ok_reference_threshold,
    )
    logger.debug("Rule-based predictions generated")

    # Step 6: Evaluate predictions
    y_binary = (y_cluster_true > 0).astype(int)
    prec = precision_score(y_binary, stage2_predictions, zero_division=0)
    reca = recall_score(y_binary, stage2_predictions, zero_division=0)
    f1_s = f1_score(y_binary, stage2_predictions, zero_division=0)
    logger.info(f"Precision={prec:.3f}, Recall={reca:.3f}, F1={f1_s:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_binary, stage2_predictions)
    logger.debug("Confusion Matrix:")
    logger.debug(f"             Predicted OK  Predicted NOK")
    logger.debug(f"Actual OK    {cm[0,0]:12d}  {cm[0,1]:13d}")
    logger.debug(f"Actual NOK   {cm[1,0]:12d}  {cm[1,1]:13d}")

    # Create cluster composition DataFrame
    cluster_stats = _create_cluster_stats_df(
        y_clusters, ok_reference_mask, y_cluster_true
    )

    logger.info("Stage 2 complete")

    return {
        "y_predictions": stage2_predictions,
        "y_clusters": y_clusters,
        "y_true": y_cluster_true,
        "ok_reference_mask": ok_reference_mask,
        "x_clustered": x_cluster,
        "metrics": {
            "precision": float(prec),
            "recall": float(reca),
            "f1": float(f1_s),
        },
        "confusion_matrix": cm,
        "cluster_stats": cluster_stats,
    }


def _determine_ok_sample_count(
    target_ok_to_sample: Union[int, float],
    n_ok_available: int,
    logger,
) -> int:
    """
    Calculate number of OK samples to use as reference.

    Args:
        target_ok_to_sample: Either ratio (0-1) or absolute count (≥1)
        n_ok_available: Number of OK samples available
        logger: Logger instance

    Returns:
        Number of OK samples to use (capped at available)
    """
    # Float: interpret as ratio
    if isinstance(target_ok_to_sample, float):
        if not 0 < target_ok_to_sample <= 1:
            raise ValueError(...)
        n_ok_sample = int(n_ok_available * target_ok_to_sample)
        logger.info(
            f"Target OK to sample: {target_ok_to_sample:.1%} = {n_ok_sample} samples"
        )

    # Int: interpret as absolute count
    elif isinstance(target_ok_to_sample, int):
        if target_ok_to_sample < 1:
            raise ValueError(...)
        n_ok_sample = target_ok_to_sample
        logger.info(f"Target OK to sample: exactly {n_ok_sample} samples")

    else:
        raise TypeError(...)

    # Ensure at least 1
    if n_ok_sample == 0:
        logger.warning("Computed 0 OK samples. Using at least 1.")
        n_ok_sample = 1

    # Cap at available
    n_ok_sample = min(n_ok_sample, n_ok_available)
    logger.debug(f"Final OK sample count: {n_ok_sample} (capped at available)")

    return n_ok_sample


def _apply_cluster_filtering_rules(
    y_clusters: NDArray,
    ok_reference_mask: NDArray,
    ok_reference_threshold: int,
) -> NDArray:
    """
    Apply rule-based filtering to convert cluster assignments to binary predictions.

    Rule: If a cluster contains MORE than ok_reference_threshold OK references,
    it's considered contaminated → all samples in that cluster → predicted as OK (0).
    Otherwise, cluster is "fault-pure" → all samples → predicted as NOK (1).

    Args:
        y_clusters: Cluster assignments for each sample
        ok_reference_mask: Boolean mask indicating OK reference samples
        ok_reference_threshold: Maximum allowed OK references per cluster
            - 0: Strict (no OK allowed)
            - >0: Allow up to N OK references
            - -1: No filtering (all clusters kept as NOK)

    Returns:
        Binary predictions (0=OK, 1=NOK) for each sample
    """
    predictions = np.zeros(len(y_clusters), dtype=int)  # Default: all OK

    # Get unique clusters
    unique_clusters = np.unique(y_clusters)

    for cluster_id in unique_clusters:
        cluster_mask = y_clusters == cluster_id
        n_ok_ref = ok_reference_mask[cluster_mask].sum()

        # Apply filtering rule
        if ok_reference_threshold == -1:
            # No filtering: keep all clusters as NOK
            keep_cluster = True
        elif ok_reference_threshold == 0:
            # Strict: only keep if 0 OK references
            keep_cluster = n_ok_ref == 0
        else:
            # Threshold: keep if OK refs <= threshold
            keep_cluster = n_ok_ref <= ok_reference_threshold

        # Set predictions for this cluster
        if keep_cluster:
            predictions[cluster_mask] = 1  # Predict as NOK (fault)
        # else: remains 0 (OK, filtered out)

    return predictions


def _create_cluster_stats_df(
    y_clusters: NDArray,
    ok_reference_mask: NDArray,
    y_cluster_true: NDArray,
) -> pd.DataFrame:
    """
    Create DataFrame with cluster composition statistics.

    Returns:
        DataFrame with columns: cluster_id, size, n_real_faults,
                                n_false_positives, n_ok_references, purity
    """
    stats = []

    for c in np.unique(y_clusters):
        cluster_mask = y_clusters == c
        n_total = cluster_mask.sum()

        # Count OK reference samples
        n_ok_ref = ok_reference_mask[cluster_mask].sum()

        # Among non-reference samples, count real faults vs false positives
        non_ref_mask = cluster_mask & ~ok_reference_mask
        y_non_ref = y_cluster_true[non_ref_mask]

        n_real_faults = int((y_non_ref > 0).sum())
        n_false_pos = int((y_non_ref == 0).sum())

        # Calculate purity (fraction of real faults among non-references)
        purity = n_real_faults / len(y_non_ref) if len(y_non_ref) > 0 else 0.0

        stats.append(
            {
                "cluster_id": int(c),
                "size": int(n_total),
                "n_real_faults": n_real_faults,
                "n_false_positives": n_false_pos,
                "n_ok_references": int(n_ok_ref),
                "purity": float(purity),
            }
        )

    return pd.DataFrame(stats)


def _analyze_clusters(
    y_clusters: NDArray, ok_reference_mask: NDArray, y_cluster_true: NDArray
):
    """
    Analyze and log cluster composition in tabular format.

    Shows for each cluster:
    - Size
    - NOK (actual): Real faults from ground truth
    - OK (false pos): Misclassified as NOK by Stage 1
    - OK (reference): Intentionally added OK samples
    """
    logger.debug("Cluster composition:")
    logger.debug(
        f"{'Cluster':<10} {'Size':<8} {'NOK (actual)':<15} {'OK (false pos)':<18} {'OK (reference)':<18}"
    )
    logger.debug("-" * 95)

    for c in np.unique(y_clusters):
        cluster_mask = y_clusters == c
        n_total = cluster_mask.sum()

        # Count OK reference samples (intentionally added)
        n_ok_ref = ok_reference_mask[cluster_mask].sum()

        # Among non-reference samples, count real faults vs false positives
        non_ref_mask = cluster_mask & ~ok_reference_mask
        y_non_ref = y_cluster_true[non_ref_mask]

        n_real_faults = (y_non_ref > 0).sum()
        n_false_pos = (y_non_ref == 0).sum()

        logger.debug(
            f"{c:<10} {n_total:<8} {n_real_faults:<15} {n_false_pos:<18} {n_ok_ref:<18}"
        )
