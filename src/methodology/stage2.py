"""
Stage 2: Cluster-based false positive filtering.

Clusters anomalies detected in Stage 1 together with OK reference samples
to identify fault-pure clusters and filter false positives.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_stage2(
    x_values: NDArray,
    y_anomalies: NDArray,
    y_true: NDArray,
    label_mapping: dict,
    ok_reference_ratio: float,
    use_dtw: bool,
    n_clusters: int,
    random_state: int,
) -> NDArray:
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
        ok_reference_ratio: Ratio of OK samples to include (e.g., 0.01 = 1%)
        use_dtw: Use Dynamic Time Warping distance (requires sktime)
        n_clusters: Number of clusters to find
        random_state: Random seed for reproducibility

    Returns:
        y_clusters: Cluster assignments for samples sent to Stage 2
    """
    logger.subsection("Stage 2: Fault Clustering")
    logger.info(f"Clustering strategy: {n_clusters} clusters, DTW={use_dtw}")
    logger.info(f"OK reference ratio: {ok_reference_ratio:.1%}")
    logger.debug(f"Random state: {random_state}")

    # 1. Prepare: Select NOK samples + OK reference
    nok_indices = np.where(y_anomalies == 1)[0]
    ok_indices = np.where(y_anomalies == 0)[0]

    logger.debug(f"Available samples: {len(nok_indices)} NOK, {len(ok_indices)} OK")

    # Sample OK reference
    n_ok_sample = int(len(ok_indices) * ok_reference_ratio)
    np.random.seed(random_state)  # Ensure reproducibility
    ok_sample_indices = np.random.choice(ok_indices, size=n_ok_sample, replace=False)

    # Combine NOK + OK reference for clustering
    clustering_indices = np.concatenate([nok_indices, ok_sample_indices])
    x_cluster = x_values[clustering_indices]
    y_cluster_true = y_true[clustering_indices]

    # Track which samples are OK reference (for analysis)
    ok_reference_mask = np.zeros(len(clustering_indices), dtype=bool)
    ok_reference_mask[len(nok_indices) :] = True

    logger.info(
        f"Clustering input: {len(nok_indices)} NOK + {n_ok_sample} OK reference"
    )
    logger.info(f"Total samples for clustering: {len(clustering_indices)}")
    logger.debug(
        f"OK reference percentage: {n_ok_sample / len(clustering_indices):.1%}"
    )

    # 2. Run clustering
    if use_dtw:
        try:
            from sktime.clustering.k_means import TimeSeriesKMeans

            logger.info("Using TimeSeriesKMeans with DTW distance metric")
            model = TimeSeriesKMeans(
                n_clusters=n_clusters, metric="dtw", random_state=random_state
            )
        except ImportError:
            logger.warning("sktime not available, falling back to standard KMeans")
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        logger.info("Using standard KMeans with Euclidean distance")
        model = KMeans(n_clusters=n_clusters, random_state=random_state)

    logger.info("Fitting clustering model")
    y_clusters = model.fit_predict(x_cluster)

    n_clusters_found = len(np.unique(y_clusters))
    logger.info(f"Clustering complete: {n_clusters_found} clusters formed")

    if n_clusters_found < n_clusters:
        logger.warning(
            f"Found fewer clusters ({n_clusters_found}) than requested ({n_clusters})"
        )

    # 3. Analyze cluster composition
    logger.info("Analyzing cluster composition")
    _analyze_clusters(y_clusters, ok_reference_mask, y_cluster_true)

    logger.info("Stage 2 complete")

    return y_clusters


def _analyze_clusters(
    y_clusters: NDArray, ok_reference_mask: NDArray, y_cluster_true: NDArray
):
    """
    Analyze and log cluster composition.

    Classifies each cluster as:
    - Fault-pure: Real faults with low false positive contamination
    - OK-dominated: Mostly OK reference samples (likely false positives)
    - FP-dominated: Mostly false positives without OK reference
    - Mixed: Combination of the above

    Args:
        y_clusters: Cluster assignments
        ok_reference_mask: Boolean mask indicating OK reference samples
        y_cluster_true: Ground truth labels for evaluation
    """
    logger.info("Cluster composition analysis:")
    logger.info(
        f"{'Cluster':<10} {'Size':<8} {'OK Ref':<10} {'Real':<10} {'FalsePos':<10} {'OK%':<10} {'FP%':<10} {'Type':<15}"
    )
    logger.info("-" * 95)

    cluster_types = {"Fault-pure": 0, "OK-dominated": 0, "FP-dominated": 0, "Mixed": 0}

    for c in np.unique(y_clusters):
        cluster_mask = y_clusters == c
        n_total = cluster_mask.sum()

        # Count OK reference samples
        n_ok_ref = ok_reference_mask[cluster_mask].sum()

        # Among non-reference samples, count real faults vs false positives
        non_ref_mask = cluster_mask & ~ok_reference_mask
        y_non_ref = y_cluster_true[non_ref_mask]

        n_real_faults = (y_non_ref > 0).sum()
        n_false_pos = (y_non_ref == 0).sum()

        # Calculate percentages
        ok_ref_pct = (n_ok_ref / n_total * 100) if n_total > 0 else 0
        fp_pct = (
            (n_false_pos / (n_total - n_ok_ref) * 100)
            if (n_total - n_ok_ref) > 0
            else 0
        )

        # Classify cluster type
        if ok_ref_pct > 50:
            cluster_type = "OK-dominated"
        elif fp_pct > 70:
            cluster_type = "FP-dominated"
        elif n_real_faults > 0 and fp_pct < 30:
            cluster_type = "Fault-pure"
        else:
            cluster_type = "Mixed"

        cluster_types[cluster_type] += 1

        logger.info(
            f"{c:<10} {n_total:<8} {n_ok_ref:<10} {n_real_faults:<10} {n_false_pos:<10} "
            f"{ok_ref_pct:>6.1f}%   {fp_pct:>6.1f}%   {cluster_type:<15}"
        )

    # Summary statistics
    logger.info("-" * 95)
    logger.info(f"Cluster type distribution: {dict(cluster_types)}")
    logger.debug(
        f"Fault-pure clusters: {cluster_types['Fault-pure']} "
        f"(ideal for actionable fault detection)"
    )
    logger.debug(
        f"OK-dominated clusters: {cluster_types['OK-dominated']} "
        f"(can be filtered as false positives)"
    )
