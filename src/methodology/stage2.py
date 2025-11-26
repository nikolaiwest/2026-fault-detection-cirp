"""Stage 2: Cluster-based false positive filtering."""

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans


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
    print("\n" + "=" * 70)
    print("STAGE 2: FAULT CLUSTERING")
    print("=" * 70)

    # 1. Prepare: Select NOK samples + OK reference
    nok_indices = np.where(y_anomalies == 1)[0]
    ok_indices = np.where(y_anomalies == 0)[0]

    # Sample OK reference
    n_ok_sample = int(len(ok_indices) * ok_reference_ratio)
    ok_sample_indices = np.random.choice(ok_indices, size=n_ok_sample, replace=False)

    # Combine NOK + OK reference for clustering
    clustering_indices = np.concatenate([nok_indices, ok_sample_indices])
    x_cluster = x_values[clustering_indices]
    y_cluster_true = y_true[clustering_indices]

    # Track which samples are OK reference (for analysis)
    ok_reference_mask = np.zeros(len(clustering_indices), dtype=bool)
    ok_reference_mask[len(nok_indices) :] = True

    print(
        f"Clustering input: {len(nok_indices)} NOK + {len(ok_sample_indices)} OK reference"
    )
    print(f"Total samples: {len(clustering_indices)}")

    # 2. Run clustering
    if use_dtw:
        try:
            from sktime.clustering.k_means import TimeSeriesKMeans

            model = TimeSeriesKMeans(
                n_clusters=n_clusters, metric="dtw", random_state=random_state
            )
            print("Using TimeSeriesKMeans with DTW")
        except ImportError:
            print("Warning: sktime not available, using standard KMeans")
            model = KMeans(n_clusters=n_clusters, random_state=random_state)
    else:
        print("Using standard KMeans (Euclidean distance)")
        model = KMeans(n_clusters=n_clusters, random_state=random_state)

    y_clusters = model.fit_predict(x_cluster)
    print(f"Found {len(np.unique(y_clusters))} clusters")

    # 3. Analyze cluster composition
    _analyze_clusters(y_clusters, ok_reference_mask, y_cluster_true)

    return y_clusters


def _analyze_clusters(
    y_clusters: NDArray, ok_reference_mask: NDArray, y_cluster_true: NDArray
):
    """Analyze and print cluster composition table."""
    print(
        f"\n{'Cluster':<10} {'Size':<8} {'OK Ref':<10} {'Real':<10} {'FalsePos':<10} {'OK%':<10} {'FP%':<10} {'Type':<15}"
    )
    print("-" * 95)

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

        print(
            f"{c:<10} {n_total:<8} {n_ok_ref:<10} {n_real_faults:<10} {n_false_pos:<10} {ok_ref_pct:>6.1f}%   {fp_pct:>6.1f}%   {cluster_type:<15}"
        )
