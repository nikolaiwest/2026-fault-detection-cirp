"""
Proof-of-Concept: Two-Stage Unsupervised Quality Control Framework

This module demonstrates a novel two-stage approach for fault detection in
manufacturing processes without requiring labeled training data:

Stage 1: Over-sensitive anomaly detection
    - Uses one-class learning (Isolation Forest) with intentionally high
      contamination parameter to maximize recall
    - Trade-off: High false positive rate accepted, to be filtered in Stage 2

Stage 2: Cluster-based false positive filtering
    - Clusters flagged anomalies with OK reference samples
    - Identifies "Fault-pure" clusters (real faults only)
    - Identifies "OK-dominated" and "FP-dominated" clusters for filtering
    - Key insight: False positives cluster with OK references or form
      distinct FP-only clusters

Key Benefits:
    - High recall without proportional increase in false alarms
    - No labeled fault examples required for training
    - Identifies pure fault clusters with high confidence
    - Adapts to unknown contamination rates

Cross-Validation Strategy:
    - Splits NOK samples into 5 folds
    - Each fold: ALL OK samples + 1/5 NOK samples
    - Upsamples OK to 99% ratio per fold (~4-5x vs. 40x without CV)
    - Provides robust performance estimates

Configuration Parameters:
    CONTAMINATION: Stage 1 sensitivity (0.02 = 2% expected anomalies)
    OK_REFERENCE_RATIO: Ratio of OK samples for Stage 2 (0.01 = 1%)
    N_CLUSTERS: Number of clusters in Stage 2 (5 recommended for top-5 faults)

Example Output:
    Fold 1 might identify:
    - Cluster 0: 18 Real Faults, 0 False Pos → Fault-pure ✓
    - Cluster 3: 40 OK Ref, 0 Faults → OK-dominated (filter)
    - Cluster 1: 56 False Pos, 5 Real → FP-dominated (mixed)

TODO: 
- I will split this script into modules with the next commit.
- To use the script, SMOTE in the data pipeline has to be disabled.
"""

import numpy as np
from numpy.typing import NDArray
from pyod.models.iforest import IForest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from src.data import load_class_config, run_data_pipeline

# ============================================================================
# CONFIGURATION - Adjust these parameters
# ============================================================================

# Data pipeline parameters
CLASSES_TO_KEEP = "top5"  # Class configuration to use
TARGET_OK_RATIO = 0.99  # Target ratio of OK samples (0.99 = 99% OK, 1% faults)
FORCE_RELOAD = False  # Force reload from PyScrew
KEEP_EXCEPTIONS = False  # Keep measurement exceptions

# Stage 1: Anomaly detection parameters
CONTAMINATION = 0.02  # Expected fraction of anomalies (0.02 = 2%)
# Higher values = more sensitive (more recall, more FPs)
RANDOM_STATE = 42  # Random seed for reproducibility

# Stage 2: Clustering parameters
OK_REFERENCE_RATIO = 0.01  # Ratio of OK samples to include in clustering
# (0.01 = 1%, 0.10 = 10%)
N_CLUSTERS = 5  # Number of clusters to find
# Should roughly match number of fault classes + OK clusters
USE_DTW = False  # Use Dynamic Time Warping distance (requires sktime)
# False = Euclidean distance (faster)


def run_two_stage_pipeline():
    """
    Main entry point for the two-stage methodology.

    Executes complete pipeline with cross-validation:
    1. Loads screw-driving data
    2. Prepares 5-fold CV with OK upsampling
    3. Runs Stage 1 (anomaly detection) per fold
    4. Runs Stage 2 (fault clustering) per fold
    5. Reports aggregated metrics

    Returns:
        None (prints results to console)
    """

    # Load data (without upsampling - done per fold)
    classes_to_keep = load_class_config(CLASSES_TO_KEEP)
    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=FORCE_RELOAD,
        keep_exceptions=KEEP_EXCEPTIONS,
        classes_to_keep=classes_to_keep,
        target_ok_ratio=1.0,  # No upsampling here - done in CV folds
    )

    # Prepare cross-validation folds
    cv_folds = _prepare_cv_folds(x_values, y_true, n_splits=5)

    # Run two-stage pipeline on each fold
    results = []
    for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):

        print(f"\n{'='*70}")
        print(f"FOLD {fold_num}/{len(cv_folds)}")
        print(f"{'='*70}")

        # Stage 1: Anomaly detection
        y_anomalies, anomaly_scores = _run_stage1(x_fold, y_fold)

        # Stage 2: Fault clustering
        y_clusters = _run_stage2(x_fold, y_anomalies, y_fold, label_mapping)

        results.append(
            {
                "y_anomalies": y_anomalies,
                "anomaly_scores": anomaly_scores,
                "y_clusters": y_clusters,
                "y_true": y_fold,
            }
        )

    # Report aggregated results
    _report_cv_results(results)


def _prepare_cv_folds(x_values, y_true, n_splits=5):
    """
    Prepare cross-validation folds with stratified NOK splitting and OK upsampling.

    Strategy:
    - Split NOK samples into n_splits folds (stratified by fault class)
    - Each fold gets: ALL OK samples + 1/n NOK samples
    - Upsample OK to achieve TARGET_OK_RATIO (e.g., 99% OK)

    This reduces SMOTE multiplication from ~40x to ~4-5x, creating more
    realistic synthetic samples.

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels (n_samples,)
        n_splits: Number of CV folds (default: 5)

    Returns:
        list of tuples: [(x_fold_1, y_fold_1), (x_fold_2, y_fold_2), ...]
        Each fold is upsampled to TARGET_OK_RATIO
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

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = []

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        # Combine ALL OK + current NOK fold
        x_combined = np.vstack([x_ok, x_nok[nok_idx]])
        y_combined = np.hstack([y_ok, y_nok[nok_idx]])

        # Upsample OK to TARGET_OK_RATIO
        n_ok = len(x_ok)
        n_nok = len(nok_idx)
        n_ok_target = int((TARGET_OK_RATIO * n_nok) / (1 - TARGET_OK_RATIO))

        if n_ok_target > n_ok:
            # Need to upsample OK
            smote = SMOTE(sampling_strategy={0: n_ok_target}, random_state=RANDOM_STATE)
            x_upsampled, y_upsampled = smote.fit_resample(x_combined, y_combined)
            print(
                f"Fold {fold_idx+1}: Upsampled {n_ok} → {n_ok_target} OK samples ({n_ok_target/n_ok:.1f}x)"
            )
        else:
            x_upsampled, y_upsampled = x_combined, y_combined
            print(f"Fold {fold_idx+1}: No upsampling needed")

        folds.append((x_upsampled, y_upsampled))

    return folds


def _report_cv_results(results):
    """
    Report aggregated cross-validation results.

    TODO: Implement comprehensive metrics:
    - Stage 1: Average precision, recall, F1
    - Stage 2: Fault-pure cluster statistics
    - Two-stage: Final metrics after cluster filtering

    Args:
        results: List of dicts containing per-fold results
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)

    n_folds = len(results)

    # Aggregate Stage 1 metrics across folds
    print("\nStage 1 Performance (averaged across folds):")
    # TODO: Calculate precision, recall, F1 per fold and average

    # Aggregate Stage 2 metrics across folds
    print("\nStage 2 Performance (averaged across folds):")
    # TODO: Calculate fault-pure cluster stats, filtering effectiveness

    print(f"\nCompleted {n_folds}-fold cross-validation")


def _run_stage1(x_values: NDArray, y_true: NDArray) -> tuple[NDArray, NDArray]:
    """
    Stage 1: Over-sensitive anomaly detection.

    Uses Isolation Forest with intentionally high contamination parameter
    to maximize recall. Accepts elevated false positive rate, which will
    be filtered in Stage 2.

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels for evaluation only (not used in training)

    Returns:
        tuple: (y_anomalies, anomaly_scores)
            - y_anomalies: Binary predictions (0=OK, 1=NOK)
            - anomaly_scores: Anomaly scores for each sample
    """
    print("\n" + "=" * 70)
    print("STAGE 1: ANOMALY DETECTION")
    print("=" * 70)

    # Train anomaly detector (unsupervised - does not use y_true)
    model = IForest(contamination=CONTAMINATION, random_state=RANDOM_STATE)
    model.fit(x_values)

    # Get predictions
    y_anomalies = model.predict(x_values)  # 0=OK, 1=NOK
    anomaly_scores = model.decision_scores_

    # Report results
    n_ok = (y_anomalies == 0).sum()
    n_nok = (y_anomalies == 1).sum()
    print(f"Detected: {n_ok} OK, {n_nok} NOK")

    # Show confusion matrix (evaluation only, not used for decisions)
    y_binary = (y_true > 0).astype(int)
    cm = confusion_matrix(y_binary, y_anomalies)
    print(f"\nConfusion Matrix:")
    print(f"             Predicted OK  Predicted NOK")
    print(f"Actual OK    {cm[0,0]:12d}  {cm[0,1]:13d}")
    print(f"Actual NOK   {cm[1,0]:12d}  {cm[1,1]:13d}")

    return y_anomalies, anomaly_scores


def _run_stage2(x_values, y_anomalies, y_true, label_mapping):
    """
    Stage 2: Cluster-based false positive filtering.

    Clusters flagged anomalies together with OK reference samples to:
    1. Identify "Fault-pure" clusters (high confidence real faults)
    2. Identify "OK-dominated" clusters (false positives that cluster with OK refs)
    3. Identify "FP-dominated" clusters (false positives forming distinct groups)

    Cluster Classification:
    - Fault-pure: <30% False Pos, contains real faults
    - OK-dominated: >50% OK Reference samples
    - FP-dominated: >70% False Positives
    - Mixed: None of the above

    Args:
        x_values: Feature matrix
        y_anomalies: Stage 1 predictions (0=OK, 1=NOK)
        y_true: Ground truth for analysis (not used for clustering)
        label_mapping: Fault class names (for future use)

    Returns:
        y_clusters: Cluster assignments for samples sent to Stage 2
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FAULT CLUSTERING")
    print("=" * 70)

    # 1. Prepare: Select NOK samples + OK reference
    nok_indices = np.where(y_anomalies == 1)[0]
    ok_indices = np.where(y_anomalies == 0)[0]

    # Sample OK reference (e.g., 1% of OK samples)
    n_ok_sample = int(len(ok_indices) * OK_REFERENCE_RATIO)
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

    # 2. Run clustering (unsupervised - does not use y_cluster_true)
    if USE_DTW:
        try:
            from sktime.clustering.k_means import TimeSeriesKMeans

            model = TimeSeriesKMeans(
                n_clusters=N_CLUSTERS, metric="dtw", random_state=RANDOM_STATE
            )
            print(f"Using TimeSeriesKMeans with DTW")
        except ImportError:
            print("Warning: sktime not available, using standard KMeans")
            model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    else:
        print(f"Using standard KMeans (Euclidean distance)")
        model = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)

    y_clusters = model.fit_predict(x_cluster)
    print(f"Found {len(np.unique(y_clusters))} clusters")

    # 3. Analyze cluster composition (post-hoc analysis using ground truth)
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

        n_real_faults = (y_non_ref > 0).sum()  # Ground truth says fault
        n_false_pos = (
            y_non_ref == 0
        ).sum()  # Ground truth says OK, but Stage 1 flagged

        # Calculate percentages
        ok_ref_pct = (n_ok_ref / n_total * 100) if n_total > 0 else 0
        fp_pct = (
            (n_false_pos / (n_total - n_ok_ref) * 100)
            if (n_total - n_ok_ref) > 0
            else 0
        )

        # Classify cluster type based on composition
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

    return y_clusters
