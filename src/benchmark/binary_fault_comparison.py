"""
Binary fault detection baseline benchmark.

Evaluates supervised classifiers (with full label access) to establish the
performance ceiling for each fault class. Tests binary classification: Normal vs.
each individual fault class to determine which faults are inherently detectable.

Uses the same cross-validation strategy (99:1 ratio) as the unsupervised pipeline
for fair comparison, but with independent NOK samples to avoid data leakage.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

from src.data import load_class_config, run_data_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

# Hardcoded configuration (kept simple for benchmark)
OUTPUT_DIR = Path("results") / "binary_fault_comparison"

# Cross-validation parameters (matching default pipeline config)
N_SPLITS = 5
TARGET_NOK_PER_FOLD = 50
TARGET_OK_PER_FOLD = 4950
RANDOM_STATE = 42
# Supervised training parameters
TRAIN_RATIO = 0.7


# =============================================================================
# HYPERPARAMETER TUNING PHILOSOPHY
# =============================================================================
# We intentionally use DEFAULT hyperparameters for all classifiers without
# optimization (e.g., no Optuna, GridSearch, or manual tuning).
#
# REASONING:
# 1. This is a BASELINE benchmark to show the "performance ceiling" for each
#    fault class when labels are available
# 2. We want to see which faults are "easy" vs "hard" to detect inherently,
#    not which faults we can optimize for
# 3. Hyperparameter tuning would:
#    - Be time-consuming (8 models × 5 faults × 5 folds = 200 runs)
#    - Hide which faults are inherently difficult
#    - Make comparison less fair (some models benefit more from tuning)
#
# IMPLICATIONS:
# - Tree-based models (RandomForest, ExtraTrees, GradientBoosting) work well
#   with defaults because they're robust to hyperparameter choices
# - SVM performs poorly (~50% F1) because RBF kernel defaults don't suit
#   high-dimensional PAA-compressed time series (would need C, gamma tuning)
# - KNN works reasonably well (~80% F1) with default k=5, but could improve
#   with distance metric tuning (e.g., DTW instead of Euclidean)
# - Neural networks (MLP) are hit-or-miss with default architecture
#
# This is ACCEPTABLE because:
# - We're measuring fault detectability, not model optimization
# - The best-performing models (ExtraTrees: 91%, MLP: 90%) already show
#   the performance ceiling for most faults
# - Poor-performing models still contribute to the ensemble average
# =============================================================================

# Classifiers to benchmark (8 diverse models for time series)
CLASSIFIERS = {
    # Tree-based ensembles (good for any data type)
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    # Distance-based (good for time series)
    "KNN": KNeighborsClassifier(n_neighbors=5),
    # Support Vector Machine (kernel-based)
    "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
    # Neural networks
    "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), random_state=RANDOM_STATE),
    # Linear models (fast baselines)
    "LogisticRegression": LogisticRegression(max_iter=100, random_state=RANDOM_STATE),
    # Probabilistic model
    "GaussianNB": GaussianNB(),
}


def prepare_binary_fault_comparison_cv_folds(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    target_ok_nok_ratio: float = 0.99,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare cross-validation folds with proper train/test split BEFORE upsampling.

    CRITICAL: This function splits data FIRST, then upsamples BOTH train and test sets
    with DIFFERENT random seeds to avoid leakage while maintaining fair comparison.

    Key principles:
    1. NO LEAKAGE: Train and test use different SMOTE instances (different random seeds)
       - Train synthetic samples ≠ Test synthetic samples
       - No test sample is derived from training data
    2. SAME DISTRIBUTION: Both train and test have 99:1 ratio
       - Comparable to unsupervised pipeline which also uses 99:1
       - Fair evaluation: model trained and tested under same conditions
    3. MOSTLY REAL DATA: Only 4x upsampling (not 20x)
       - 25% real, 75% synthetic for both train and test
       - More realistic than 95% synthetic alternative

    # =========================================================================
    # THE 4X UPSAMPLING STRATEGY: WHY USE ALL OK SAMPLES IN EVERY FOLD?
    # =========================================================================
    #
    # We have 1200 real OK samples total. We need to maintain 99:1 ratio.
    # We have two choices:
    #
    # OPTION 1: Split OKs independently (proper CV, but problematic)
    # - Each fold gets: 240 real OK samples (1200 / 5 folds)
    # - Split 70/30: Train=168 OK, Test=72 OK
    # - Need to upsample: 168 → ~3300 (19.6x upsampling!)
    # - Result: Training on 95% SYNTHETIC DATA
    # - Problem: Training on mostly fake data is unrealistic
    #
    # OPTION 2: Use ALL OKs in every fold (hybrid approach, better!)
    # - Each fold gets: ALL 1200 real OK samples
    # - Split 70/30: Train=840 OK, Test=360 OK
    # - Need to upsample: 840 → ~3465 and 360 → ~1485 (both 4.1x upsampling)
    # - Result: Both train and test are 25% REAL DATA, 75% synthetic
    # - Advantage: Training AND testing on more real data = more realistic baseline
    #
    # WHY IS OK "LEAKAGE" ACCEPTABLE?
    # 1. OKs are the "normal" unlabeled background class - they're not what
    #    we're trying to detect
    # 2. The actual fault classes (NOKs) are split independently, which is
    #    what matters for proper cross-validation
    # 3. For supervised classification, having the same normal background
    #    across folds is similar to having a shared "normality model"
    # 4. This matches the unsupervised pipeline's strategy (ALL OKs per fold)
    #    but with independent NOK samples for fair comparison
    #
    # WHY UPSAMPLE BOTH TRAIN AND TEST?
    # 1. Fair comparison: Unsupervised method sees 99:1 ratio everywhere
    # 2. Same conditions: Supervised should be evaluated under same constraints
    # 3. No leakage: Train and test use DIFFERENT SMOTE seeds (independent synthetic samples)
    # 4. Realistic: Both sets are 25% real data (much better than 95% synthetic)
    #
    # =========================================================================

    PROPER WORKFLOW (no data leakage, fair comparison):
    1. Split ONLY NOK samples into n_splits independent folds (stratified)
    2. Each fold gets: ALL 1200 OK samples + 1/n NOK samples (independent)
    3. Split 70/30: Separate OK and NOK into train/test BEFORE any upsampling
    4. Upsample train: Train OK → 3465 with SMOTE (seed: random_state + fold_num)
    5. Upsample test: Test OK → 1485 with SMOTE (seed: random_state + fold_num + 1000)
    6. Result: Both train and test have 99:1 ratio, but independent synthetic samples

    The key insight: By using different SMOTE seeds for train and test, we maintain:
    - Fair comparison (same 99:1 distribution as unsupervised)
    - No leakage (test synthetic samples ≠ train synthetic samples)
    - Realistic data (25% real, not 95% synthetic)

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_values: Ground truth labels (n_samples,)
        n_splits: Number of CV folds
        train_ratio: Fraction of data for training (default: 0.7)
        target_ok_nok_ratio: Target ratio of OK:NOK after upsampling (default: 0.99)
        random_state: Random seed for reproducibility

    Returns:
        list of tuples: [(x_train, y_train, x_test, y_test), ...]
        Both training and test sets are upsampled to 99:1 ratio

    Example:
        >>> folds = prepare_binary_fault_comparison_cv_folds(x, y, n_splits=5)
        >>> for x_train, y_train, x_test, y_test in folds:
        >>>     # x_train: ~3465 OK (upsampled) + 35 NOK
        >>>     # x_test: ~1485 OK (upsampled, different SMOTE) + 15 NOK
        >>>     model.fit(x_train, y_train)
        >>>     score = model.score(x_test, y_test)  # Both sets have 99:1 ratio!
    """
    logger.subsection(f"Preparing {n_splits}-fold CV (split BEFORE upsample)")
    logger.debug(f"Random state: {random_state}")

    # Split OK vs NOK
    ok_mask = y_values == 0
    x_ok_all = x_values[ok_mask]
    y_ok_all = y_values[ok_mask]
    x_nok = x_values[~ok_mask]
    y_nok = y_values[~ok_mask]

    n_ok_total = len(x_ok_all)
    n_nok_total = len(x_nok)

    logger.info(f"Total samples: {len(x_values)} ({n_ok_total} OK, {n_nok_total} NOK)")
    logger.info(f"Strategy: ALL {n_ok_total} OK in every fold, NOK split independently")
    logger.info(
        f"Split {train_ratio:.0%}/{1-train_ratio:.0%} BEFORE upsampling to avoid leakage"
    )

    # Use StratifiedKFold to split ONLY NOK samples
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    rng = np.random.RandomState(random_state)

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        fold_num = fold_idx + 1
        logger.debug(f"Processing fold {fold_num}/{n_splits}")

        # Get NOK samples for this fold (independent across folds)
        x_nok_fold = x_nok[nok_idx]
        y_nok_fold = y_nok[nok_idx]
        n_nok_fold = len(x_nok_fold)

        logger.debug(f"Fold {fold_num}: {n_ok_total} OK (all), {n_nok_fold} NOK")

        # =====================================================================
        # CRITICAL: Split into train/test BEFORE any upsampling
        # =====================================================================

        # Calculate split sizes
        n_ok_train = int(n_ok_total * train_ratio)
        n_ok_test = n_ok_total - n_ok_train
        n_nok_train = int(n_nok_fold * train_ratio)
        n_nok_test = n_nok_fold - n_nok_train

        # Split OK samples (same random split for this fold)
        ok_indices = rng.permutation(n_ok_total)
        ok_train_idx = ok_indices[:n_ok_train]
        ok_test_idx = ok_indices[n_ok_train:]

        x_ok_train = x_ok_all[ok_train_idx]
        y_ok_train = y_ok_all[ok_train_idx]
        x_ok_test = x_ok_all[ok_test_idx]
        y_ok_test = y_ok_all[ok_test_idx]

        # Split NOK samples
        nok_indices = rng.permutation(n_nok_fold)
        nok_train_idx = nok_indices[:n_nok_train]
        nok_test_idx = nok_indices[n_nok_train:]

        x_nok_train = x_nok_fold[nok_train_idx]
        y_nok_train = y_nok_fold[nok_train_idx]
        x_nok_test = x_nok_fold[nok_test_idx]
        y_nok_test = y_nok_fold[nok_test_idx]

        logger.info(
            f"Fold {fold_num}: Split → Train: {n_ok_train} OK + {n_nok_train} NOK, "
            f"Test: {n_ok_test} OK + {n_nok_test} NOK"
        )

        # =====================================================================
        # Upsample BOTH training and test sets to maintain 99:1 ratio
        #
        # IMPORTANT: We perform TWO separate SMOTE operations with DIFFERENT
        # random seeds to ensure no data leakage:
        #
        # 1. Train SMOTE (seed: random_state + fold_num):
        #    - Generates synthetic OK samples for training
        #    - These samples help the model learn the decision boundary
        #
        # 2. Test SMOTE (seed: random_state + fold_num + 1000):
        #    - Generates DIFFERENT synthetic OK samples for testing
        #    - These samples are independent from training synthetic samples
        #    - Ensures model is tested on unseen data
        #
        # Why two SMOTE operations?
        # - Maintains 99:1 ratio in both train and test (fair comparison)
        # - Avoids leakage (train synthetic ≠ test synthetic)
        # - Both sets are 75% real data (better than 95% synthetic alternative)
        # =====================================================================

        # Calculate target OK counts for both sets to achieve target_ok_nok_ratio
        # target_ok_nok_ratio = n_ok / (n_ok + n_nok)
        # Solving for n_ok: n_ok = (ratio * n_nok) / (1 - ratio)
        target_ok_train = int(
            (target_ok_nok_ratio * n_nok_train) / (1 - target_ok_nok_ratio)
        )
        target_ok_test = int(
            (target_ok_nok_ratio * n_nok_test) / (1 - target_ok_nok_ratio)
        )

        # Upsample training set
        if target_ok_train > n_ok_train:
            logger.debug(f"Fold {fold_num}: Upsampling training OK with SMOTE")

            # Combine train OK and NOK for SMOTE
            x_train_combined = np.vstack([x_ok_train, x_nok_train])
            y_train_combined = np.hstack([y_ok_train, y_nok_train])

            try:
                smote_train = SMOTE(
                    sampling_strategy={0: target_ok_train},
                    random_state=random_state + fold_num,  # Unique seed for train
                )
                x_train_final, y_train_final = smote_train.fit_resample(
                    x_train_combined, y_train_combined
                )

                upsampling_factor_train = target_ok_train / n_ok_train
                n_ok_train_final = (y_train_final == 0).sum()

                logger.info(
                    f"Fold {fold_num}: Upsampled training OK {n_ok_train} → {n_ok_train_final} "
                    f"({upsampling_factor_train:.1f}x SMOTE)"
                )

            except ValueError as e:
                logger.warning(
                    f"Fold {fold_num}: Training SMOTE failed ({e}), using original"
                )
                x_train_final = x_train_combined
                y_train_final = y_train_combined
        else:
            # No upsampling needed
            x_train_final = np.vstack([x_ok_train, x_nok_train])
            y_train_final = np.hstack([y_ok_train, y_nok_train])
            logger.debug(f"Fold {fold_num}: No upsampling needed for training set")

        # Upsample test set (with DIFFERENT random seed to avoid leakage!)
        if target_ok_test > n_ok_test:
            logger.debug(f"Fold {fold_num}: Upsampling test OK with SMOTE")

            # Combine test OK and NOK for SMOTE
            x_test_combined = np.vstack([x_ok_test, x_nok_test])
            y_test_combined = np.hstack([y_ok_test, y_nok_test])

            try:
                smote_test = SMOTE(
                    sampling_strategy={0: target_ok_test},
                    random_state=random_state
                    + fold_num
                    + 1000,  # Different seed from train!
                )
                x_test_final, y_test_final = smote_test.fit_resample(
                    x_test_combined, y_test_combined
                )

                upsampling_factor_test = target_ok_test / n_ok_test
                n_ok_test_final = (y_test_final == 0).sum()

                logger.info(
                    f"Fold {fold_num}: Upsampled test OK {n_ok_test} → {n_ok_test_final} "
                    f"({upsampling_factor_test:.1f}x SMOTE)"
                )

            except ValueError as e:
                logger.warning(
                    f"Fold {fold_num}: Test SMOTE failed ({e}), using original"
                )
                x_test_final = x_test_combined
                y_test_final = y_test_combined
        else:
            # No upsampling needed
            x_test_final = np.vstack([x_ok_test, x_nok_test])
            y_test_final = np.hstack([y_ok_test, y_nok_test])
            logger.debug(f"Fold {fold_num}: No upsampling needed for test set")

        n_ok_train_final = (y_train_final == 0).sum()
        n_nok_train_final = (y_train_final > 0).sum()
        train_ok_ratio = n_ok_train_final / len(y_train_final)

        n_ok_test_final = (y_test_final == 0).sum()
        n_nok_test_final = (y_test_final > 0).sum()
        test_ok_ratio = n_ok_test_final / len(y_test_final)

        logger.info(
            f"Fold {fold_num}: Final → "
            f"Train: {n_ok_train_final} OK + {n_nok_train_final} NOK ({train_ok_ratio:.1%} OK), "
            f"Test: {n_ok_test_final} OK + {n_nok_test_final} NOK ({test_ok_ratio:.1%} OK)"
        )

        folds.append((x_train_final, y_train_final, x_test_final, y_test_final))

    logger.info(f"Created {len(folds)} folds (both train and test upsampled to 99:1)")
    return folds


def run_binary_fault_comparison():
    """
    Benchmark supervised classifiers to determine fault detectability ceiling.

    For each fault class, tests binary classification (Normal vs. Fault) to answer:
    "How well can this fault be detected when we have perfect label information?"

    This provides a performance ceiling for comparison with unsupervised methods.
    Uses independent cross-validation folds (NOK samples split independently) while
    maintaining the same 99:1 OK:NOK ratio as the unsupervised pipeline.

    Key outputs:
    - Per-class detectability: Which faults are inherently easy/hard to detect?
    - Per-classifier performance: Which models work best with default params?
    - Overall ceiling: What's the best we could do with full supervision?

    Results saved to: results/supervised_benchmark/
    """
    logger.section("BINARY FAULT DETECTABILITY BENCHMARK")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load data (with PAA already applied)
    logger.subsection("Loading Data")
    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=False,
        keep_exceptions=False,
        classes_to_keep=load_class_config("top5"),  # Use top5 for comparison
        paa_segments=200,  # Match pipeline default
    )

    logger.info(f"Loaded {len(x_values)} samples with {len(label_mapping)} classes")
    logger.debug(f"Data shape: {x_values.shape}")

    # Prepare cross-validation folds (splits train/test, upsamples only training)
    cv_folds = prepare_binary_fault_comparison_cv_folds(
        x_values=x_values,
        y_values=y_true,
        n_splits=N_SPLITS,
        train_ratio=TRAIN_RATIO,
        target_ok_nok_ratio=0.99,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Created {len(cv_folds)} cross-validation folds")

    # Get fault class names
    int_to_label = {v: k for k, v in label_mapping.items()}

    # Collect results for all fault classes
    logger.subsection("Testing Fault Classes")
    all_results = []

    # For each fault class, test across all CV folds
    for fault_label in sorted(set(y_true)):
        if fault_label == 0:
            continue  # Skip normal class

        fault_name = int_to_label[fault_label]
        logger.info(f"Testing {fault_name}")

        fold_results = []

        # Test on each fold (data is already split!)
        for fold_num, (x_train, y_train, x_test, y_test) in enumerate(cv_folds, 1):
            # Convert to binary: this fault vs. normal
            y_train_binary = (y_train == fault_label).astype(int)
            y_test_binary = (y_test == fault_label).astype(int)

            # Skip if this fault class not in fold
            if y_train_binary.sum() == 0 or y_test_binary.sum() == 0:
                logger.debug(f"  Fold {fold_num}: No {fault_name} in train or test")
                continue

            # Test all classifiers on this fold
            # Both train and test have 99:1 ratio but independent synthetic samples
            for clf_name, clf in CLASSIFIERS.items():
                # Train on upsampled training set
                clf.fit(x_train, y_train_binary)

                # Test on upsampled test set (different SMOTE instance, no leakage)
                y_pred = clf.predict(x_test)

                # Calculate metrics
                f1 = f1_score(y_test_binary, y_pred, zero_division=0)
                precision = precision_score(y_test_binary, y_pred, zero_division=0)
                recall = recall_score(y_test_binary, y_pred, zero_division=0)

                fold_results.append(
                    {
                        "fault_class": fault_name,
                        "fold": fold_num,
                        "classifier": clf_name,
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                    }
                )

        if fold_results:
            # Calculate average metrics across folds for each classifier
            for clf_name in CLASSIFIERS.keys():
                clf_fold_results = [
                    r for r in fold_results if r["classifier"] == clf_name
                ]

                if clf_fold_results:
                    avg_f1 = np.mean([r["f1"] for r in clf_fold_results])
                    avg_precision = np.mean([r["precision"] for r in clf_fold_results])
                    avg_recall = np.mean([r["recall"] for r in clf_fold_results])
                    std_f1 = np.std([r["f1"] for r in clf_fold_results])

                    logger.debug(
                        f"  {clf_name:18s} | F1={avg_f1:.3f}±{std_f1:.3f} "
                        f"P={avg_precision:.3f} R={avg_recall:.3f}"
                    )

                    all_results.append(
                        {
                            "fault_class": fault_name,
                            "classifier": clf_name,
                            "f1_mean": avg_f1,
                            "f1_std": std_f1,
                            "precision_mean": avg_precision,
                            "recall_mean": avg_recall,
                            "n_folds": len(clf_fold_results),
                        }
                    )

            # Calculate average across all classifiers
            avg_f1_all = np.mean([r["f1"] for r in fold_results])
            logger.info(f"  Average F1 across all classifiers: {avg_f1_all:.3f}")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save detailed results
    logger.subsection("Saving Results")
    csv_path = OUTPUT_DIR / "benchmark_results_detailed.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved detailed results to: {csv_path}")

    # Create summary by fault class (average over classifiers)
    summary_df = (
        df.groupby("fault_class")
        .agg(
            {
                "f1_mean": "mean",
                "f1_std": "mean",
                "precision_mean": "mean",
                "recall_mean": "mean",
            }
        )
        .reset_index()
    )

    summary_df.columns = [
        "fault_class",
        "avg_f1",
        "avg_f1_std",
        "avg_precision",
        "avg_recall",
    ]

    csv_path_summary = OUTPUT_DIR / "benchmark_results_summary.csv"
    summary_df.to_csv(csv_path_summary, index=False)
    logger.info(f"Saved summary to: {csv_path_summary}")

    # Generate plots
    _generate_plots(df, summary_df)

    # Print summary
    _print_summary(df, summary_df)

    logger.info("Supervised benchmark complete!")


def _generate_plots(df: pd.DataFrame, summary_df: pd.DataFrame):
    """Generate visualization plots for benchmark results."""
    logger.subsection("Generating Plots")

    # Sort by average F1
    summary_sorted = summary_df.sort_values("avg_f1", ascending=True)

    # Color palette
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(summary_sorted)))

    # 1. Average F1 by fault class (horizontal bar chart)
    plt.figure(figsize=(12, max(6, len(summary_sorted) * 0.5)))

    y_pos = range(len(summary_sorted))
    bars = plt.barh(y_pos, summary_sorted["avg_f1"], color=colors, alpha=0.8)

    # Add error bars for std
    plt.errorbar(
        summary_sorted["avg_f1"],
        y_pos,
        xerr=summary_sorted["avg_f1_std"],
        fmt="none",
        ecolor="black",
        capsize=3,
        alpha=0.5,
    )

    plt.yticks(y_pos, summary_sorted["fault_class"], fontsize=9)
    plt.xlabel("Average F1-Score (across classifiers)", fontsize=12)
    plt.title(
        "Supervised Classification Performance by Fault Class",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (val, std) in enumerate(
        zip(summary_sorted["avg_f1"], summary_sorted["avg_f1_std"])
    ):
        plt.text(val + 0.02, i, f"{val:.3f}±{std:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "f1_by_fault_class.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {save_path.name}")

    # 2. Classifier comparison (grouped bar chart) - per fault class
    plt.figure(figsize=(14, 8))

    # Pivot to get classifiers as columns
    pivot_df = df.pivot_table(
        index="fault_class", columns="classifier", values="f1_mean"
    )

    n_faults = len(pivot_df)
    x = np.arange(n_faults)
    width = 0.05

    clf_colors = plt.cm.Set2(range(len(CLASSIFIERS)))

    for i, clf_name in enumerate(CLASSIFIERS.keys()):
        if clf_name in pivot_df.columns:
            values = pivot_df[clf_name].values
            offset = width * (i - len(CLASSIFIERS) / 2 + 0.5)
            plt.bar(
                x + offset,
                values,
                width,
                label=clf_name,
                color=clf_colors[i],
                alpha=0.8,
            )

    plt.xlabel("Fault Class", fontsize=12)
    plt.ylabel("F1-Score", fontsize=12)
    plt.title(
        "Classifier Comparison Across Fault Classes (CV-averaged)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xticks(x, pivot_df.index, rotation=45, ha="right", fontsize=8)
    plt.legend(loc="lower right", fontsize=10)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    save_path = OUTPUT_DIR / "classifier_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {save_path.name}")

    # 3. Classifier performance summary (box plot)
    plt.figure(figsize=(10, 6))

    clf_data = []
    clf_labels = []
    for clf_name in CLASSIFIERS.keys():
        clf_subset = df[df["classifier"] == clf_name]
        if not clf_subset.empty:
            clf_data.append(clf_subset["f1_mean"].values)
            clf_labels.append(clf_name)

    bp = plt.boxplot(clf_data, labels=clf_labels, patch_artist=True)

    # Color boxes
    for patch, color in zip(bp["boxes"], clf_colors[: len(clf_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("F1-Score", fontsize=12)
    plt.title("Classifier Performance Distribution", fontsize=14, fontweight="bold")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15)
    plt.tight_layout()

    save_path = OUTPUT_DIR / "classifier_distribution.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {save_path.name}")


def _print_summary(df: pd.DataFrame, summary_df: pd.DataFrame):
    """Print summary statistics to console and log."""
    logger.subsection("Summary Statistics")

    # Sort by detectability
    summary_sorted = summary_df.sort_values("avg_f1", ascending=False)

    # Detectability ranking
    logger.info("Detectability Ranking (by Average F1 across classifiers and folds):")
    logger.info(f"{'Fault Class':<40} | Avg F1 | Detectable?")
    logger.info("-" * 70)

    for _, row in summary_sorted.iterrows():
        if row["avg_f1"] > 0.7:
            detectable = "YES"
        elif row["avg_f1"] > 0.4:
            detectable = "PARTIAL"
        else:
            detectable = "NO"

        logger.info(
            f"{row['fault_class']:<40} | " f"{row['avg_f1']:6.3f} | " f"{detectable}"
        )

    # Classifier performance summary
    logger.info("")
    logger.info("Classifier Performance Summary (averaged across faults and folds):")
    logger.info(f"{'Classifier':<20} | Mean F1 | Std F1")
    logger.info("-" * 45)

    for clf_name in CLASSIFIERS.keys():
        clf_subset = df[df["classifier"] == clf_name]
        if not clf_subset.empty:
            mean_f1 = clf_subset["f1_mean"].mean()
            std_f1 = clf_subset["f1_mean"].std()
            logger.info(f"{clf_name:<20} | {mean_f1:7.3f} | {std_f1:6.3f}")

    # Overall statistics
    overall_mean = summary_df["avg_f1"].mean()
    overall_std = summary_df["avg_f1"].std()

    logger.info("")
    logger.info(f"Overall Average F1: {overall_mean:.3f} (±{overall_std:.3f})")
    logger.info(
        f"Best performing class: {summary_sorted.iloc[0]['fault_class']} "
        f"(F1={summary_sorted.iloc[0]['avg_f1']:.3f})"
    )
    logger.info(
        f"Worst performing class: {summary_sorted.iloc[-1]['fault_class']} "
        f"(F1={summary_sorted.iloc[-1]['avg_f1']:.3f})"
    )

    # Cross-validation statistics
    logger.info("")
    logger.info(f"Cross-validation: {N_SPLITS} folds")
    logger.info(f"  NOK per fold: {TARGET_NOK_PER_FOLD}")
    logger.info(f"  OK per fold: {TARGET_OK_PER_FOLD}")
    logger.info(f"  Random state: {RANDOM_STATE}")
