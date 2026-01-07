"""
Supervised classification benchmark for two-stage pipeline comparison.

Establishes supervised performance ceiling with two variants:
1. Binary classification: Normal (0) vs. All Faults (1)
2. Multi-class classification: Normal (0) + 5 fault classes (1-5)

Uses same cross-validation strategy as unsupervised pipeline:
- 5-fold stratified CV
- 99:1 OK:NOK ratio (50 NOK, 4950 OK per fold)
- SMOTE upsampling with different seeds for train/test (no leakage)
- All OK samples in every fold (matching unsupervised pipeline)

This provides a fair comparison baseline showing what's achievable with
full label information under the same data constraints.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.data import load_class_config, run_data_pipeline
from src.utils import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directory
OUTPUT_DIR = Path("results") / "supervised_classification"

# Cross-validation parameters (matching pipeline defaults)
N_SPLITS = 5
TARGET_NOK_PER_FOLD = 50
TARGET_OK_PER_FOLD = 4950  # Gives 99:1 ratio
TRAIN_RATIO = 0.7
RANDOM_STATE = 42

# Classifiers (same as binary_fault_comparison but with multi-class support)
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE
    ),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100, 50), max_iter=1000, random_state=RANDOM_STATE
    ),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "GaussianNB": GaussianNB(),
}


# =============================================================================
# CROSS-VALIDATION PREPARATION
# =============================================================================


def prepare_multiclass_cv_folds(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    target_ok_per_fold: int = 4950,
    target_nok_per_fold: int = 50,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Prepare stratified cross-validation folds for multi-class classification.

    Strategy matches unsupervised pipeline:
    - ALL OK samples in every fold (shared background)
    - NOK samples split independently across folds (stratified by class)
    - Both train and test upsampled to 99:1 ratio with different SMOTE seeds
    - NOK downsampled to target_nok_per_fold for equal fold sizes

    Key differences from binary_fault_comparison:
    - Preserves all fault class labels (not collapsed to 1)
    - SMOTE handles multi-class upsampling automatically
    - Stratification ensures balanced fault class distribution

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_values: Ground truth labels (n_samples,) - multi-class
        n_splits: Number of CV folds
        train_ratio: Fraction of data for training (default: 0.7)
        target_ok_per_fold: Target OK count after upsampling (default: 4950)
        target_nok_per_fold: Target NOK count per fold (default: 50)
        random_state: Random seed for reproducibility

    Returns:
        list of tuples: [(x_train, y_train, x_test, y_test), ...]
        Both train and test have 99:1 ratio with independent SMOTE
    """
    logger.subsection(f"Preparing {n_splits}-fold Multi-class CV")
    logger.debug(f"Random state: {random_state}")

    # Split OK vs NOK
    ok_mask = y_values == 0
    x_ok_all = x_values[ok_mask]
    y_ok_all = y_values[ok_mask]
    x_nok = x_values[~ok_mask]
    y_nok = y_values[~ok_mask]

    n_ok_total = len(x_ok_all)
    n_nok_total = len(x_nok)

    logger.info(f"Total samples: {len(y_values)} ({n_ok_total} OK, {n_nok_total} NOK)")
    logger.info(
        f"Fault class distribution: {dict(zip(*np.unique(y_nok, return_counts=True)))}"
    )
    logger.info(
        f"Strategy: ALL {n_ok_total} OK per fold, NOK split with stratification"
    )

    # Use StratifiedKFold to split NOK samples (maintains class balance)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    rng = np.random.RandomState(random_state)

    for fold_idx, (_, nok_idx) in enumerate(skf.split(x_nok, y_nok)):
        fold_num = fold_idx + 1
        logger.debug(f"Processing fold {fold_num}/{n_splits}")

        # Get NOK samples for this fold
        x_nok_fold = x_nok[nok_idx]
        y_nok_fold = y_nok[nok_idx]

        # Downsample NOK to target size if needed (simple random sampling)
        if target_nok_per_fold and len(nok_idx) > target_nok_per_fold:
            sample_idx = rng.choice(
                len(nok_idx), size=target_nok_per_fold, replace=False
            )
            x_nok_fold = x_nok_fold[sample_idx]
            y_nok_fold = y_nok_fold[sample_idx]
            logger.info(
                f"Fold {fold_num}: Downsampled NOK {len(nok_idx)} → {target_nok_per_fold}"
            )

        n_nok_fold = len(y_nok_fold)
        nok_class_dist = dict(zip(*np.unique(y_nok_fold, return_counts=True)))
        logger.debug(f"Fold {fold_num}: NOK class distribution: {nok_class_dist}")

        # Split OK samples (70/30 train/test)
        n_ok_train = int(n_ok_total * train_ratio)
        n_ok_test = n_ok_total - n_ok_train

        ok_indices = rng.permutation(n_ok_total)
        ok_train_idx = ok_indices[:n_ok_train]
        ok_test_idx = ok_indices[n_ok_train:]

        x_ok_train = x_ok_all[ok_train_idx]
        y_ok_train = y_ok_all[ok_train_idx]
        x_ok_test = x_ok_all[ok_test_idx]
        y_ok_test = y_ok_all[ok_test_idx]

        # Split NOK samples (70/30 train/test)
        n_nok_train = int(n_nok_fold * train_ratio)
        n_nok_test = n_nok_fold - n_nok_train

        # Check if we have enough samples per class for stratified split
        nok_class_counts = pd.Series(y_nok_fold).value_counts()
        min_class_count = nok_class_counts.min()

        # For stratified split to work:
        # - Smallest split needs at least 1 sample per class
        # - With 70/30 split, test gets 30%, so need: min_class_count * 0.3 >= 1
        # - This means: min_class_count >= 4 (4 * 0.3 = 1.2 ≈ 1)
        can_stratify = min_class_count >= 4

        from sklearn.model_selection import train_test_split

        if can_stratify:
            try:
                x_nok_train, x_nok_test, y_nok_train, y_nok_test = train_test_split(
                    x_nok_fold,
                    y_nok_fold,
                    train_size=n_nok_train,
                    random_state=random_state + fold_num,
                    stratify=y_nok_fold,
                )
                logger.info(
                    f"Fold {fold_num}: Split NOK stratified → "
                    f"Train: {n_nok_train} NOK, Test: {n_nok_test} NOK"
                )
            except ValueError as e:
                # Fallback to random if stratification fails
                logger.debug(
                    f"Fold {fold_num}: Stratified split failed ({e}), using random"
                )
                can_stratify = False

        if not can_stratify:
            # Random split (no stratification)
            nok_indices = rng.permutation(n_nok_fold)
            nok_train_idx = nok_indices[:n_nok_train]
            nok_test_idx = nok_indices[n_nok_train:]

            x_nok_train = x_nok_fold[nok_train_idx]
            y_nok_train = y_nok_fold[nok_train_idx]
            x_nok_test = x_nok_fold[nok_test_idx]
            y_nok_test = y_nok_fold[nok_test_idx]

            logger.info(
                f"Fold {fold_num}: Split NOK randomly (insufficient samples for stratified) → "
                f"Train: {n_nok_train} NOK, Test: {n_nok_test} NOK"
            )

        logger.info(
            f"Fold {fold_num}: Final split → "
            f"Train: {n_ok_train} OK + {n_nok_train} NOK, "
            f"Test: {n_ok_test} OK + {n_nok_test} NOK"
        )

        # Calculate target OK counts for 99:1 ratio
        # Ratio formula: n_ok / (n_ok + n_nok) = 0.99
        # Solving: n_ok = (0.99 * n_nok) / 0.01
        # TODO: Update to use the parameter 'target_ok_per_fold' instead
        target_ok_train = int((0.99 * n_nok_train) / 0.01)
        target_ok_test = int((0.99 * n_nok_test) / 0.01)

        # Combine for SMOTE
        x_train_combined = np.vstack([x_ok_train, x_nok_train])
        y_train_combined = np.hstack([y_ok_train, y_nok_train])

        x_test_combined = np.vstack([x_ok_test, x_nok_test])
        y_test_combined = np.hstack([y_ok_test, y_nok_test])

        # Upsample training set (SMOTE handles multi-class automatically)
        if target_ok_train > n_ok_train:
            logger.debug(f"Fold {fold_num}: Upsampling training set with SMOTE")

            try:
                smote_train = SMOTE(
                    sampling_strategy={0: target_ok_train},
                    random_state=random_state + fold_num,
                )
                x_train_final, y_train_final = smote_train.fit_resample(
                    x_train_combined, y_train_combined
                )

                n_ok_final = (y_train_final == 0).sum()
                logger.info(
                    f"Fold {fold_num}: Upsampled train OK {n_ok_train} → {n_ok_final} "
                    f"({n_ok_final/n_ok_train:.1f}x)"
                )

            except ValueError as e:
                logger.warning(
                    f"Fold {fold_num}: Train SMOTE failed ({e}), using original"
                )
                x_train_final = x_train_combined
                y_train_final = y_train_combined
        else:
            x_train_final = x_train_combined
            y_train_final = y_train_combined

        # Upsample test set (different SMOTE seed!)
        if target_ok_test > n_ok_test:
            logger.debug(f"Fold {fold_num}: Upsampling test set with SMOTE")

            try:
                smote_test = SMOTE(
                    sampling_strategy={0: target_ok_test},
                    random_state=random_state + fold_num + 1000,  # Different seed!
                )
                x_test_final, y_test_final = smote_test.fit_resample(
                    x_test_combined, y_test_combined
                )

                n_ok_final = (y_test_final == 0).sum()
                logger.info(
                    f"Fold {fold_num}: Upsampled test OK {n_ok_test} → {n_ok_final} "
                    f"({n_ok_final/n_ok_test:.1f}x)"
                )

            except ValueError as e:
                logger.warning(
                    f"Fold {fold_num}: Test SMOTE failed ({e}), using original"
                )
                x_test_final = x_test_combined
                y_test_final = y_test_combined
        else:
            x_test_final = x_test_combined
            y_test_final = y_test_combined

        # Log final fold statistics
        train_ok_ratio = (y_train_final == 0).sum() / len(y_train_final)
        test_ok_ratio = (y_test_final == 0).sum() / len(y_test_final)
        train_class_dist = dict(zip(*np.unique(y_train_final, return_counts=True)))
        test_class_dist = dict(zip(*np.unique(y_test_final, return_counts=True)))

        logger.info(
            f"Fold {fold_num}: Final → "
            f"Train: {len(y_train_final)} samples ({train_ok_ratio:.1%} OK), "
            f"Test: {len(y_test_final)} samples ({test_ok_ratio:.1%} OK)"
        )
        logger.debug(f"Fold {fold_num}: Train class dist: {train_class_dist}")
        logger.debug(f"Fold {fold_num}: Test class dist: {test_class_dist}")

        folds.append((x_train_final, y_train_final, x_test_final, y_test_final))

    logger.info(
        f"Created {len(folds)} folds with 99:1 ratio (train and test upsampled)"
    )
    return folds


# =============================================================================
# BENCHMARK EXECUTION
# =============================================================================


def run_supervised_classification_benchmark():
    """
    Main benchmark comparing binary and multi-class supervised classification.

    Runs two experiments:
    1. Binary: Normal (0) vs. All Faults (1)
    2. Multi-class: Normal (0) + 5 individual fault classes (1-5)

    Both use same 5-fold CV strategy with 99:1 ratio as unsupervised pipeline.

    Results:
    - Per-classifier performance for binary and multi-class
    - Confusion matrices for multi-class classification
    - Comparison plots showing supervised performance ceiling
    """
    logger.section("SUPERVISED CLASSIFICATION BENCHMARK")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Load data
    logger.subsection("Loading Data")
    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=False,
        keep_exceptions=False,
        classes_to_keep=load_class_config("top5"),
        paa_segments=200,
    )

    logger.info(f"Loaded {len(x_values)} samples with {len(label_mapping)} classes")
    logger.debug(f"Data shape: {x_values.shape}")
    logger.debug(f"Label mapping: {label_mapping}")

    # Reverse mapping for interpretability
    int_to_label = {v: k for k, v in label_mapping.items()}

    # Prepare folds (multi-class preserves all labels)
    logger.subsection("Preparing Cross-Validation Folds")
    cv_folds = prepare_multiclass_cv_folds(
        x_values=x_values,
        y_values=y_true,
        n_splits=N_SPLITS,
        train_ratio=TRAIN_RATIO,
        target_ok_per_fold=TARGET_OK_PER_FOLD,
        target_nok_per_fold=TARGET_NOK_PER_FOLD,
        random_state=RANDOM_STATE,
    )

    # ==========================================================================
    # EXPERIMENT 1: Binary Classification (OK vs NOK)
    # ==========================================================================

    logger.subsection("Experiment 1: Binary Classification (OK vs NOK)")
    binary_results = _run_binary_classification(cv_folds, int_to_label)

    # ==========================================================================
    # EXPERIMENT 2: Multi-class Classification (6 classes)
    # ==========================================================================

    logger.subsection("Experiment 2: Multi-class Classification (6 classes)")
    multiclass_results = _run_multiclass_classification(cv_folds, int_to_label)

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================

    logger.subsection("Saving Results")

    # Save binary results
    binary_df = pd.DataFrame(binary_results)
    binary_csv = OUTPUT_DIR / "binary_classification_results.csv"
    binary_df.to_csv(binary_csv, index=False)
    logger.info(f"Saved binary results: {binary_csv}")

    # Save multi-class results
    multiclass_df = pd.DataFrame(multiclass_results)
    multiclass_csv = OUTPUT_DIR / "multiclass_classification_results.csv"
    multiclass_df.to_csv(multiclass_csv, index=False)
    logger.info(f"Saved multi-class results: {multiclass_csv}")

    # ==========================================================================
    # GENERATE PLOTS
    # ==========================================================================

    logger.subsection("Generating Comparison Plots")
    _generate_comparison_plots(binary_df, multiclass_df)

    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================

    _print_summary(binary_df, multiclass_df, int_to_label)

    logger.info("Supervised classification benchmark complete!")


def _run_binary_classification(cv_folds: list, int_to_label: dict) -> list[dict]:
    """
    Run binary classification: Normal (0) vs. All Faults (1).

    Args:
        cv_folds: List of (x_train, y_train, x_test, y_test) tuples
        int_to_label: Mapping from integer labels to class names

    Returns:
        list of result dictionaries with per-fold, per-classifier metrics
    """
    logger.info("Testing binary classification: OK (0) vs. NOK (1)")

    results = []

    for fold_num, (x_train, y_train, x_test, y_test) in enumerate(cv_folds, 1):
        logger.debug(f"Processing fold {fold_num}/{len(cv_folds)}")

        # Convert to binary: 0 = OK, 1 = any fault
        y_train_binary = (y_train > 0).astype(int)
        y_test_binary = (y_test > 0).astype(int)

        n_train_ok = (y_train_binary == 0).sum()
        n_train_nok = (y_train_binary == 1).sum()
        n_test_ok = (y_test_binary == 0).sum()
        n_test_nok = (y_test_binary == 1).sum()

        logger.debug(
            f"Fold {fold_num}: Train={n_train_ok} OK + {n_train_nok} NOK, "
            f"Test={n_test_ok} OK + {n_test_nok} NOK"
        )

        # Test each classifier
        for clf_name, clf in CLASSIFIERS.items():
            logger.debug(f"  Testing {clf_name}...")

            # Train
            clf.fit(x_train, y_train_binary)

            # Predict
            y_pred = clf.predict(x_test)

            # Calculate metrics
            precision = precision_score(y_test_binary, y_pred, zero_division=0)
            recall = recall_score(y_test_binary, y_pred, zero_division=0)
            f1 = f1_score(y_test_binary, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test_binary, y_pred)

            # Store results
            results.append(
                {
                    "fold": fold_num,
                    "classifier": clf_name,
                    "task": "binary",
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "accuracy": accuracy,
                }
            )

            logger.debug(
                f"    {clf_name}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}"
            )

    return results


def _run_multiclass_classification(cv_folds: list, int_to_label: dict) -> list[dict]:
    """
    Run multi-class classification: 6 classes (OK + 5 faults).

    Args:
        cv_folds: List of (x_train, y_train, x_test, y_test) tuples
        int_to_label: Mapping from integer labels to class names

    Returns:
        list of result dictionaries with per-fold, per-classifier metrics
    """
    logger.info("Testing multi-class classification: 6 classes (OK + 5 faults)")

    results = []

    for fold_num, (x_train, y_train, x_test, y_test) in enumerate(cv_folds, 1):
        logger.debug(f"Processing fold {fold_num}/{len(cv_folds)}")

        # Class distribution
        train_dist = dict(zip(*np.unique(y_train, return_counts=True)))
        test_dist = dict(zip(*np.unique(y_test, return_counts=True)))

        logger.debug(f"Fold {fold_num}: Train classes: {train_dist}")
        logger.debug(f"Fold {fold_num}: Test classes: {test_dist}")

        # Test each classifier
        for clf_name, clf in CLASSIFIERS.items():
            logger.debug(f"  Testing {clf_name}...")

            # Train
            clf.fit(x_train, y_train)

            # Predict
            y_pred = clf.predict(x_test)

            # Calculate metrics (macro averaged for multi-class)
            precision = precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
            recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)

            # Also calculate per-class metrics
            precision_per_class = precision_score(
                y_test, y_pred, average=None, zero_division=0
            )
            recall_per_class = recall_score(
                y_test, y_pred, average=None, zero_division=0
            )
            f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

            # Store overall results
            results.append(
                {
                    "fold": fold_num,
                    "classifier": clf_name,
                    "task": "multiclass",
                    "precision_macro": precision,
                    "recall_macro": recall,
                    "f1_macro": f1,
                    "accuracy": accuracy,
                    # Per-class metrics as lists
                    "precision_per_class": precision_per_class.tolist(),
                    "recall_per_class": recall_per_class.tolist(),
                    "f1_per_class": f1_per_class.tolist(),
                }
            )

            logger.debug(
                f"    {clf_name}: F1={f1:.3f}, P={precision:.3f}, R={recall:.3f}, Acc={accuracy:.3f}"
            )

    return results


def _generate_comparison_plots(binary_df: pd.DataFrame, multiclass_df: pd.DataFrame):
    """Generate comparison plots for binary and multi-class results."""

    # 1. Binary vs Multi-class F1 comparison
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    # Binary F1 by classifier
    binary_summary = (
        binary_df.groupby("classifier").agg({"f1": ["mean", "std"]}).reset_index()
    )
    binary_summary.columns = ["classifier", "f1_mean", "f1_std"]
    binary_summary = binary_summary.sort_values("f1_mean", ascending=True)

    y_pos = range(len(binary_summary))
    axes[0].barh(
        y_pos,
        binary_summary["f1_mean"],
        xerr=binary_summary["f1_std"],
        color="steelblue",
        alpha=0.8,
        capsize=3,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(binary_summary["classifier"], fontsize=10)
    axes[0].set_xlabel("F1-Score", fontsize=12)
    axes[0].set_title(
        "Binary Classification (OK vs NOK)", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (mean, std) in enumerate(
        zip(binary_summary["f1_mean"], binary_summary["f1_std"])
    ):
        axes[0].text(mean + 0.02, i, f"{mean:.3f}±{std:.3f}", va="center", fontsize=9)

    # Multi-class F1 by classifier
    multiclass_summary = (
        multiclass_df.groupby("classifier")
        .agg({"f1_macro": ["mean", "std"]})
        .reset_index()
    )
    multiclass_summary.columns = ["classifier", "f1_mean", "f1_std"]
    multiclass_summary = multiclass_summary.sort_values("f1_mean", ascending=True)

    y_pos = range(len(multiclass_summary))
    axes[1].barh(
        y_pos,
        multiclass_summary["f1_mean"],
        xerr=multiclass_summary["f1_std"],
        color="coral",
        alpha=0.8,
        capsize=3,
    )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(multiclass_summary["classifier"], fontsize=10)
    axes[1].set_xlabel("F1-Score (Macro Avg.)", fontsize=12)
    axes[1].set_title(
        "Multi-class Classification (6 classes)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, (mean, std) in enumerate(
        zip(multiclass_summary["f1_mean"], multiclass_summary["f1_std"])
    ):
        axes[1].text(mean + 0.02, i, f"{mean:.3f}±{std:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "binary_vs_multiclass_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {save_path.name}")

    # 2. Task difficulty comparison (grouped bar chart)
    fig, ax = plt.subplots(figsize=(9, 5.0))
    classifiers = binary_summary["classifier"].tolist()
    x = np.arange(len(classifiers))
    width = 0.35
    binary_means = [
        binary_summary[binary_summary["classifier"] == clf]["f1_mean"].values[0]
        for clf in classifiers
    ]
    multiclass_means = [
        multiclass_summary[multiclass_summary["classifier"] == clf]["f1_mean"].values[0]
        for clf in classifiers
    ]

    # Get standard deviations
    binary_stds = [
        binary_summary[binary_summary["classifier"] == clf]["f1_std"].values[0]
        for clf in classifiers
    ]
    multiclass_stds = [
        multiclass_summary[multiclass_summary["classifier"] == clf]["f1_std"].values[0]
        for clf in classifiers
    ]

    # Create bars with error bars
    bars1 = ax.bar(
        x - width / 2,
        binary_means,
        width,
        yerr=binary_stds,
        capsize=5,
        label="Binary classification (1200 OK : 247 NOK)",
        color="teal",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        multiclass_means,
        width,
        yerr=multiclass_stds,
        capsize=5,
        label="Multi-class classification (1200 OK : five classes of NOK)",
        color="mediumpurple",
        alpha=0.8,
    )

    # Add percentage labels inside bars at 0.05 height
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.2,
            f"{height*100:.1f}%",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.2,
            f"{height*100:.1f}%",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    ax.set_xlabel(
        "Supervised classification models (sklearn implementation with default configuration)",
        fontsize=10,
    )
    ax.set_ylabel("F1-Score (macro avg.)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha="right", fontsize=10)
    ax.legend(fontsize=10, loc="lower left")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    save_path = OUTPUT_DIR / "task_difficulty_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved plot: {save_path.name}")


def _print_summary(
    binary_df: pd.DataFrame, multiclass_df: pd.DataFrame, int_to_label: dict
):
    """Print comprehensive summary of benchmark results."""

    logger.subsection("Benchmark Summary")

    # Binary classification summary
    logger.info("=" * 80)
    logger.info("BINARY CLASSIFICATION (OK vs NOK)")
    logger.info("=" * 80)

    binary_summary = (
        binary_df.groupby("classifier")
        .agg(
            {
                "precision": ["mean", "std"],
                "recall": ["mean", "std"],
                "f1": ["mean", "std"],
                "accuracy": ["mean", "std"],
            }
        )
        .round(3)
    )

    for clf_name in CLASSIFIERS.keys():
        if clf_name in binary_summary.index:
            row = binary_summary.loc[clf_name]
            logger.info(
                f"{clf_name:20s} | F1: {row[('f1', 'mean')]:.3f}±{row[('f1', 'std')]:.3f} | "
                f"P: {row[('precision', 'mean')]:.3f}±{row[('precision', 'std')]:.3f} | "
                f"R: {row[('recall', 'mean')]:.3f}±{row[('recall', 'std')]:.3f}"
            )

    binary_avg_f1 = binary_df.groupby("classifier")["f1"].mean().mean()
    logger.info(f"\nOverall Average F1: {binary_avg_f1:.3f}")

    # Multi-class classification summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("MULTI-CLASS CLASSIFICATION (6 classes)")
    logger.info("=" * 80)

    multiclass_summary = (
        multiclass_df.groupby("classifier")
        .agg(
            {
                "precision_macro": ["mean", "std"],
                "recall_macro": ["mean", "std"],
                "f1_macro": ["mean", "std"],
                "accuracy": ["mean", "std"],
            }
        )
        .round(3)
    )

    for clf_name in CLASSIFIERS.keys():
        if clf_name in multiclass_summary.index:
            row = multiclass_summary.loc[clf_name]
            logger.info(
                f"{clf_name:20s} | F1: {row[('f1_macro', 'mean')]:.3f}±{row[('f1_macro', 'std')]:.3f} | "
                f"P: {row[('precision_macro', 'mean')]:.3f}±{row[('precision_macro', 'std')]:.3f} | "
                f"R: {row[('recall_macro', 'mean')]:.3f}±{row[('recall_macro', 'std')]:.3f}"
            )

    multiclass_avg_f1 = multiclass_df.groupby("classifier")["f1_macro"].mean().mean()
    logger.info(f"\nOverall Average F1 (Macro Avg.): {multiclass_avg_f1:.3f}")

    # Task difficulty comparison
    logger.info("")
    logger.info("=" * 80)
    logger.info("TASK DIFFICULTY ANALYSIS")
    logger.info("=" * 80)

    avg_binary = binary_df["f1"].mean()
    avg_multiclass = multiclass_df["f1_macro"].mean()
    difficulty_ratio = (avg_multiclass / avg_binary) * 100

    logger.info(f"Average Binary F1:       {avg_binary:.3f}")
    logger.info(f"Average Multi-class F1:  {avg_multiclass:.3f}")
    logger.info(
        f"Difficulty ratio:        {difficulty_ratio:.1f}% "
        f"(multi-class retains {difficulty_ratio:.1f}% of binary performance)"
    )

    # Best performing classifiers
    logger.info("")
    logger.info("Best Performers:")
    best_binary_clf = binary_df.groupby("classifier")["f1"].mean().idxmax()
    best_binary_f1 = binary_df.groupby("classifier")["f1"].mean().max()
    best_multiclass_clf = (
        multiclass_df.groupby("classifier")["f1_macro"].mean().idxmax()
    )
    best_multiclass_f1 = multiclass_df.groupby("classifier")["f1_macro"].mean().max()

    logger.info(f"  Binary:      {best_binary_clf} (F1={best_binary_f1:.3f})")
    logger.info(f"  Multi-class: {best_multiclass_clf} (F1={best_multiclass_f1:.3f})")

    # Cross-validation info
    logger.info("")
    logger.info("=" * 80)
    logger.info("Experimental Setup:")
    logger.info(f"  Cross-validation: {N_SPLITS} folds")
    logger.info(f"  NOK per fold: {TARGET_NOK_PER_FOLD}")
    logger.info(f"  OK per fold: {TARGET_OK_PER_FOLD}")
    logger.info(f"  OK:NOK ratio: 99:1")
    logger.info(f"  Train/test split: {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}")
    logger.info(f"  Random state: {RANDOM_STATE}")
