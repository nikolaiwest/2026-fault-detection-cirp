"""
Two-stage quality control pipeline orchestrator.

Coordinates the complete workflow:
1. Data loading and preprocessing
2. Cross-validation fold preparation
3. Stage 1: Anomaly detection (per fold)
4. Stage 2: Fault clustering (per fold)
5. Results aggregation and reporting
"""

from dataclasses import asdict

import numpy as np
import pandas as pd

from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds, report_cv_results
from src.methodology.stage1 import run_stage1
from src.methodology.stage2 import run_stage2
from src.utils import ResultManager, get_logger

logger = get_logger(__name__)


def run_two_stage_pipeline(config_name: str = "default-top5.yml"):
    """
    Main entry point for the two-stage methodology.

    Executes complete pipeline with cross-validation:
    1. Loads screw-driving data
    2. Prepares k-fold CV with OK upsampling
    3. Runs Stage 1 (anomaly detection) per fold
    4. Runs Stage 2 (fault clustering) per fold
    5. Reports aggregated metrics

    Args:
        config_name: Name of YAML config file in configs/ directory
                    (default: "default-top5.yml")

    Returns:
        None (logs results to console and file)
    """
    logger.section("TWO-STAGE PIPELINE EXECUTION")
    logger.info(f"Using configuration: {config_name}")

    # Initialize results manager
    results_manager = ResultManager(config_name=config_name)
    logger.info(f"Results will be saved to: {results_manager.get_run_directory()}")

    # Load typed config
    logger.info("Loading configuration")
    config = load_pipeline_config(config_name)

    # Load data
    logger.subsection("Data Loading")
    x_values, y_values, label_mapping = run_data_pipeline(
        force_reload=config.data.force_reload,
        keep_exceptions=config.data.keep_exceptions,
        classes_to_keep=load_class_config(config.data.classes_to_keep),
        paa_segments=config.data.paa_segments,
    )

    # Save metadata
    results_manager.save_metadata(
        config=asdict(config),
        additional_info={
            "data_shape": x_values.shape,
            "n_samples": len(y_values),
            "class_distribution": {
                int(k): int(v) for k, v in zip(*np.unique(y_values, return_counts=True))
            },
            "label_mapping": label_mapping,
        },
    )

    # Prepare cross-validation folds
    logger.subsection("Cross-Validation Setup")
    cv_folds = prepare_cv_folds(
        x_values=x_values,
        y_values=y_values,
        n_splits=config.cross_validation.n_splits,
        target_nok_per_fold=config.cross_validation.target_nok_per_fold,
        target_ok_per_fold=config.cross_validation.target_ok_per_fold,
        random_state=config.cross_validation.random_state,
    )
    logger.info(f"Prepared {len(cv_folds)} cross-validation folds")

    # Run two-stage pipeline on each fold
    logger.subsection("Fold Processing")
    results = []
    fold_metrics_data = []

    for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
        logger.section(f"FOLD {fold_num}/{len(cv_folds)}")
        logger.debug(f"Fold {fold_num} shapes: X={x_fold.shape}, y={y_fold.shape}")

        # Stage 1: Anomaly detection
        logger.info(
            f"Running Stage 1: {config.stage1.model_name} "
            f"(contamination={config.stage1.contamination})"
        )
        stage1_results = run_stage1(
            x_values=x_fold,
            y_values=y_fold,
            model_name=config.stage1.model_name,
            contamination=config.stage1.contamination,
            random_state=config.stage1.random_state,
        )
        logger.info(
            f"Stage 1 complete: {stage1_results['y_anomalies'].sum()} anomalies detected"
        )

        # Save Stage 1 results
        results_manager.save_stage1_results(
            fold_num=fold_num,
            y_true=stage1_results["y_true"],
            y_anomalies=stage1_results["y_anomalies"],
            anomaly_scores=stage1_results["anomaly_scores"],
            metrics=stage1_results["metrics"],
            confusion_matrix=stage1_results["confusion_matrix"],
        )

        # Stage 2: Fault clustering
        logger.info(
            f"Running Stage 2: {config.stage2.model_name} "
            f"(metric={config.stage2.metric}, n_clusters={config.stage2.n_clusters})"
        )
        stage2_results = run_stage2(
            x_values=x_fold,
            y_anomalies=stage1_results["y_anomalies"],
            y_true=y_fold,
            label_mapping=label_mapping,
            model_name=config.stage2.model_name,
            target_ok_to_sample=config.stage2.target_ok_to_sample,
            metric=config.stage2.metric,
            ok_reference_threshold=config.stage2.ok_reference_threshold,
            n_clusters=config.stage2.n_clusters,
            random_state=config.stage2.random_state,
        )
        logger.info(
            f"Stage 2 complete: {stage2_results['y_predictions'].sum()} faults predicted"
        )

        # Save Stage 2 results
        results_manager.save_stage2_results(
            fold_num=fold_num,
            y_true=stage2_results["y_true"],
            y_clusters=stage2_results["y_clusters"],
            y_predictions=stage2_results["y_predictions"],
            ok_reference_mask=stage2_results["ok_reference_mask"],
            metrics=stage2_results["metrics"],
            confusion_matrix=stage2_results["confusion_matrix"],
            cluster_stats=stage2_results["cluster_stats"],
            x_values=stage2_results["x_clustered"],
        )

        # Store for aggregation
        results.append(
            {
                "stage1": stage1_results,
                "stage2": stage2_results,
            }
        )

        # Collect metrics for comparison
        fold_metrics_data.append(
            {
                "fold": fold_num,
                "stage1_precision": stage1_results["metrics"]["precision"],
                "stage1_recall": stage1_results["metrics"]["recall"],
                "stage1_f1": stage1_results["metrics"]["f1"],
                "stage2_precision": stage2_results["metrics"]["precision"],
                "stage2_recall": stage2_results["metrics"]["recall"],
                "stage2_f1": stage2_results["metrics"]["f1"],
            }
        )

        logger.debug(f"Fold {fold_num} results saved")

    # Report and save aggregated results
    logger.subsection("Results Aggregation")
    fold_metrics_df = pd.DataFrame(fold_metrics_data)
    report_cv_results(results)  # Console logging

    results_manager.save_aggregated_results(
        results=results,
        fold_metrics=fold_metrics_df,
    )

    logger.section("PIPELINE COMPLETE")
    logger.info(f"All results saved to: {results_manager.get_run_directory()}")
