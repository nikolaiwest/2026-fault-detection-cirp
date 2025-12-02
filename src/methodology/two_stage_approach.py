"""
Two-stage quality control pipeline orchestrator.

Coordinates the complete workflow:
1. Data loading and preprocessing
2. Cross-validation fold preparation
3. Stage 1: Anomaly detection (per fold)
4. Stage 2: Fault clustering (per fold)
5. Results aggregation and reporting
"""

from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds, report_cv_results
from src.methodology.stage1 import run_stage1
from src.methodology.stage2 import run_stage2
from src.utils.logger import get_logger

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

    for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
        logger.section(f"FOLD {fold_num}/{len(cv_folds)}")
        logger.debug(f"Fold {fold_num} shapes: X={x_fold.shape}, y={y_fold.shape}")

        # Stage 1: Anomaly detection
        logger.info(
            f"Running Stage 1: {config.stage1.model_name} (contamination={config.stage1.contamination})"
        )
        y_anomalies, anomaly_scores = run_stage1(
            x_values=x_fold,
            y_values=y_fold,
            model_name=config.stage1.model_name,
            contamination=config.stage1.contamination,
            random_state=config.stage1.random_state,
        )
        logger.info(f"Stage 1 complete: {y_anomalies.sum()} anomalies detected")

        # Stage 2: Fault clustering
        logger.info(
            f"Running Stage 2: {config.stage2.model_name} "
            f"(metric={config.stage2.metric}, n_clusters={config.stage2.n_clusters})"
        )
        stage2_predictions = run_stage2(
            x_values=x_fold,
            y_anomalies=y_anomalies,
            y_true=y_fold,
            label_mapping=label_mapping,
            model_name=config.stage2.model_name,
            target_ok_to_sample=config.stage2.target_ok_to_sample,
            metric=config.stage2.metric,
            ok_reference_threshold=config.stage2.ok_reference_threshold,
            n_clusters=config.stage2.n_clusters,
            random_state=config.stage2.random_state,
        )
        logger.info(f"Stage 2 complete: {stage2_predictions.sum()} faults predicted")

        results.append(
            {
                "y_anomalies": y_anomalies,
                "anomaly_scores": anomaly_scores,
                "stage2_predictions": stage2_predictions,
                "y_true": y_fold,
            }
        )

        logger.debug(f"Fold {fold_num} results stored")

    # Report aggregated results
    logger.subsection("Results Aggregation")
    report_cv_results(results)

    logger.section("PIPELINE COMPLETE")
