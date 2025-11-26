from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology.cross_validation import prepare_cv_folds, report_cv_results
from src.methodology.stage1 import run_stage1
from src.methodology.stage2 import run_stage2


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
        None (prints results to console)
    """
    # Load typed config
    config = load_pipeline_config(config_name)

    # Load data
    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=config.data.force_reload,
        keep_exceptions=config.data.keep_exceptions,
        classes_to_keep=load_class_config(config.data.classes_to_keep),
    )

    # Prepare cross-validation folds
    cv_folds = prepare_cv_folds(
        x_values=x_values,
        y_true=y_true,
        n_splits=config.cross_validation.n_splits,
        target_nok_per_fold=config.cross_validation.target_nok_per_fold,
        target_ok_per_fold=config.cross_validation.target_ok_per_fold,
        random_state=config.cross_validation.random_state,
    )

    # Run two-stage pipeline on each fold
    results = []
    for fold_num, (x_fold, y_fold) in enumerate(cv_folds, 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_num}/{len(cv_folds)}")
        print(f"{'='*70}")

        # Stage 1: Anomaly detection
        y_anomalies, anomaly_scores = run_stage1(
            x_values=x_fold,
            y_true=y_fold,
            contamination=config.stage1.contamination,
            random_state=config.stage1.random_state,
        )

        # Stage 2: Fault clustering
        y_clusters = run_stage2(
            x_values=x_fold,
            y_anomalies=y_anomalies,
            y_true=y_fold,
            label_mapping=label_mapping,
            ok_reference_ratio=config.stage2.ok_reference_ratio,
            use_dtw=config.stage2.use_dtw,
            n_clusters=config.stage2.n_clusters,
            random_state=config.stage2.random_state,
        )

        results.append(
            {
                "y_anomalies": y_anomalies,
                "anomaly_scores": anomaly_scores,
                "y_clusters": y_clusters,
                "y_true": y_fold,
            }
        )

    # Report aggregated results
    report_cv_results(results)
