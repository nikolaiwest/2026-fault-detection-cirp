import logging

from src.benchmark import (
    run_binary_fault_comparison,
    run_semi_supervised_classification_benchmark,
    run_stage1_benchmark,
    run_stage2_benchmark,
    run_supervised_classification_benchmark,
)
from src.data import load_class_config, load_pipeline_config, run_data_pipeline
from src.methodology import run_two_stage_pipeline
from src.utils import get_logger, set_level

# Initialize logger for main module
logger = get_logger(__name__)


def test_pipeline():
    """Quick test for the data pipeline with default parameters."""
    # Load typed config
    config_name = "default-top5.yml"
    config = load_pipeline_config(config_name)

    # Load data
    x_values, y_true, label_mapping = run_data_pipeline(
        force_reload=config.data.force_reload,
        keep_exceptions=config.data.keep_exceptions,
        classes_to_keep=load_class_config(config.data.classes_to_keep),
        paa_segments=config.data.paa_segments,
    )

    logger.info("Pipeline test complete!")


if __name__ == "__main__":
    set_level(logging.DEBUG)

    logger.section("TWO-STAGE QUALITY CONTROL PIPELINE")

    # Temporary test during development
    # test_pipeline()

    run_two_stage_pipeline()
    logger.info("Execution complete")
