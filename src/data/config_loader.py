"""
Configuration loading utilities for the pipeline.

Provides functions to load and validate configuration from YAML and TOML files:
- Pipeline configuration (YAML)
- Fault class sets (TOML)
- Model hyperparameters (YAML)
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    force_reload: bool
    keep_exceptions: bool
    classes_to_keep: str
    paa_segments: int


@dataclass
class CVConfig:
    """Cross-validation configuration."""

    n_splits: int
    target_nok_per_fold: Optional[int]
    target_ok_per_fold: Optional[Union[int, float]]
    random_state: int


@dataclass
class Stage1Config:
    """Stage 1 anomaly detection configuration."""

    model_name: str
    contamination: float
    random_state: int


@dataclass
class Stage2Config:
    """Stage 2 clustering configuration."""

    model_name: str
    metric: str
    target_ok_to_sample: int
    n_clusters: int
    ok_reference_threshold: int
    random_state: int


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""

    data: DataConfig
    cross_validation: CVConfig
    stage1: Stage1Config
    stage2: Stage2Config


def load_pipeline_config(config_name: str = "default-top5.yml") -> PipelineConfig:
    """
    Load pipeline configuration from YAML file.

    TODO: I might add config validation with pydantic later.

    Args:
        config_name: Name of config file (e.g., "default-top5.yml")

    Returns:
        PipelineConfig: Typed configuration object

    Raises:
        FileNotFoundError: If config file doesn't exist
        KeyError: If required config keys are missing
        TypeError: If config values have wrong types
    """
    config_path = Path("configs") / config_name

    logger.debug(f"Loading pipeline config from: {config_path}")

    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections exist
    required_sections = ["data", "cross_validation", "stage1", "stage2"]
    missing = [s for s in required_sections if s not in config_dict]
    if missing:
        logger.error(f"Missing required config sections: {missing}")
        raise KeyError(f"Missing required config sections: {missing}")

    # Build typed config from dict (will raise TypeError if keys missing)
    try:
        config = PipelineConfig(
            data=DataConfig(**config_dict["data"]),
            cross_validation=CVConfig(**config_dict["cross_validation"]),
            stage1=Stage1Config(**config_dict["stage1"]),
            stage2=Stage2Config(**config_dict["stage2"]),
        )
        logger.debug(f"Successfully loaded config: {config_name}")
        return config

    except TypeError as e:
        logger.error(f"Invalid config structure: {e}")
        logger.debug(f"Stage2 config content: {config_dict.get('stage2', 'MISSING')}")
        raise TypeError(f"Invalid config structure: {e}") from e


def load_class_config(class_set: str = "all"):
    """
    Load fault class configuration from TOML file.

    Args:
        class_set: The class set to load (e.g., 'all', 'top5', 'top10')

    Returns:
        list: List of class names from the specified class set

    Raises:
        ValueError: If class_set not found in configuration
    """
    # Path from project root
    config_file = Path(__file__).parent.parent.parent / "configs" / "fault_classes.toml"

    logger.debug(f"Loading class config '{class_set}' from: {config_file}")

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    try:
        classes = config["class_sets"][class_set]
        logger.debug(f"Loaded {len(classes)} classes from set '{class_set}'")
        return classes

    except KeyError:
        available = list(config["class_sets"].keys())
        logger.error(f"Invalid class_set '{class_set}'. Available: {available}")
        raise ValueError(
            f"Invalid class_set '{class_set}'. Available options: {available}"
        )


def load_model_config(stage_number: int, model_name: str) -> dict:
    """
    Load model-specific hyperparameters from YAML file.

    Args:
        stage_number: Stage number (1 or 2)
        model_name: Model name (e.g., "isolation_forest", "kmeans")

    Returns:
        dict: Hyperparameters for the specified model

    Raises:
        FileNotFoundError: If hyperparameters.yml doesn't exist
        ValueError: If model_name not found in config
    """
    # Path based on stage number
    config_file = (
        Path(__file__).parent.parent
        / "models"
        / f"stage_{stage_number}"
        / "hyperparameters.yml"
    )

    logger.debug(
        f"Loading hyperparameters for stage {stage_number}, model '{model_name}'"
    )

    if not config_file.exists():
        logger.error(f"Hyperparameters file not found: {config_file}")
        raise FileNotFoundError(
            f"Hyperparameters file not found: {config_file}\n"
            f"Expected structure: configs/stage_{stage_number}/hyperparameters.yml"
        )

    with open(config_file, "r") as f:
        all_hyperparams = yaml.safe_load(f)

    if model_name not in all_hyperparams:
        available = list(all_hyperparams.keys())
        logger.error(
            f"Model '{model_name}' not found in stage {stage_number} config. Available: {available}"
        )
        raise ValueError(
            f"Model '{model_name}' not found in stage {stage_number} config.\n"
            f"Available models: {available}"
        )

    hyperparams = all_hyperparams[model_name]
    logger.debug(f"Loaded hyperparameters for '{model_name}': {hyperparams}")
    return hyperparams
