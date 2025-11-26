import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    classes_to_keep: str
    force_reload: bool
    keep_exceptions: bool


@dataclass
class CVConfig:
    """Cross-validation configuration."""

    n_splits: int
    target_nok_per_fold: int | None
    target_ok_per_fold: int | float | None
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

    ok_reference_ratio: float
    n_clusters: int
    use_dtw: bool
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

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Validate required sections exist
    required_sections = ["data", "cross_validation", "stage1", "stage2"]
    missing = [s for s in required_sections if s not in config_dict]
    if missing:
        raise KeyError(f"Missing required config sections: {missing}")

    # Build typed config from dict (will raise TypeError if keys missing)
    try:
        return PipelineConfig(
            data=DataConfig(**config_dict["data"]),
            cross_validation=CVConfig(**config_dict["cross_validation"]),
            stage1=Stage1Config(**config_dict["stage1"]),
            stage2=Stage2Config(**config_dict["stage2"]),
        )
    except TypeError as e:
        raise TypeError(f"Invalid config structure: {e}") from e


def load_class_config(class_set: str = "all"):
    """Load fault class configuration from TOML file.

    Args:
        class_set (str): The class set to load (e.g., 'all', 'top5', 'top10').

    Returns:
        list: List of class names from the specified class set.
    """
    # Path from project root
    config_file = Path(__file__).parent.parent.parent / "configs" / "fault_classes.toml"

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    try:
        return config["class_sets"][class_set]
    except KeyError:
        available = list(config["class_sets"].keys())
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

    if not config_file.exists():
        raise FileNotFoundError(
            f"Hyperparameters file not found: {config_file}\n"
            f"Expected structure: configs/stage_{stage_number}/hyperparameters.yml"
        )

    with open(config_file, "r") as f:
        all_hyperparams = yaml.safe_load(f)

    if model_name not in all_hyperparams:
        available = list(all_hyperparams.keys())
        raise ValueError(
            f"Model '{model_name}' not found in stage {stage_number} config.\n"
            f"Available models: {available}"
        )

    return all_hyperparams[model_name]
