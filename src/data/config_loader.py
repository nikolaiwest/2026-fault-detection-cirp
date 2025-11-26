import tomllib
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DataConfig:
    """Data pipeline configuration."""

    classes_to_keep: str = "top5"
    force_reload: bool = False
    keep_exceptions: bool = False


@dataclass
class CVConfig:
    """Cross-validation configuration."""

    n_splits: int = 5
    # NOK: int=target, None=use all samples
    target_nok_per_fold: int | None = None
    # OK: int=count, float=ratio, None=no upsampling
    target_ok_per_fold: int | float | None = 0.99
    random_state: int = 42


@dataclass
class Stage1Config:
    """Stage 1 anomaly detection configuration."""

    contamination: float = 0.02
    random_state: int = 42


@dataclass
class Stage2Config:
    """Stage 2 clustering configuration."""

    ok_reference_ratio: float = 0.01
    n_clusters: int = 5
    use_dtw: bool = False
    random_state: int = 42


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

    Args:
        config_name: Name of config file (e.g., "default-top5.yml")

    Returns:
        PipelineConfig: Typed configuration object
    """
    config_path = Path("configs") / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Build typed config from dict
    return PipelineConfig(
        data=DataConfig(**config_dict["data"]),
        cross_validation=CVConfig(**config_dict["cross_validation"]),
        stage1=Stage1Config(**config_dict["stage1"]),
        stage2=Stage2Config(**config_dict["stage2"]),
    )


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
