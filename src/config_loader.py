import tomllib
from pathlib import Path


def load_class_config(class_set: str = "all"):
    """Load fault class configuration from TOML file.

    Args:
        class_set (str): The class set to load (e.g., 'all', 'top3', 'top5', 'top10').

    Returns:
        list: List of class names from the specified class set.
    """
    # Path from project root
    config_file = Path(__file__).parent.parent / "config" / "fault_classes.toml"

    with open(config_file, "rb") as f:
        config = tomllib.load(f)

    try:
        return config["class_sets"][class_set]
    except KeyError:
        available = list(config["class_sets"].keys())
        raise ValueError(
            f"Invalid class_set '{class_set}'. Available options: {available}"
        )
