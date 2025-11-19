import pickle
from pathlib import Path
from typing import Dict, List

import pyscrew

CLASSES_TO_KEEP = ["001_control-group", "101_deformed-thread"]


def run_data_pipeline(
    force_reload: bool = False,
    keep_exceptions: bool = False,
    classes_to_keep: list[str] = CLASSES_TO_KEEP,
    target_ok_ratio: float = 0.99,
) -> Dict:
    """Main interface to execute the complete data preprocessing pipeline.

    Args:
        force_reload: Reload from PyScrew (will ignore the cache)
        keep_exceptions: If True, keep measurement exceptions (default: remove)
        classes_to_keep: List of class names to keep (default set here as globals)
        target_ok_ratio: Target ratio of OK samples (0.99 = 99% OK, 1% faults)"""

    print("Starting pipeline...")

    # Step 1: Load s04 data from PyScrew (hosted public on Zenodo)
    data = load_data(force_reload)

    # Step 2: Remove all scenario exceptions (with issues during recording)
    data = remove_exceptions(data, keep_exceptions)

    # Step 3: Limit the number of classes (for easier understanding)
    data = filter_classes(data, classes_to_keep)

    # Step 4: Extract torque as only measurements (others are not needed)
    data = keep_only_torque(data)

    # Step 5: Upsample normal observations (to achieve a target OK ratio)
    data = upsample_normal_runs(data, target_ok_ratio)

    # Step 6: Use ints to represent the value values (originally str)
    data = encode_labels(data)

    print("Pipeline complete!")
    return data


def load_data(force_reload=False):
    """Load and cache preprocessed screw driving data."""
    # Define cache path
    cache_file = Path("data/processed/pyscrew_s04.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if cache exists and should be used
    if cache_file.exists() and not force_reload:
        print(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        print(f"Loaded {len(data['torque_values'])} samples from cache")
        return data

    # Load fresh data from pyscrew
    print("Loading data from pyscrew (this may take a few minutes)...")
    data = pyscrew.get_data(
        scenario="s04",
        screw_positions="left",
        cache_dir="data/pyscrew/",
        force_download=force_reload,
        handle_duplicates="first",
        handle_missings="mean",
        target_length=2000,  # Magic number, but we simply set it to the max length here
    )

    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    print(f"Loaded and cached {len(data['torque_values'])} samples")
    return data


def remove_exceptions(data: Dict, keep: bool = False) -> Dict:
    """Remove measurement problems (not real faults)."""
    pass


def filter_classes(data: Dict, classes: List[str]) -> Dict:
    """Keep only selected fault classes."""
    pass


def keep_only_torque(data: Dict) -> Dict:
    """Extract torque + labels, drop other signals."""
    pass


def upsample_normal_runs(data: Dict, ratio: float) -> Dict:
    """SMOTE upsampling of OK class."""
    pass


def encode_labels(data: Dict) -> Dict:
    """Int-encoding for the string representations of 'class_values'."""
    pass
