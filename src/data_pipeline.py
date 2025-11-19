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
    """Remove samples with measurement problems (not real faults).

    Measurement exceptions are indicated by 'scenario_exception' == 1.
    These are recording issues, not actual process faults."""

    if keep:
        print("Keeping all samples (including exceptions)")
        return data

    # Get exception mask (True = keep, False = remove)
    exceptions = data["scenario_exception"]
    keep_mask = [exc != 1 for exc in exceptions]

    n_total = len(exceptions)
    n_exceptions = sum(1 for exc in exceptions if exc == 1)
    n_keep = sum(keep_mask)

    print(f"- Found {n_exceptions} exceptions in {n_total} samples, keeping {n_keep}")

    # Use the mask to filter all dict fields
    filtered_data = {}
    for key, values in data.items():
        # Check if it's a list/array we should filter
        if isinstance(values, (list, tuple)) and len(values) == n_total:
            filtered_data[key] = [v for v, keep in zip(values, keep_mask) if keep]
        elif isinstance(values, (list, tuple)):
            # Different length found, something is wrong with the raw data
            raise ValueError(
                f"Field '{key}' has unexpected length {len(values)} "
                f"(expected {n_total}). Data structure inconsistent!"
            )
        else:
            # Not a list, this should not happen in pyscrew data
            raise TypeError(
                f"Field '{key}' is type {type(values).__name__}, expected list. "
                f"Data structure unexpected!"
            )

    return filtered_data


def filter_classes(data: Dict, classes: List[str]) -> Dict:
    """Keep only selected fault classes."""
    return data


def keep_only_torque(data: Dict) -> Dict:
    """Extract torque + labels, drop other signals."""
    return data


def upsample_normal_runs(data: Dict, ratio: float) -> Dict:
    """SMOTE upsampling of OK class."""
    return data


def encode_labels(data: Dict) -> Dict:
    """Int-encoding for the string representations of 'class_values'."""
    return data
