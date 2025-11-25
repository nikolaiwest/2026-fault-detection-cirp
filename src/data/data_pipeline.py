import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyscrew
from imblearn.over_sampling import SMOTE

from .config_loader import load_class_config

NORMAL_CLASS_VALUE = "000_normal-observations"


def run_data_pipeline(
    force_reload: bool = False,
    keep_exceptions: bool = False,
    classes_to_keep: list[str] | None = None,
    target_ok_ratio: float = 0.99,
) -> Dict:
    """
    Main interface to execute the complete data preprocessing pipeline.

    Args:
        force_reload: Reload from PyScrew (will ignore the cache)
        keep_exceptions: If True, keep measurement exceptions (default: remove)
        classes_to_keep: List of class names to keep (uses "all" if set to None)
        target_ok_ratio: Target ratio of OK samples (0.99 = 99% OK, 1% faults)
    """

    if classes_to_keep is None:
        classes_to_keep = load_class_config("all")

    print("Starting pipeline...")

    # Step 1: Load s04 data from PyScrew (hosted public on Zenodo)
    data = load_data(force_reload)

    # Step 2: Remove all scenario exceptions (with issues during recording)
    data = remove_exceptions(data, keep_exceptions)

    # Step 3: Create normal class from scenario_condition == 'normal'
    data = create_ok_class(data)

    # Step 4: Limit the number of classes (for easier understanding)
    data = filter_classes(data, classes_to_keep)

    # Step 5: Extract torque as only measurements (others are not needed)
    data = keep_only_torque(data)

    # Step 6: Upsample normal observations (to achieve a target OK ratio)
    data = upsample_normal_runs(data, target_ok_ratio)

    # Step 7: Use ints to represent the class values (originally str)
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
    """
    Remove samples with measurement problems (not real faults).

    Measurement exceptions are indicated by 'scenario_exception' == 1.
    These are recording issues, not actual process faults.
    """

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


def create_ok_class(data: Dict) -> Dict:
    """
    Create '000_normal-observations' class from all normal samples. Overwrites class_values
    with '000_normal-observations' where scenario_condition == 'normal'.

    Originally, normal and faulty data was recorded alternately to prevent temporal factors
    from influencing the observations. To obtain a pure OK class, normal observations must
    therefore be sorted into their own class.
    """

    scenario_conditions = data["scenario_condition"]
    class_values = data["class_values"]

    # Overwrite class_values for normal observations
    new_class_values = []
    for condition, original_class in zip(scenario_conditions, class_values):
        if condition == "normal":
            new_class_values.append(NORMAL_CLASS_VALUE)
        else:
            new_class_values.append(original_class)

    n_ok = sum(1 for c in new_class_values if c == NORMAL_CLASS_VALUE)
    n_faults = len(new_class_values) - n_ok

    print(f"- Separated '000_normal-observations': {n_ok} normal, {n_faults} faults")

    data["class_values"] = new_class_values
    return data


def filter_classes(data: Dict, classes: List[str]) -> Dict:
    """
    Keep only the selected fault classes by simple filtering.
    The classes used in the list were selected based on their origin (one per each
    group of error causes) and their general sense of uniqueness.
    """

    # Always include normal class
    classes_with_normal = classes + [NORMAL_CLASS_VALUE]

    if not classes_with_normal:
        print("- No classes specified, keeping all samples")
        return data

    # Get class mask (True = keep, False = remove)
    class_values = data["class_values"]
    keep_mask = [cls in classes_with_normal for cls in class_values]

    n_total = len(class_values)
    n_keep = sum(keep_mask)
    n_remove = n_total - n_keep

    print(f"- Filter to {len(classes)} classes, removing {n_remove}, keeping {n_keep}")

    # Use the mask to filter all dict fields
    filtered_data = {}
    for key, values in data.items():
        if isinstance(values, (list, tuple)) and len(values) == n_total:
            filtered_data[key] = [v for v, keep in zip(values, keep_mask) if keep]
        elif isinstance(values, (list, tuple)):
            raise ValueError(
                f"Field '{key}' has unexpected length {len(values)} "
                f"(expected {n_total}). Data structure inconsistent!"
            )
        else:
            raise TypeError(
                f"Field '{key}' is type {type(values).__name__}, expected list. "
                f"Data structure unexpected!"
            )

    return filtered_data


def keep_only_torque(data: Dict) -> Dict:
    """Extract torque only labels,by dropping all other measurements."""

    print(f"- Keeping only torque and classes, dropping {len(data) - 2} fields")

    return {
        "torque_values": data["torque_values"],
        "class_values": data["class_values"],
    }


def upsample_normal_runs(data: Dict, ratio: float) -> Dict:
    """
    SMOTE upsampling of OK class to achieve target ratio.

    Upsampling the normal class aims to create more natural imbalances in the screw
    data. Synthetic samples are labeled as '001_artificial_ok' for transparency.
    """

    # Identify normal samples
    normal_label = "000_normal-observations"  # default value for OK in PyScrew "s04"
    class_values = data["class_values"]

    n_total = len(class_values)
    n_normal = sum(1 for c in class_values if c == normal_label)
    n_faults = n_total - n_normal

    # Calculate how many normal samples are needed
    # ratio = n_normal_target / (n_normal_target + n_faults)
    n_normal_target = int((ratio * n_faults) / (1 - ratio))
    n_synthetic = n_normal_target - n_normal

    if n_synthetic <= 0:
        print(f"- Already have {n_normal} ok samples (ratio: {n_normal/n_total:.2%})")
        print(f"- No upsampling needed for target ratio {ratio:.2%}")
        return data

    print(f"- Current: {n_normal} OK, {n_faults} NOK (ratio: {n_normal/n_total:.2%})")
    print(f"- Target ratio: {ratio:.2%} → need {n_normal_target} normal samples")
    print(f"- Generating {n_synthetic} synthetic samples with SMOTE...")

    # Prepare data for SMOTE
    torque_array = np.array(data["torque_values"])
    labels_binary = [1 if c == normal_label else 0 for c in class_values]

    # Apply SMOTE
    smote = SMOTE(sampling_strategy={1: n_normal_target}, random_state=42)
    torque_resampled, _ = smote.fit_resample(torque_array, labels_binary)

    # Split back into original + synthetic
    torque_synthetic = torque_resampled[n_total:]

    # Create labels for synthetic samples
    labels_synthetic = ["001_artificial_ok"] * n_synthetic

    # Append to original data
    data["torque_values"] = data["torque_values"] + torque_synthetic.tolist()
    data["class_values"] = data["class_values"] + labels_synthetic

    print(f"- Added {n_synthetic} synth. samples, total: {len(data['class_values'])}")

    return data


def encode_labels(data: Dict) -> Dict:
    """
    Int-encoding for the string representations of 'class_values'.

    Normal class ('000_normal-observations' or '001_artificial_ok') always gets label 0.
    Saves label mapping to JSON for later reference.
    """

    import json

    class_values = data["class_values"]

    # Separate normal and fault classes
    normal_classes = [NORMAL_CLASS_VALUE, "001_artificial_ok"]
    unique_classes = sorted(set(class_values))

    # Build mapping: normal classes = 0, faults = 1, 2, 3, ...
    label_to_int = {}

    # First: all normal classes to 0
    for normal_cls in normal_classes:
        if normal_cls in unique_classes:
            label_to_int[normal_cls] = 0

    # Then: fault classes to 1, 2, 3, ...
    fault_idx = 1
    for cls in unique_classes:
        if cls not in normal_classes:
            label_to_int[cls] = fault_idx
            fault_idx += 1

    # Encode labels
    encoded_labels = [label_to_int[label] for label in class_values]

    # Save mapping to JSON
    mapping_file = Path("data/processed/label_mapping.json")
    mapping_file.parent.mkdir(parents=True, exist_ok=True)

    with open(mapping_file, "w") as f:
        json.dump(label_to_int, f, indent=2)

    n_normal = sum(1 for lbl in encoded_labels if lbl == 0)
    n_faults = len(encoded_labels) - n_normal

    print(
        f"- Encoded {len(label_to_int)} classes: {n_normal} normal (0), {n_faults} faults (1-{fault_idx-1})"
    )
    print(f"- Saved mapping to {mapping_file}")

    return {
        "torque_values": data["torque_values"],
        "labels": encoded_labels,
        "label_mapping": label_to_int,
    }
