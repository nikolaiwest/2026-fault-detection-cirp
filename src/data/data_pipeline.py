"""
Data preprocessing pipeline for screw driving quality control.

Handles loading, filtering, and preprocessing of screw driving measurements
from the PyScrew dataset. Transforms raw data into ML-ready format with
encoded labels and extracted torque features.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyscrew

from src.utils.logger import get_logger

from .config_loader import load_class_config

logger = get_logger(__name__)

NORMAL_CLASS_VALUE = "000_normal-observations"


def run_data_pipeline(
    force_reload: bool = False,
    keep_exceptions: bool = False,
    classes_to_keep: list[str] | None = None,
    paa_segments: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Main interface to execute the complete data preprocessing pipeline.

    Args:
        force_reload: Reload from PyScrew (will ignore the cache)
        keep_exceptions: If True, keep measurement exceptions (default: remove)
        classes_to_keep: List of class names to keep (uses "all" if set to None)

    Returns:
        tuple: (x_values, y_values, label_mapping)
            - x_values: numpy array of torque measurements
            - y_values: numpy array of encoded labels
            - label_mapping: dict mapping class names to integer labels
    """

    if classes_to_keep is None:
        classes_to_keep = load_class_config("all")

    logger.info("Starting data pipeline")
    logger.debug(
        f"Parameters: force_reload={force_reload}, keep_exceptions={keep_exceptions}, n_classes={len(classes_to_keep)}"
    )

    # Step 1: Load s04 data from PyScrew (hosted public on Zenodo)
    data = _load_data(force_reload)

    # Step 2: Remove all scenario exceptions (with issues during recording)
    data = _remove_exceptions(data, keep_exceptions)

    # Step 3: Create normal class from scenario_condition == 'normal'
    data = _create_ok_class(data)

    # Step 4: Limit the number of classes (for easier understanding)
    data = _filter_classes(data, classes_to_keep)

    # Step 5: Extract torque as only measurements (others are not needed)
    data = _keep_only_torque(data)

    # Step 6: Use ints to represent the class values (originally str)
    data = _encode_labels(data)

    # Step 7:  Apply PAA
    data = _apply_paa(data, paa_segments)

    # Step 8: Unpack and convert to numpy arrays
    data = _unpack_and_convert(data)

    return data


def _load_data(force_reload=False):
    """
    Load and cache preprocessed screw driving data.

    Args:
        force_reload: If True, download fresh data from PyScrew

    Returns:
        dict: PyScrew data dictionary with torque and metadata
    """
    # Define cache path
    cache_file = Path("data/processed/pyscrew_s04.pkl")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if cache exists and should be used
    if cache_file.exists() and not force_reload:
        logger.info(f"Loading cached data from {cache_file}")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        logger.info(f"Loaded {len(data['torque_values'])} samples from cache")
        return data

    # Load fresh data from pyscrew
    logger.info("Loading data from PyScrew (this may take a few minutes)")
    logger.debug("Downloading from Zenodo repository")

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
    logger.info(f"Caching data to {cache_file}")
    with open(cache_file, "wb") as f:
        pickle.dump(data, f)

    logger.info(f"Loaded and cached {len(data['torque_values'])} samples")
    return data


def _remove_exceptions(data: Dict, keep: bool = False) -> Dict:
    """
    Remove samples with measurement problems (not real faults).

    Measurement exceptions are indicated by 'scenario_exception' == 1.
    These are recording issues, not actual process faults.

    Args:
        data: PyScrew data dictionary
        keep: If True, keep exception samples

    Returns:
        dict: Filtered data dictionary
    """

    if keep:
        logger.info("Keeping all samples (including exceptions)")
        return data

    # Get exception mask (True = keep, False = remove)
    exceptions = data["scenario_exception"]
    keep_mask = [exc != 1 for exc in exceptions]

    n_total = len(exceptions)
    n_exceptions = sum(1 for exc in exceptions if exc == 1)
    n_keep = sum(keep_mask)

    logger.info(
        f"Found {n_exceptions} exceptions in {n_total} samples, keeping {n_keep}"
    )
    logger.debug(f"Exception removal rate: {n_exceptions/n_total:.1%}")

    # Use the mask to filter all dict fields
    filtered_data = {}
    for key, values in data.items():
        # Check if it's a list/array we should filter
        if isinstance(values, (list, tuple)) and len(values) == n_total:
            filtered_data[key] = [v for v, keep in zip(values, keep_mask) if keep]
        elif isinstance(values, (list, tuple)):
            # Different length found, something is wrong with the raw data
            logger.error(
                f"Field '{key}' has unexpected length {len(values)} (expected {n_total})"
            )
            raise ValueError(
                f"Field '{key}' has unexpected length {len(values)} "
                f"(expected {n_total}). Data structure inconsistent!"
            )
        else:
            # Not a list, this should not happen in pyscrew data
            logger.error(f"Field '{key}' has unexpected type {type(values).__name__}")
            raise TypeError(
                f"Field '{key}' is type {type(values).__name__}, expected list. "
                f"Data structure unexpected!"
            )

    return filtered_data


def _create_ok_class(data: Dict) -> Dict:
    """
    Create '000_normal-observations' class from all normal samples.

    Overwrites class_values with '000_normal-observations' where
    scenario_condition == 'normal'.

    Originally, normal and faulty data was recorded alternately to prevent temporal
    factors from influencing the observations. To obtain a pure OK class, normal
    observations must therefore be sorted into their own class.

    Args:
        data: PyScrew data dictionary

    Returns:
        dict: Data with separated normal class
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

    logger.info(f"Separated normal class: {n_ok} OK samples, {n_faults} fault samples")
    logger.debug(f"OK ratio: {n_ok/len(new_class_values):.1%}")

    data["class_values"] = new_class_values
    return data


def _filter_classes(data: Dict, classes: List[str]) -> Dict:
    """
    Keep only the selected fault classes by simple filtering.

    The classes used in the list were selected based on their origin (one per each
    group of error causes) and their general sense of uniqueness.

    Args:
        data: PyScrew data dictionary
        classes: List of fault class names to keep

    Returns:
        dict: Filtered data with selected classes only
    """

    # Always include normal class
    classes_with_normal = classes + [NORMAL_CLASS_VALUE]

    if not classes_with_normal:
        logger.warning("No classes specified, keeping all samples")
        return data

    # Get class mask (True = keep, False = remove)
    class_values = data["class_values"]
    keep_mask = [cls in classes_with_normal for cls in class_values]

    n_total = len(class_values)
    n_keep = sum(keep_mask)
    n_remove = n_total - n_keep

    logger.info(
        f"Filtering to {len(classes)} fault classes: removing {n_remove}, keeping {n_keep}"
    )
    logger.debug(
        f"Selected classes: {', '.join(classes[:3])}{'...' if len(classes) > 3 else ''}"
    )

    # Use the mask to filter all dict fields
    filtered_data = {}
    for key, values in data.items():
        if isinstance(values, (list, tuple)) and len(values) == n_total:
            filtered_data[key] = [v for v, keep in zip(values, keep_mask) if keep]
        elif isinstance(values, (list, tuple)):
            logger.error(
                f"Field '{key}' has unexpected length {len(values)} (expected {n_total})"
            )
            raise ValueError(
                f"Field '{key}' has unexpected length {len(values)} "
                f"(expected {n_total}). Data structure inconsistent!"
            )
        else:
            logger.error(f"Field '{key}' has unexpected type {type(values).__name__}")
            raise TypeError(
                f"Field '{key}' is type {type(values).__name__}, expected list. "
                f"Data structure unexpected!"
            )

    return filtered_data


def _keep_only_torque(data: Dict) -> Dict:
    """
    Extract torque and labels only, dropping all other measurements.

    Args:
        data: PyScrew data dictionary

    Returns:
        dict: Reduced data with torque and class values only
    """

    n_dropped = len(data) - 2
    logger.info(
        f"Keeping only torque and class labels, dropping {n_dropped} metadata fields"
    )
    logger.debug(
        f"Dropped fields: {', '.join([k for k in data.keys() if k not in ['torque_values', 'class_values']][:5])}"
    )

    return {
        "torque_values": data["torque_values"],
        "class_values": data["class_values"],
    }


def _encode_labels(data: Dict) -> Dict:
    """
    Int-encoding for the string representations of 'class_values'.

    Normal class ('000_normal-observations' or '001_artificial_ok') always gets label 0.
    Saves label mapping to JSON for later reference.

    Args:
        data: Data dictionary with string class labels

    Returns:
        dict: Data with integer-encoded labels and mapping
    """

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

    logger.info(
        f"Encoded {len(label_to_int)} classes: {n_normal} normal (0), {n_faults} faults (1-{fault_idx-1})"
    )
    logger.info(f"Saved label mapping to {mapping_file}")
    logger.debug(
        f"Label distribution: {dict(sorted([(v, class_values.count(k)) for k, v in label_to_int.items()]))}"
    )

    return {
        "torque_values": data["torque_values"],
        "labels": encoded_labels,
        "label_mapping": label_to_int,
    }


def _apply_paa(data: Dict, n_segments: int) -> Dict:
    """
    Apply Piecewise Aggregate Approximation (PAA) to torque values.

    Args:
        data: Dictionary containing 'torque_values'
        n_segments: Number of segments for PAA compression

    Returns:
        dict: Updated data dictionary with PAA-compressed torque_values
    """
    if n_segments is None:
        logger.warning("Received n_segments=None, skipping PAA step")
        return data

    X = np.array(data["torque_values"])
    n_samples, n_timepoints = X.shape

    # Case 1: impossible configuration
    if n_segments > n_timepoints:
        logger.error(
            f"PAA failed: n_segments ({n_segments}) > n_timepoints ({n_timepoints})"
        )
        raise ValueError(
            f"PAA requires n_segments <= n_timepoints, but got {n_segments} > {n_timepoints}"
        )

    # Case 2: same length → no PAA needed
    if n_segments == n_timepoints:
        logger.info(
            f"PAA skipped: n_segments equals current length ({n_timepoints}), data unchanged"
        )
        return data

    # Case 3: apply PAA normally
    segment_size = n_timepoints // n_segments
    cutoff = segment_size * n_segments

    logger.info(
        f"Applying PAA: {n_timepoints} → {n_segments} (segment size={segment_size})"
    )
    logger.debug(f"PAA cutoff: using first {cutoff} of {n_timepoints} values")

    X_reshaped = X[:, :cutoff].reshape(n_samples, n_segments, segment_size)
    X_paa = X_reshaped.mean(axis=2)

    data["torque_values"] = X_paa.tolist()
    return data


def _unpack_and_convert(data: Dict) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Unpack pipeline output dict into convenient tuple format.

    Args:
        data: Output from encode_labels()

    Returns:
        tuple: (x_values, y_values, label_mapping)
    """
    x_values = np.array(data["torque_values"])
    y_values = np.array(data["labels"])
    label_mapping = data["label_mapping"]

    logger.debug(f"Final shapes: X={x_values.shape}, y={y_values.shape}")

    return x_values, y_values, label_mapping
