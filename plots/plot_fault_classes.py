"""
Visualization module for fault class time series.

Create this file as: src/visualization/plot_fault_classes.py

This module provides functions to visualize torque time series for different
fault classes in a clean 2x3 grid layout.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from src.utils import get_logger

logger = get_logger(__name__)


def plot_fault_classes(
    x_values: NDArray,
    y_true: NDArray,
    label_mapping: Dict[str, int],
    n_samples_per_class: int = 10,
    output_dir: str = "results/figures",
    filename: str = "fault_classes_overview.png",
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> Path:
    """
    Create a 2x3 grid plot showing example torque curves for each fault class.

    Args:
        x_values: Time series data (n_samples, n_timepoints)
        y_true: Ground truth labels (n_samples,)
        label_mapping: Dict mapping class names to integer labels
        n_samples_per_class: Number of example curves to plot per class
        output_dir: Directory to save the figure
        filename: Filename for the saved figure
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure

    Returns:
        Path: Path to saved figure

    Example:
        >>> from src.data import run_data_pipeline, load_class_config
        >>> from src.visualization import plot_fault_classes
        >>>
        >>> x, y, mapping = run_data_pipeline(classes_to_keep=load_class_config("top5"))
        >>> plot_path = plot_fault_classes(x, y, mapping)
        >>> print(f"Saved plot to: {plot_path}")
    """
    logger.info("Creating fault class overview plot")
    logger.debug(
        f"Parameters: n_samples={n_samples_per_class}, figsize={figsize}, dpi={dpi}"
    )

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Reverse the label mapping to get int -> name
    int_to_name = {v: k for k, v in label_mapping.items()}
    n_classes = len(int_to_name)

    logger.debug(f"Found {n_classes} classes: {list(int_to_name.values())}")

    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Plot each class
    for class_idx in range(n_classes):
        ax = axes[class_idx]

        # Get samples for this class
        class_mask = y_true == class_idx
        class_samples = x_values[class_mask]

        if len(class_samples) == 0:
            logger.warning(
                f"No samples found for class {class_idx} ({int_to_name.get(class_idx, 'unknown')})"
            )
            ax.text(
                0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"Class {class_idx}: {int_to_name.get(class_idx, 'unknown')}")
            continue

        # Randomly select n_samples_per_class (or all if fewer available)
        n_to_plot = min(n_samples_per_class, len(class_samples))
        indices = np.random.choice(len(class_samples), size=n_to_plot, replace=False)
        samples_to_plot = class_samples[indices]

        logger.debug(
            f"Class {class_idx}: plotting {n_to_plot}/{len(class_samples)} samples"
        )

        # Plot each sample as a semi-transparent line
        for sample in samples_to_plot:
            ax.plot(sample, alpha=0.6, linewidth=1)

        # Add mean curve as thick line
        mean_curve = class_samples.mean(axis=0)
        ax.plot(mean_curve, color="black", linewidth=2.5, label="Mean", alpha=0.8)

        # Formatting
        class_name = int_to_name.get(class_idx, f"Class {class_idx}")
        # Shorten long class names for readability
        if len(class_name) > 40:
            class_name = class_name[:37] + "..."

        ax.set_title(f"Class {class_idx}: {class_name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("Time Point", fontsize=9)
        ax.set_ylabel("Torque", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

        # Add sample count annotation
        ax.text(
            0.02,
            0.98,
            f"n={len(class_samples)}",
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Hide unused subplots (if less than 6 classes)
    for idx in range(n_classes, 6):
        axes[idx].axis("off")

    # Overall title
    fig.suptitle(
        "Fault Class Time Series Overview",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save figure
    save_path = output_path / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {save_path}")

    return save_path.absolute()


def plot_single_class(
    x_values: NDArray,
    y_true: NDArray,
    class_label: int,
    class_name: str,
    n_samples: int = 20,
    output_dir: str = "results/figures",
    filename: Optional[str] = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
) -> Path:
    """
    Create a detailed plot for a single fault class.

    Args:
        x_values: Time series data (n_samples, n_timepoints)
        y_true: Ground truth labels (n_samples,)
        class_label: Integer label of the class to plot
        class_name: Name of the class (for title)
        n_samples: Number of example curves to plot
        output_dir: Directory to save the figure
        filename: Filename for saved figure (auto-generated if None)
        figsize: Figure size (width, height) in inches
        dpi: Resolution for saved figure

    Returns:
        Path: Path to saved figure
    """
    logger.info(f"Creating detailed plot for class {class_label}: {class_name}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get samples for this class
    class_mask = y_true == class_label
    class_samples = x_values[class_mask]

    if len(class_samples) == 0:
        logger.error(f"No samples found for class {class_label}")
        raise ValueError(f"No samples found for class {class_label}")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Randomly select samples
    n_to_plot = min(n_samples, len(class_samples))
    indices = np.random.choice(len(class_samples), size=n_to_plot, replace=False)
    samples_to_plot = class_samples[indices]

    # Plot individual samples
    for i, sample in enumerate(samples_to_plot):
        ax.plot(sample, alpha=0.4, linewidth=1, color="steelblue")

    # Plot mean and std
    mean_curve = class_samples.mean(axis=0)
    std_curve = class_samples.std(axis=0)

    ax.plot(mean_curve, color="black", linewidth=2.5, label="Mean", zorder=10)
    ax.fill_between(
        range(len(mean_curve)),
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.2,
        color="gray",
        label="±1 Std Dev",
    )

    # Formatting
    ax.set_title(
        f"Detailed View: Class {class_label} - {class_name}\n(Showing {n_to_plot} of {len(class_samples)} samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Time Point", fontsize=11)
    ax.set_ylabel("Torque", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    # Generate filename if not provided
    if filename is None:
        # Clean class name for filename
        clean_name = class_name.replace(" ", "_").replace("/", "-")[:30]
        filename = f"class_{class_label}_{clean_name}.png"

    save_path = output_path / filename
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved plot to: {save_path}")

    return save_path.absolute()


# Example usage
def plot_top_5_errors():
    # This demonstrates how to use the functions
    from src.data import load_class_config, run_data_pipeline

    # Load data
    x_values, y_true, label_mapping = run_data_pipeline(
        classes_to_keep=load_class_config("top5")
    )

    # Create overview plot
    overview_path = plot_fault_classes(x_values, y_true, label_mapping)
    print(f"Created overview plot: {overview_path}")

    # Create detailed plot for all five classes:
    for i in range(1, 6):
        int_to_name = {v: k for k, v in label_mapping.items()}
        if i in int_to_name:
            detail_path = plot_single_class(
                x_values, y_true, class_label=i, class_name=int_to_name[i]
            )
            print(f"Created detailed plot: {detail_path}")
