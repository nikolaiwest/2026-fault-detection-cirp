"""
Handles saving experiment results in a structured format.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import ConfusionMatrixDisplay


class ResultManager:
    """Manages structured saving of experiment results."""

    def __init__(self, config_name: str, base_dir: str = "results"):
        """
        Initialize results manager with timestamped directory.

        Args:
            config_name: Name of config file (e.g., "default-top5.yml")
            base_dir: Base directory for all results (default: "results")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_stem = Path(config_name).stem
        self.run_dir = Path(base_dir) / f"{timestamp}__{config_stem}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.stage1_dir = self.run_dir / "stage1"
        self.stage2_dir = self.run_dir / "stage2"
        self.aggregated_dir = self.run_dir / "aggregated"

        for dir_path in [self.stage1_dir, self.stage2_dir, self.aggregated_dir]:
            dir_path.mkdir(exist_ok=True)

    def save_metadata(self, config: dict, additional_info: Optional[dict] = None):
        """
        Save run metadata and configuration.

        Args:
            config: Full configuration dictionary
            additional_info: Extra metadata (e.g., data shapes, class distribution)
        """
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "additional_info": additional_info or {},
        }

        with open(self.run_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    def save_stage1_results(
        self,
        fold_num: int,
        y_true: NDArray,
        y_anomalies: NDArray,
        anomaly_scores: NDArray,
        metrics: dict,
        confusion_matrix: NDArray,
    ):
        """
        Save Stage 1 results for a single fold.

        Args:
            fold_num: Fold number
            y_true: Ground truth labels
            y_anomalies: Binary predictions (0=OK, 1=NOK)
            anomaly_scores: Anomaly scores from decision function
            metrics: Dictionary with precision, recall, f1
            confusion_matrix: 2x2 confusion matrix
        """
        fold_dir = self.stage1_dir / f"fold_{fold_num}"
        fold_dir.mkdir(exist_ok=True)

        # Save predictions CSV
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(len(y_true)),
                "y_true": y_true,
                "y_true_binary": (y_true > 0).astype(int),
                "y_predicted": y_anomalies,
                "anomaly_score": anomaly_scores,
            }
        )
        df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save metrics JSON
        metrics_full = {
            **metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "n_samples": len(y_true),
            "n_ok_predicted": int((y_anomalies == 0).sum()),
            "n_nok_predicted": int((y_anomalies == 1).sum()),
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics_full, f, indent=2)

        # Save confusion matrix plot
        self._plot_confusion_matrix(
            confusion_matrix,
            title=f"Stage 1 - Fold {fold_num}",
            save_path=fold_dir / "confusion_matrix.png",
        )

    def save_stage2_results(
        self,
        fold_num: int,
        y_true: NDArray,
        y_clusters: NDArray,
        y_predictions: NDArray,
        ok_reference_mask: NDArray,
        metrics: dict,
        confusion_matrix: NDArray,
        cluster_stats: pd.DataFrame,
        x_values: Optional[NDArray] = None,
    ):
        """
        Save Stage 2 results for a single fold.

        Args:
            fold_num: Fold number
            y_true: Ground truth labels
            y_clusters: Cluster assignments
            y_predictions: Final binary predictions after filtering
            ok_reference_mask: Boolean mask for OK reference samples
            metrics: Dictionary with precision, recall, f1
            confusion_matrix: 2x2 confusion matrix
            cluster_stats: DataFrame with cluster composition analysis
            x_values: Optional feature matrix for visualization
        """
        fold_dir = self.stage2_dir / f"fold_{fold_num}"
        fold_dir.mkdir(exist_ok=True)

        # Save predictions CSV
        df = pd.DataFrame(
            {
                "sample_idx": np.arange(len(y_true)),
                "y_true": y_true,
                "y_true_binary": (y_true > 0).astype(int),
                "cluster_id": y_clusters,
                "is_ok_reference": ok_reference_mask,
                "y_predicted": y_predictions,
            }
        )
        df.to_csv(fold_dir / "predictions.csv", index=False)

        # Save metrics JSON
        metrics_full = {
            **metrics,
            "confusion_matrix": confusion_matrix.tolist(),
            "n_clusters": int(len(np.unique(y_clusters))),
            "n_samples": len(y_true),
            "n_ok_references": int(ok_reference_mask.sum()),
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(metrics_full, f, indent=2)

        # Save cluster composition
        cluster_stats.to_csv(fold_dir / "cluster_composition.csv", index=False)

        # Save confusion matrix plot
        self._plot_confusion_matrix(
            confusion_matrix,
            title=f"Stage 2 - Fold {fold_num}",
            save_path=fold_dir / "confusion_matrix.png",
        )

        # Save cluster visualization if features provided
        if x_values is not None:
            self._plot_clusters(
                x_values=x_values,
                y_clusters=y_clusters,
                y_true=y_true,
                ok_reference_mask=ok_reference_mask,
                title=f"Stage 2 Clusters - Fold {fold_num}",
                save_path=fold_dir / "cluster_visualization.png",
            )

    def save_aggregated_results(self, results: list[dict], fold_metrics: pd.DataFrame):
        """
        Save aggregated cross-validation results.

        Args:
            results: List of per-fold result dictionaries
            fold_metrics: DataFrame with metrics for each fold
        """
        # Calculate summary statistics
        summary = {
            "n_folds": len(results),
            "stage1_metrics": {
                "precision_mean": float(fold_metrics["stage1_precision"].mean()),
                "precision_std": float(fold_metrics["stage1_precision"].std()),
                "recall_mean": float(fold_metrics["stage1_recall"].mean()),
                "recall_std": float(fold_metrics["stage1_recall"].std()),
                "f1_mean": float(fold_metrics["stage1_f1"].mean()),
                "f1_std": float(fold_metrics["stage1_f1"].std()),
            },
            "stage2_metrics": {
                "precision_mean": float(fold_metrics["stage2_precision"].mean()),
                "precision_std": float(fold_metrics["stage2_precision"].std()),
                "recall_mean": float(fold_metrics["stage2_recall"].mean()),
                "recall_std": float(fold_metrics["stage2_recall"].std()),
                "f1_mean": float(fold_metrics["stage2_f1"].mean()),
                "f1_std": float(fold_metrics["stage2_f1"].std()),
            },
        }

        # Save summary JSON
        with open(self.aggregated_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save metrics by fold CSV
        fold_metrics.to_csv(self.aggregated_dir / "metrics_by_fold.csv", index=False)

        # Plot performance across folds
        self._plot_performance_comparison(
            fold_metrics, save_path=self.aggregated_dir / "performance_plots.png"
        )

    def _plot_confusion_matrix(self, cm: NDArray, title: str, save_path: Path):
        """Plot and save confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["OK", "NOK"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_clusters(
        self,
        x_values: NDArray,
        y_clusters: NDArray,
        y_true: NDArray,
        ok_reference_mask: NDArray,
        title: str,
        save_path: Path,
    ):
        """Plot cluster visualization using dimensionality reduction."""
        from sklearn.decomposition import PCA

        # Reduce to 2D for visualization
        if x_values.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            x_2d = pca.fit_transform(x_values)
            explained_var = pca.explained_variance_ratio_.sum()
        else:
            x_2d = x_values
            explained_var = 1.0

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Colored by cluster
        scatter1 = ax1.scatter(
            x_2d[:, 0],
            x_2d[:, 1],
            c=y_clusters,
            cmap="tab10",
            alpha=0.6,
            edgecolors="k",
            linewidths=0.5,
        )
        # Mark OK references with red border
        ok_refs = x_2d[ok_reference_mask]
        ax1.scatter(
            ok_refs[:, 0],
            ok_refs[:, 1],
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            s=100,
            label="OK Reference",
        )
        ax1.set_title(f"Clusters (explained var: {explained_var:.1%})")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        ax1.legend()
        plt.colorbar(scatter1, ax=ax1, label="Cluster ID")

        # Plot 2: Colored by ground truth
        y_binary = (y_true > 0).astype(int)
        scatter2 = ax2.scatter(
            x_2d[:, 0],
            x_2d[:, 1],
            c=y_binary,
            cmap="RdYlGn_r",
            alpha=0.6,
            edgecolors="k",
            linewidths=0.5,
        )
        ax2.set_title("Ground Truth")
        ax2.set_xlabel("Component 1")
        ax2.set_ylabel("Component 2")
        plt.colorbar(scatter2, ax=ax2, label="Class", ticks=[0, 1])

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_performance_comparison(self, fold_metrics: pd.DataFrame, save_path: Path):
        """Plot performance metrics across folds."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        metrics = ["precision", "recall", "f1"]
        stages = ["stage1", "stage2"]

        for i, stage in enumerate(stages):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                col_name = f"{stage}_{metric}"

                # Box plot
                ax.boxplot([fold_metrics[col_name]], labels=[stage.upper()])
                ax.scatter(
                    [1] * len(fold_metrics),
                    fold_metrics[col_name],
                    alpha=0.5,
                    s=50,
                )

                # Add mean line
                mean_val = fold_metrics[col_name].mean()
                ax.axhline(mean_val, color="r", linestyle="--", alpha=0.7)

                ax.set_ylabel(metric.capitalize())
                ax.set_title(f"{stage.upper()} - {metric.capitalize()}")
                ax.set_ylim([0, 1.05])
                ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    def get_run_directory(self) -> Path:
        """Get the main run directory path."""
        return self.run_dir
