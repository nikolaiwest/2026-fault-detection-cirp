"""
Base class for Stage 2 clustering models.

Supports both sktime (time-series aware) and sklearn (fast & simple) models.
"""

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from src.data.config_loader import load_model_config
from src.utils import get_logger

logger = get_logger(__name__)


class Stage2Model(ABC):
    """Abstract base class for Stage 2 clustering models."""

    stage_number = 2  # Stage 2 models

    def __init__(self, n_clusters: int, random_state: int, metric: str = "euclidean"):
        """
        Initialize Stage 2 model.

        Args:
            n_clusters: Number of clusters (may be ignored by density-based methods)
            random_state: Random seed for reproducibility
            metric: Distance metric ('euclidean', 'dtw', 'msm', etc.)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.model = None

        # Validate metric support
        if metric not in self.supported_metrics:
            logger.warning(
                f"{self.name} does not support metric '{metric}'. "
                f"Supported metrics: {self.supported_metrics}. "
                f"Falling back to '{self.supported_metrics[0]}'"
            )
            self.metric = self.supported_metrics[0]

        # Load model-specific hyperparameters
        try:
            self.hyperparams = load_model_config(
                stage_number=self.stage_number,
                model_name=self.model_name,
            )
            logger.debug(
                f"Loaded hyperparameters for {self.model_name}: {self.hyperparams}"
            )
        except (FileNotFoundError, ValueError):
            logger.warning(
                f"No hyperparameters found for {self.model_name}, using defaults"
            )
            self.hyperparams = {}

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name used for config lookup (e.g., 'kmeans', 'dbscan')."""
        pass

    @property
    def supported_metrics(self) -> list[str]:
        """
        List of supported distance metrics.

        Override in subclasses to specify which metrics are supported.

        Returns:
            List of metric names (e.g., ['euclidean', 'dtw', 'msm'])
        """
        return ["euclidean"]  # Default: only Euclidean

    @abstractmethod
    def fit_predict(self, x_values: NDArray) -> NDArray:
        """
        Fit clustering model and return cluster assignments.

        Args:
            x_values: Feature matrix (n_samples, n_features)

        Returns:
            y_clusters: Cluster assignments (n_samples,)
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        return self.__class__.__name__
