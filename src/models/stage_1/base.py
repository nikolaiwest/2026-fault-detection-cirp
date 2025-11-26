"""Base class for Stage 1 anomaly detection models."""

from abc import ABC, abstractmethod

from numpy.typing import NDArray

from src.data.config_loader import load_model_config


class Stage1Model(ABC):
    """Abstract base class for Stage 1 anomaly detection models."""

    stage_number = 1  # Stage 1 models

    def __init__(self, contamination: float, random_state: int):
        """
        Initialize Stage 1 model.

        Args:
            contamination: Expected fraction of anomalies (e.g., 0.02 = 2%)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None

        # Load model-specific hyperparameters
        self.hyperparams = load_model_config(
            stage_number=self.stage_number,
            model_name=self.model_name,
        )

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name used for config lookup (e.g., 'isolation_forest')."""
        pass

    @abstractmethod
    def fit(self, x_values: NDArray) -> None:
        """Train the anomaly detection model."""
        pass

    @abstractmethod
    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        pass

    @abstractmethod
    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        pass

    @property
    def name(self) -> str:
        """Human-readable model name for logging."""
        return self.__class__.__name__
