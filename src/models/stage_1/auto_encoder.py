"""Autoencoder implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.auto_encoder import AutoEncoder as PyODAutoEncoder

from .base import Stage1Model


class AutoEncoder(Stage1Model):
    """AutoEncoder anomaly detection model."""

    model_name = "auto_encoder"

    def __init__(self, contamination: float, random_state: int):
        """Initialize AutoEncoder model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train AutoEncoder model."""
        self.model = PyODAutoEncoder(
            contamination=self.contamination,
            random_state=self.random_state,  # AE needs random_state
            **self.hyperparams,
        )
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
