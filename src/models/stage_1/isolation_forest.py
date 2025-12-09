"""Isolation Forest implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.iforest import IForest as PyODIForest

from .base import Stage1Model


class IsolationForest(Stage1Model):
    """Isolation Forest anomaly detection model."""

    model_name = "isolation_forest"

    def __init__(self, contamination: float, random_state: int):
        """Initialize Isolation Forest model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train Isolation Forest model."""
        self.model = PyODIForest(
            contamination=self.contamination,
            random_state=self.random_state,
            **self.hyperparams,
        )
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
