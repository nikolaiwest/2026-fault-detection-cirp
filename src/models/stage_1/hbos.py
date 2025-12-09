"""HBOS (Histogram-based Outlier Score) implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.hbos import HBOS as PyODHBOS

from .base import Stage1Model


class HBOS(Stage1Model):
    """HBOS anomaly detection model."""

    model_name = "hbos"

    def __init__(self, contamination: float, random_state: int):
        """Initialize HBOS model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train HBOS model."""
        self.model = PyODHBOS(contamination=self.contamination, **self.hyperparams)
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
