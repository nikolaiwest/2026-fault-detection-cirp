"""Local Outlier Factor implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.lof import LOF as PyODLOF

from .base import Stage1Model


class LocalOutlierFactor(Stage1Model):
    """Local Outlier Factor anomaly detection model."""

    model_name = "local_outlier_factor"

    def __init__(self, contamination: float, random_state: int):
        """Initialize LOF model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train LOF model."""
        self.model = PyODLOF(contamination=self.contamination, **self.hyperparams)
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
