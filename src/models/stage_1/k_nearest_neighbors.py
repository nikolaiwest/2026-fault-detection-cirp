"""K-Nearest Neighbors implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.knn import KNN as PyODKNN

from .base import Stage1Model


class KNearestNeighbors(Stage1Model):
    """KNN anomaly detection model."""

    model_name = "k_nearest_neighbors"

    def __init__(self, contamination: float, random_state: int):
        """Initialize KNN model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train KNN model."""
        self.model = PyODKNN(contamination=self.contamination, **self.hyperparams)
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
