"""One-Class SVM implementation for Stage 1."""

from numpy.typing import NDArray
from pyod.models.ocsvm import OCSVM as PyODOCSVM

from .base import Stage1Model


class OneClassSVM(Stage1Model):
    """One-Class SVM anomaly detection model."""

    model_name = "one_class_svm"

    def __init__(self, contamination: float, random_state: int):
        """Initialize One-Class SVM model."""
        super().__init__(contamination, random_state)

    def fit(self, x_values: NDArray) -> None:
        """Train One-Class SVM model."""
        self.model = PyODOCSVM(
            contamination=self.contamination,
            **self.hyperparams,
        )
        self.model.fit(x_values)

    def predict(self, x_values: NDArray) -> NDArray:
        """Predict anomalies (0=OK, 1=NOK)."""
        return self.model.predict(x_values)

    def decision_function(self, x_values: NDArray) -> NDArray:
        """Get anomaly scores."""
        return self.model.decision_scores_
