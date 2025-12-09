"""BIRCH clustering from sklearn (hierarchical)."""

from numpy.typing import NDArray
from sklearn.cluster import Birch as SklearnBirch

from .base import Stage2Model


class BirchSklearn(Stage2Model):
    """BIRCH clustering from sklearn (hierarchical, memory-efficient)."""

    model_name = "sklearn_birch"

    @property
    def supported_metrics(self) -> list[str]:
        """BIRCH uses Euclidean distance."""
        return ["euclidean"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit BIRCH and return cluster assignments."""
        self.model = SklearnBirch(n_clusters=self.n_clusters, **self.hyperparams)

        return self.model.fit_predict(x_values)
