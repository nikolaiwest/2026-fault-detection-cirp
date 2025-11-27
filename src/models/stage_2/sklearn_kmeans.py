"""Standard KMeans from sklearn (fast, Euclidean only)."""

from numpy.typing import NDArray
from sklearn.cluster import KMeans as SklearnKMeans

from .base import Stage2Model


class KMeansSklearn(Stage2Model):
    """Standard KMeans from sklearn (fast, Euclidean distance only)."""

    model_name = "sklearn_kmeans"

    @property
    def supported_metrics(self) -> list[str]:
        """Only supports Euclidean distance."""
        return ["euclidean"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit sklearn KMeans and return cluster assignments."""
        self.model = SklearnKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            **self.hyperparams,
        )

        return self.model.fit_predict(x_values)
