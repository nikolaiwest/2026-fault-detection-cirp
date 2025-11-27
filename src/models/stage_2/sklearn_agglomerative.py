"""Agglomerative Clustering from sklearn."""

from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from .base import Stage2Model


class AgglomerativeSklearn(Stage2Model):
    """Agglomerative hierarchical clustering from sklearn."""

    model_name = "sklearn_agglomerative"

    @property
    def supported_metrics(self) -> list[str]:
        """Supports various linkage metrics."""
        return ["euclidean", "manhattan", "cosine"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit Agglomerative Clustering and return cluster assignments."""
        self.model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric=self.metric if self.metric != "euclidean" else None,
            **self.hyperparams,  # linkage, etc.
        )

        return self.model.fit_predict(x_values)
