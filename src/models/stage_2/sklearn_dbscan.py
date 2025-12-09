"""Standard DBSCAN from sklearn."""

from numpy.typing import NDArray
from sklearn.cluster import DBSCAN as SklearnDBSCAN

from .base import Stage2Model


class DBSCANSklearn(Stage2Model):
    """Standard DBSCAN from sklearn (density-based)."""

    model_name = "sklearn_dbscan"

    def __init__(self, n_clusters: int, random_state: int, metric: str = "euclidean"):
        """Initialize DBSCAN (n_clusters ignored)."""
        super().__init__(n_clusters, random_state, metric)

    @property
    def supported_metrics(self) -> list[str]:
        """Supports many sklearn distance metrics."""
        return ["euclidean", "manhattan", "cosine", "chebyshev"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit sklearn DBSCAN and return cluster assignments."""
        self.model = SklearnDBSCAN(
            metric=self.metric, **self.hyperparams  # eps, min_samples
        )

        return self.model.fit_predict(x_values)
