"""TimeSeriesKMedoids from sktime."""

from numpy.typing import NDArray

from .base import Stage2Model


class TimeSeriesKMedoids(Stage2Model):
    """Time Series K-Medoids from sktime."""

    model_name = "sktime_kmedoids"

    @property
    def supported_metrics(self) -> list[str]:
        """Supports multiple distance metrics."""
        return ["euclidean", "dtw", "msm", "erp"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit TimeSeriesKMedoids and return cluster assignments."""
        try:
            from sktime.clustering.k_medoids import TimeSeriesKMedoids as SKTimeKMedoids
        except ImportError:
            raise ImportError(
                "sktime is required for this model. " "Install with: pip install sktime"
            )

        self.model = SKTimeKMedoids(
            n_clusters=self.n_clusters,
            metric=self.metric,
            random_state=self.random_state,
            **self.hyperparams,
        )

        return self.model.fit_predict(x_values)
