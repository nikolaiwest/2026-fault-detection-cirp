"""TimeSeriesKMeans from sktime (supports multiple metrics)."""

from numpy.typing import NDArray

from .base import Stage2Model


class TimeSeriesKMeansSktime(Stage2Model):
    """Time Series KMeans from sktime."""

    model_name = "sktime_kmeans"

    @property
    def supported_metrics(self) -> list[str]:
        """Supports Euclidean and various time-series metrics."""
        return ["euclidean", "dtw", "msm", "erp", "lcss", "twe"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit TimeSeriesKMeans and return cluster assignments."""
        try:
            from sktime.clustering.k_means import TimeSeriesKMeans
        except ImportError:
            raise ImportError(
                "sktime is required for this model. " "Install with: pip install sktime"
            )

        self.model = TimeSeriesKMeans(
            n_clusters=self.n_clusters,
            metric=self.metric,
            random_state=self.random_state,
            **self.hyperparams,
        )

        return self.model.fit_predict(x_values)
