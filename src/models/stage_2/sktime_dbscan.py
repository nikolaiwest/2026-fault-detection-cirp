"""TimeSeriesDBSCAN from sktime."""

from numpy.typing import NDArray

from .base import Stage2Model


class TimeSeriesDBSCAN(Stage2Model):
    """Time Series DBSCAN from sktime (density-based clustering)."""

    model_name = "sktime_dbscan"

    def __init__(self, n_clusters: int, random_state: int, metric: str = "euclidean"):
        """
        Initialize DBSCAN.

        Note: n_clusters is ignored by DBSCAN (density-based).
        """
        super().__init__(n_clusters, random_state, metric)

    @property
    def supported_metrics(self) -> list[str]:
        """Supports multiple distance metrics."""
        return ["euclidean", "dtw", "msm"]

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """
        Fit TimeSeriesDBSCAN and return cluster assignments.

        Note: sktime DBSCAN has a non-standard API:
        - Uses 'distance' parameter instead of 'metric'
        - Does not support fit_predict(), only fit()
        - Returns labels via labels_ attribute
        """
        try:
            from sktime.clustering.dbscan import TimeSeriesDBSCAN as SKTimeDBSCAN
        except ImportError:
            raise ImportError(
                "sktime is required for this model. " "Install with: pip install sktime"
            )

        # Initialize model with sktime's 'distance' parameter
        self.model = SKTimeDBSCAN(
            distance=self.metric,  # sktime calls it 'distance', not 'metric'
            **self.hyperparams,  # eps, min_samples, algorithm, etc.
        )

        # sktime DBSCAN only supports fit(), not fit_predict()
        self.model.fit(x_values)

        # Get cluster assignments from labels_ attribute
        return self.model.labels_
