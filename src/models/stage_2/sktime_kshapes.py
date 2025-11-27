"""TimeSeriesKShapes from sktime."""

from numpy.typing import NDArray

from .base import Stage2Model


class TimeSeriesKShapes(Stage2Model):
    """Time Series K-Shapes from sktime (shape-based clustering)."""

    model_name = "kshapes_sktime"

    @property
    def supported_metrics(self) -> list[str]:
        """K-Shapes uses shape-based distance (not configurable)."""
        return ["shape"]  # K-Shapes has its own distance

    def fit_predict(self, x_values: NDArray) -> NDArray:
        """Fit TimeSeriesKShapes and return cluster assignments."""
        try:
            from sktime.clustering.k_shapes import TimeSeriesKShapes as SKTimeKShapes
        except ImportError:
            raise ImportError(
                "sktime is required for this model. " "Install with: pip install sktime"
            )

        self.model = SKTimeKShapes(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            **self.hyperparams,
        )

        return self.model.fit_predict(x_values)
