"""
Stage 2 clustering models registry.

Includes both sktime (time-series aware) and sklearn (fast) models.
Each model uses its default distance metric as specified in hyperparameters.yml.
"""

# sktime models (time-series aware, support DTW/MSM/etc.)
from .sktime_kmeans import TimeSeriesKMeansSktime
from .sktime_kmedoids import TimeSeriesKMedoids
from .sktime_dbscan import TimeSeriesDBSCAN
from .sktime_kshapes import TimeSeriesKShapes

# sklearn models (fast, Euclidean-focused)
from .sklearn_kmeans import KMeansSklearn
from .sklearn_dbscan import DBSCANSklearn
from .sklearn_birch import BirchSklearn
from .sklearn_agglomerative import AgglomerativeSklearn


# Model registry for Stage 2
# Use these names in your pipeline config (stage2.model_name)
STAGE2_MODELS = {
    # sktime models (recommended for time-series with DTW support)
    "sktime_kmeans": TimeSeriesKMeansSktime,  # TimeSeriesKMeans (default: DTW)
    "sktime_kmedoids": TimeSeriesKMedoids,  # TimeSeriesKMedoids (default: DTW)
    "sktime_dbscan": TimeSeriesDBSCAN,  # TimeSeriesDBSCAN (default: euclidean)
    "sktime_kshapes": TimeSeriesKShapes,  # TimeSeriesKShapes (shape-based)
    # sklearn models (fast alternatives, Euclidean only)
    "sklearn_kmeans": KMeansSklearn,  # Standard KMeans (fast, Euclidean)
    "sklearn_dbscan": DBSCANSklearn,  # Standard DBSCAN (fast, Euclidean)
    "sklearnbirch": BirchSklearn,  # BIRCH (hierarchical, memory-efficient)
    "sklearn_agglomerative": AgglomerativeSklearn,  # Agglomerative (hierarchical)
}


__all__ = [
    "STAGE2_MODELS",
    # sktime
    "TimeSeriesKMeansSktime",
    "TimeSeriesKMedoids",
    "TimeSeriesDBSCAN",
    "TimeSeriesKShapes",
    # sklearn
    "KMeansSklearn",
    "DBSCANSklearn",
    "BirchSklearn",
    "AgglomerativeSklearn",
]
