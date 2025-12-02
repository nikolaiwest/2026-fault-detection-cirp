"""Stage 1 anomaly detection models."""

from .base import Stage1Model
from .auto_encoder import AutoEncoder
from .ecod import ECOD
from .hbos import HBOS
from .isolation_forest import IsolationForest
from .k_nearest_neighbors import KNearestNeighbors
from .local_outlier_factor import LocalOutlierFactor
from .one_class_svm import OneClassSVM

# Model registry for easy access
STAGE1_MODELS = {
    "auto_encoder": AutoEncoder,
    "ecod": ECOD,
    "hbos": HBOS,
    "isolation_forest": IsolationForest,
    "k_nearest_neighbors": KNearestNeighbors,
    "local_outlier_factor": LocalOutlierFactor,
    "one_class_svm": OneClassSVM,
}

__all__ = [
    "AutoEncoder",
    "ECOD",
    "HBOS",
    "IsolationForest",
    "KNearestNeighbors",
    "LocalOutlierFactor",
    "OneClassSVM",
    "STAGE1_MODELS",
]
