from .isolation_forest import IsolationForest
from .one_class_svm import OneClassSVM

# Model registry for easy access
STAGE1_MODELS = {
    "isolation_forest": IsolationForest,
    "one_class_svm": OneClassSVM,
}

__all__ = ["STAGE1_MODELS"]
