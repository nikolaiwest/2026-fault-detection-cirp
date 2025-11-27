"""
Stage 1: Over-sensitive anomaly detection.

Implements the first stage of the two-stage pipeline using one-class
anomaly detection with intentionally high contamination to maximize recall.
False positives are accepted and will be filtered in Stage 2.
"""

from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.stage_1 import STAGE1_MODELS
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_stage1(
    x_values: NDArray,
    y_true: NDArray,
    model_name: str,
    contamination: float,
    random_state: int,
) -> tuple[NDArray, NDArray]:
    """
    Stage 1: Over-sensitive anomaly detection.

    Uses one-class anomaly detection with intentionally high contamination
    parameter to maximize recall. Accepts elevated false positive rate,
    which will be filtered in Stage 2.

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels for evaluation only (not used in training)
        model_name: Name of anomaly detection model to use
        contamination: Expected fraction of anomalies (e.g., 0.02 = 2%)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (y_anomalies, anomaly_scores)
            - y_anomalies: Binary predictions (0=OK, 1=NOK)
            - anomaly_scores: Anomaly scores for each sample

    Raises:
        ValueError: If model_name not found in STAGE1_MODELS registry
    """
    logger.subsection("Stage 1: Anomaly Detection")
    logger.info(f"Model: {model_name}")
    logger.info(f"Contamination: {contamination:.1%}")
    logger.debug(f"Input shape: {x_values.shape}")
    logger.debug(f"Random state: {random_state}")

    # Get model from registry
    if model_name not in STAGE1_MODELS:
        available = list(STAGE1_MODELS.keys())
        logger.error(f"Unknown model '{model_name}'. Available models: {available}")
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    ModelClass = STAGE1_MODELS[model_name]
    logger.debug(f"Instantiating {ModelClass.__name__}")
    model = ModelClass(contamination=contamination, random_state=random_state)

    # Train anomaly detector (unsupervised - does not use y_true)
    logger.info("Training anomaly detector (unsupervised)")
    model.fit(x_values)
    logger.debug("Model training complete")

    # Get predictions
    logger.info("Generating predictions")
    y_anomalies = model.predict(x_values)
    anomaly_scores = model.decision_function(x_values)

    # Report detection statistics
    n_ok = (y_anomalies == 0).sum()
    n_nok = (y_anomalies == 1).sum()
    logger.info(f"Detected: {n_ok} OK samples, {n_nok} NOK samples")
    logger.debug(f"Detection rate: {n_nok / len(y_anomalies):.1%}")

    # Calculate evaluation metrics (ground truth comparison)
    y_binary = (y_true > 0).astype(int)

    precision = precision_score(y_binary, y_anomalies, zero_division=0)
    recall = recall_score(y_binary, y_anomalies, zero_division=0)
    f1 = f1_score(y_binary, y_anomalies, zero_division=0)

    logger.info(
        f"Performance: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}"
    )

    # Show confusion matrix (evaluation only, not used for decisions)
    cm = confusion_matrix(y_binary, y_anomalies)

    logger.debug("Confusion Matrix:")
    logger.debug(f"             Predicted OK  Predicted NOK")
    logger.debug(f"Actual OK    {cm[0,0]:12d}  {cm[0,1]:13d}")
    logger.debug(f"Actual NOK   {cm[1,0]:12d}  {cm[1,1]:13d}")

    logger.info("Stage 1 complete")

    return y_anomalies, anomaly_scores
