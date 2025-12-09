"""
Stage 1: Over-sensitive anomaly detection.

Implements the first stage of the two-stage pipeline using one-class
anomaly detection with intentionally high contamination to maximize recall.
False positives are accepted and will be filtered in Stage 2.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.models.stage_1 import STAGE1_MODELS, Stage1Model
from src.utils import get_logger

logger = get_logger(__name__)


def run_stage1(
    x_values: NDArray,
    y_values: NDArray,
    model_name: str,
    contamination: float,
    random_state: int,
) -> dict:
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
        dict: Complete results including:
            - y_anomalies: Binary predictions
            - anomaly_scores: Anomaly scores
            - metrics: Dict with precision, recall, f1
            - confusion_matrix: 2x2 numpy array
            - y_true: Ground truth (for saving)

    Raises:
        ValueError: If model_name not found in STAGE1_MODELS registry
    """

    # Step 1: Log stage configuration
    logger.subsection("Stage 1: Anomaly Detection")
    logger.info(f"Model: {model_name}")
    logger.info(f"Contamination: {contamination:.1%}")
    logger.debug(f"Input shape: {x_values.shape}")
    logger.debug(f"Random state: {random_state}")

    # Step 2: Instantiate the "stage 1" model from the registry using model_name
    if model_name not in STAGE1_MODELS:
        available = list(STAGE1_MODELS.keys())
        logger.error(f"Unknown model '{model_name}'. Available models: {available}")
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    ModelClass = STAGE1_MODELS[model_name]
    model: Stage1Model = ModelClass(
        contamination=contamination, random_state=random_state
    )
    # Note: All remaining parameter are set via src/models/stage_1/hyperparameters.yml
    logger.debug(f"Instantiatied {ModelClass.__name__}")

    # Step 3: Log training data and label distributions (purely logging, for symmetry)
    n_ok = (y_values == 0).sum()
    n_nok = (y_values > 0).sum()
    logger.info(f"Training data: {len(x_values)} samples ({n_ok} OK, {n_nok} NOK)")
    logger.debug(f"Label distribution: {np.bincount(y_values)}")

    # Step 4: Fit anomaly detector
    logger.info("Training anomaly detector (unsupervised)")
    model.fit(x_values)
    logger.debug("Model training complete")

    # Step 5: Generate predictions
    logger.info("Generating predictions")
    y_anomalies = model.predict(x_values)
    anomaly_scores = model.decision_function(x_values)
    logger.debug("Predictions generated")

    # Step 6: Evaluate predictions and report detection statistics (again, just logging)
    n_ok = (y_anomalies == 0).sum()
    n_nok = (y_anomalies == 1).sum()
    logger.info(f"Detected: {n_ok} OK, {n_nok} NOK")
    logger.debug(f"Detection rate: {n_nok / len(y_anomalies):.1%}")

    # Calculate metrics
    y_binary = (y_values > 0).astype(int)
    prec = precision_score(y_binary, y_anomalies, zero_division=0)
    reca = recall_score(y_binary, y_anomalies, zero_division=0)
    f1_s = f1_score(y_binary, y_anomalies, zero_division=0)
    logger.info(f"Precision={prec:.3f}, Recall={reca:.3f}, F1={f1_s:.3f}")

    # Confusion matrix
    cm = confusion_matrix(y_binary, y_anomalies)
    logger.debug("Confusion Matrix:")
    logger.debug(f"             Predicted OK  Predicted NOK")
    logger.debug(f"Actual OK    {cm[0,0]:12d}  {cm[0,1]:13d}")
    logger.debug(f"Actual NOK   {cm[1,0]:12d}  {cm[1,1]:13d}")

    logger.info("Stage 1 complete")

    # Return structured results
    return {
        "y_anomalies": y_anomalies,
        "anomaly_scores": anomaly_scores,
        "y_true": y_values,
        "metrics": {
            "precision": float(prec),
            "recall": float(reca),
            "f1": float(f1_s),
        },
        "confusion_matrix": cm,
    }
