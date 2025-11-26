"""Stage 1: Over-sensitive anomaly detection."""

from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix

from src.models.stage_1 import STAGE1_MODELS


def run_stage1(
    x_values: NDArray,
    y_true: NDArray,
    model_name: str,
    contamination: float,
    random_state: int,
) -> tuple[NDArray, NDArray]:
    """
    Stage 1: Over-sensitive anomaly detection.

    Uses Isolation Forest with intentionally high contamination parameter
    to maximize recall. Accepts elevated false positive rate, which will
    be filtered in Stage 2.

    Args:
        x_values: Feature matrix (n_samples, n_features)
        y_true: Ground truth labels for evaluation only (not used in training)
        contamination: Expected fraction of anomalies (e.g., 0.02 = 2%)
        random_state: Random seed for reproducibility

    Returns:
        tuple: (y_anomalies, anomaly_scores)
            - y_anomalies: Binary predictions (0=OK, 1=NOK)
            - anomaly_scores: Anomaly scores for each sample
    """
    print("\n" + "=" * 70)
    print("STAGE 1: ANOMALY DETECTION")
    print("=" * 70)

    # Get model from registry
    if model_name not in STAGE1_MODELS:
        available = list(STAGE1_MODELS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    ModelClass = STAGE1_MODELS[model_name]
    model = ModelClass(contamination=contamination, random_state=random_state)

    # Train anomaly detector (unsupervised - does not use y_true)
    model.fit(x_values)

    # Get predictions
    y_anomalies = model.predict(x_values)
    anomaly_scores = model.decision_function(x_values)

    # Report results
    n_ok = (y_anomalies == 0).sum()
    n_nok = (y_anomalies == 1).sum()
    print(f"Detected: {n_ok} OK, {n_nok} NOK")

    # Show confusion matrix (evaluation only, not used for decisions)
    y_binary = (y_true > 0).astype(int)
    cm = confusion_matrix(y_binary, y_anomalies)
    print(f"\nConfusion Matrix:")
    print(f"             Predicted OK  Predicted NOK")
    print(f"Actual OK    {cm[0,0]:12d}  {cm[0,1]:13d}")
    print(f"Actual NOK   {cm[1,0]:12d}  {cm[1,1]:13d}")

    return y_anomalies, anomaly_scores
