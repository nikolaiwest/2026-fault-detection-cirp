import numpy as np
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data import load_class_config, run_data_pipeline


def test_anomaly_detection():
    """Test 5 PyOD models on top10 detectable classes."""

    print("=" * 70)
    print("Stage 1: Anomaly Detection with 5 PyOD Models")
    print("Testing on top10 most detectable fault classes")
    print("=" * 70)

    # Define 5 anomaly detectors
    models = {
        "IsolationForest": IForest(contamination=0.01, random_state=42),
        "OC-SVM": OCSVM(contamination=0.01),
        "LOF": LOF(contamination=0.01),
        "KNN": KNN(contamination=0.01),
        "COPOD": COPOD(contamination=0.01),
    }

    # Load data with top10 classes
    print("\n[1/5] Loading data (top10 classes)...")
    data = run_data_pipeline(
        force_reload=False,
        target_ok_ratio=0.99,
        classes_to_keep=load_class_config("top5"),
    )

    # Prepare arrays
    X = np.array(data["torque_values"])
    y = np.array(data["labels"])

    # Convert to binary: 0=normal, 1=anomaly
    y_binary = (y > 0).astype(int)

    print(f"Data shape: {X.shape}")
    print(f"Normal samples: {(y_binary == 0).sum()}")
    print(f"Anomaly samples: {(y_binary == 1).sum()}")
    print(f"Classes used: {load_class_config('top10')}")

    # Apply PAA
    print("\n[2/5] Applying PAA compression...")
    X_paa = apply_paa(X, n_segments=200)
    print(f"Compressed shape: {X_paa.shape}")

    # Train/Val split
    print("\n[3/5] Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_paa, y_binary, test_size=0.3, random_state=42, stratify=y_binary
    )

    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")

    # Test all models
    print("\n[4/5] Training and evaluating models...")
    print("=" * 70)

    results = {}

    for name, model in models.items():
        print(f"\n{name}")
        print("-" * 50)

        # Train (without labels!)
        model.fit(X_train)

        # Predict on validation
        y_pred = model.predict(X_val)  # 0=normal, 1=anomaly

        # Evaluate
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        fpr = (y_pred[y_val == 0] == 1).sum() / (y_val == 0).sum()

        results[name] = {
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "fpr": fpr,
        }

        print(f"  F1:        {f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  FPR:       {fpr:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("[5/5] Summary: Model Comparison")
    print("=" * 70)

    print(f"\n{'Model':<18s} | F1    | Precision | Recall | FPR")
    print("-" * 70)

    for name, metrics in results.items():
        print(
            f"{name:<18s} | {metrics['f1_score']:.3f} | "
            f"{metrics['precision']:.3f}     | {metrics['recall']:.3f}  | {metrics['fpr']:.3f}"
        )

    # Average
    avg_f1 = np.mean([m["f1_score"] for m in results.values()])
    avg_precision = np.mean([m["precision"] for m in results.values()])
    avg_recall = np.mean([m["recall"] for m in results.values()])
    avg_fpr = np.mean([m["fpr"] for m in results.values()])

    print("-" * 70)
    print(
        f"{'AVERAGE':<18s} | {avg_f1:.3f} | "
        f"{avg_precision:.3f}     | {avg_recall:.3f}  | {avg_fpr:.3f}"
    )

    # Best model
    best_model = max(results.items(), key=lambda x: x[1]["f1_score"])
    print(f"\nBest Model: {best_model[0]} (F1={best_model[1]['f1_score']:.3f})")

    # Target check
    print("\n" + "=" * 70)
    print("Target Achievement")
    print("=" * 70)
    print(f"Paper target: F1 ≥ 0.85")

    if best_model[1]["f1_score"] >= 0.85:
        print(f"ACHIEVED: F1={best_model[1]['f1_score']:.3f}")
    else:
        print(
            f"NOT MET: F1={best_model[1]['f1_score']:.3f} (need {0.85 - best_model[1]['f1_score']:.3f} more)"
        )


if __name__ == "__main__":
    test_anomaly_detection()
